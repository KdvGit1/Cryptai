import ccxt
import pandas as pd
import json
import os
import numpy as np
import torch
from datetime import datetime
from train_transformer_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_transformer_models.ai_engine import CryptoTransformer

JSON_FILENAME = "live_market_data.json"

MODEL_MAP = {
    '5m' : 'train_transformer_models/finalized_models/ETH_TUNED_5m_MODEL.pth',
    '15m' : 'train_transformer_models/finalized_models/ETH_TUNED_15m_MODEL.pth',
    '1h' : 'train_transformer_models/finalized_models/ETH_TUNED_1h_MODEL.pth'
}

MODEL_CONFIG = {
    'input_dim': 14,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'seq_len': 120,
    'output_dim': 1
}

_MODEL_CACHE = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ai_model(timeframe):
    """
    İstenilen timeframe için doğru modeli yükler ve hafızada tutar.
    """
    global _MODEL_CACHE

    # 1. Cache Kontrolü
    if timeframe in _MODEL_CACHE:
        return _MODEL_CACHE[timeframe]

    # 2. Dosya Yolunu Bul
    if timeframe not in MODEL_MAP:
        print(f"❌ '{timeframe}' için tanımlı bir model yok! (MODEL_MAP'i kontrol et)")
        return None

    model_path = MODEL_MAP[timeframe]

    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return None

    print(f"🧠 '{timeframe}' için Yapay Zeka Modeli Yükleniyor: {model_path} ...")

    # 3. Modeli Oluştur
    try:
        model = CryptoTransformer(
            input_dim=MODEL_CONFIG['input_dim'],
            d_model=MODEL_CONFIG['d_model'],
            nhead=MODEL_CONFIG['nhead'],
            num_layers=MODEL_CONFIG['num_layers']
            # output_dim parametresi class'ında varsa ekle
        ).to(_DEVICE)

        # 4. Ağırlıkları Yükle
        state_dict = torch.load(model_path, map_location=_DEVICE, weights_only=True)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)

        model.eval()  # Sınav modu

        # 5. Cache'e At
        _MODEL_CACHE[timeframe] = model
        return model

    except Exception as e:
        print(f"❌ Model yükleme hatası ({timeframe}): {e}")
        return None

def get_available_exhanges():
    return ccxt.exchanges

def get_all_pairs(exchange_name="binance"):
    exchange_name = exchange_name.lower()
    exchange = getattr(ccxt, exchange_name)()
    exchange.load_markets()
    pair_list = [ symbol for symbol in exchange.symbols
                  if symbol.endswith(("/USDT"))
                  ]
    print(pair_list)
    return pair_list

def calculate_needed_months(timeframe_str, candle_count=500):
    """
    İstenilen mum sayısı için kaç ay geriye gidilmesi gerektiğini hesaplar.
    Güvenlik payı olarak %10 fazlasını hesaplar.
    """
    # 1. Zaman dilimini dakikaya çevir
    tf_minutes = 0
    if timeframe_str == '1h':
        tf_minutes = 60
    elif timeframe_str == '15m':
        tf_minutes = 15
    elif timeframe_str == '5m':
        tf_minutes = 5
    else:
        # Bilinmeyen bir time frame ise varsayılan 1 ay döndür
        return 1.0

        # 2. Toplam gereken dakika (500 mum * periyot)
    total_minutes = candle_count * tf_minutes

    # 3. Bir aydaki dakika sayısı (30 gün * 24 saat * 60 dk)
    minutes_in_month = 30 * 24 * 60

    # 4. Oranla ve %10 güvenlik payı ekle (Veri eksik gelmesin)
    months_needed = (total_minutes / minutes_in_month) * 1.1

    return months_needed


def scan_market(timeframe, exchange_name="binance"):
    # 1. Kaç ay (float) gerektiğin hesapla
    # Örn: 1h için yaklaşık 0.7, 5m için 0.06 döner.
    months_to_fetch = calculate_needed_months(timeframe, candle_count=500)

    print(f"🛠️ {timeframe} için son 500 mum yaklaşık {months_to_fetch:.4f} ay ediyor.")

    ai_model = load_ai_model(timeframe)

    if ai_model is None:
        print("⚠️ Model YÜKLENEMEDİ! Tahminler 0.0 olacak.")

    all_pairs = get_all_pairs(exchange_name)
    market_data_storage = {}

    for pair in all_pairs:
        try:
            # get_crypto_history fonksiyonuna hesaplanan ayı gönderiyoruz
            df = get_crypto_history(
                symbol=pair,
                timeframe=timeframe,
                months_back=months_to_fetch,
                exchange_name=exchange_name
            )

            if len(df) < 120:
                print(f"{pair} yetersiz veriye sahip. Atlanıyor.")
                continue

            # ELDE EDİLEN VERİ KONTROLÜ
            # Bazen hesapladığımızdan fazla gelebilir, tam 500'ü kesip alalım (son 500)
            if len(df) > 500:
                raw_df = df.tail(500)
            else:
                raw_df = df

            print(f"{pair} -> {len(raw_df)} mum alındı. İşleme hazır.")

            df_display, df_ai = prepare_dual_dataframes(raw_df)

            ai_prediction_value = 0.0 #şimdilik böyle

            if ai_model is not None and len(df_ai) >= MODEL_CONFIG['seq_len']:
                # Model son 120 mumu istiyor
                input_data = df_ai.tail(MODEL_CONFIG['seq_len']).values
                # Eğer veride hala NaN veya Sonsuz varsa bu coini atla
                if np.isnan(input_data).any() or np.isinf(input_data).any():
                    print(f"⚠️ {pair}: Veri bozuk (NaN/Inf tespit edildi), atlanıyor.")
                    continue
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(_DEVICE)

                #DEBUG RSI ORTALAMSI
                print(f"Debug Input Mean: {input_data.mean():.4f}")
                with torch.no_grad():
                    output = ai_model(input_tensor).item()

                # REVERSE SCALING
                ai_prediction_value = output / 100.0

                print(f"   🤖 AI Tahmini ({timeframe}): %{ai_prediction_value * 100:.4f}")

            export_df = df_display.copy()
            export_df.reset_index(inplace=True)
            export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Veriyi Sözlüğe Ekle
            market_data_storage[pair] = {
                # Kullanıcıya göstermek için son 1 mumu (veya son 10) kaydetmek yeterli
                # 'records' formatı: [{col: val}, {col: val}]
                "last_indicators": export_df.tail(5).to_dict(orient='records'),

                # AI tahmini
                "ai_prediction": ai_prediction_value,

                # AI için hazırlanan verinin son satırı (Debug veya Log için)
                # "ai_input_data": df_ai.tail(1).to_dict(orient='records'),

                "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"❌ {pair} hatası: {e}")
            continue

    if market_data_storage:
        print(f"\n💾 Veriler '{JSON_FILENAME}' dosyasına yazılıyor...")
        with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(market_data_storage, f, indent=4, ensure_ascii=False)

        print("🏁 İşlem Başarıyla Tamamlandı.")
    else:
        print("⚠️ Kaydedilecek veri bulunamadı.")

if __name__ == "__main__":
    scan_market("1h","binance")
    scan_market("15m","binance")
    scan_market("5m","bitget")