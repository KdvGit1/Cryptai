import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta

# ==========================================
# 1. VERİ ÇEKME KATMANI
# ==========================================
def get_crypto_history(symbol, timeframe, months_back):
    """Borsadan ham mum verilerini çeker."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    now = datetime.now()
    start_date = now - timedelta(days=30 * months_back)
    since = int(start_date.timestamp() * 1000)

    print(f"🚀 BAŞLIYOR: {symbol} - {timeframe}")
    all_candles = []

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break

            all_candles += candles
            last_candle_time = candles[-1][0]
            since = last_candle_time + 1

            # İlerleme göstergesi
            if len(all_candles) % 5000 == 0:
                print(f"📦 Çekilen: {len(all_candles)} mum...")

            if last_candle_time >= exchange.milliseconds():
                print("✅ Veri çekimi tamamlandı.")
                break

        except Exception as e:
            print(f"❌ Hata: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    return df

# ==========================================
# 2. İNDİKATÖR HESAPLAMA KATMANI
# ==========================================
def add_smart_indicators(df):
    """Yapay zeka için gerekli matematiksel hesaplamaları yapar."""
    df = df.copy()

    # --- Hacim ve Heikin Ashi ---
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + df['HA_Close'].iloc[i-1]) / 2)
    df['HA_Open'] = ha_open

    # Not: HA High/Low AI için ham fiyat olduğundan oranlamak lazım,
    # şimdilik indikatör hesaplarında kullanmak için tutuyoruz.

    # Hacim Analizi
    df['Vol_SMA_20'] = talib.SMA(df['Volume'], timeperiod=20)
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA_20']
    df['Vol_Spike'] = (df['Vol_Ratio'] > 2.0).astype(int)

    # --- Hareketli Ortalamalar (Distance - Uzaklık) ---
    sma_50 = talib.SMA(df['Close'], timeperiod=50)
    df['Dist_SMA_50'] = (df['Close'] - sma_50) / sma_50

    ema_200 = talib.EMA(df['Close'], timeperiod=200)
    df['Dist_EMA_200'] = (df['Close'] - ema_200) / ema_200

    # --- Bollinger Bands (%B ve Width) ---
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower)
    df['BB_Width'] = (upper - lower) / middle

    # --- Osilatörler ---
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14) / 100.0

    macd, macdsignal, macdhist = talib.MACD(df['Close'])
    df['MACD_Norm'] = macd / df['Close']

    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']

    # Saat 23:00 (23) ile 00:00 (0) sayısal olarak uzaktır ama zamansal olarak yakındır.
    # Sin/Cos dönüşümü bu yakınlığı modele öğretir.

    # 24 Saatlik Döngü
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # 7 Günlük Döngü (Hafta sonu etkisi için)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # --- KRİTİK EKLEME: Log Returns (Hedef Değişken) ---
    # Modelin neyi tahmin edeceğini (veya geçmiş hareketi) bilmesi için
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    return df

# ==========================================
# 3. AYRIŞTIRMA VE KAYDETME KATMANI (YENİ)
# ==========================================
def prepare_dual_dataframes(df):
    """
    Hesaplanmış DataFrame'i alır, NaN'ları temizler ve ikiye ayırır.
    Return: (df_original, df_ai)
    """
    # 1. Önce hesaplamaları yap
    df_calculated = add_smart_indicators(df)

    # 2. NaN (Boş) satırları temizle
    # İndikatörler (EMA 200 gibi) ilk 200 satırı boş bırakır.
    # Bunları silmezsek AI hata verir.
    df_clean = df_calculated.dropna()

    print(f"🧹 Temizlik: İlk {len(df_calculated) - len(df_clean)} satır (NaN) silindi.")

    # 3. Sütunları Seç ve Ayır

    # A) Orijinal (Vitrin) Verisi: Fiyatlar, Tarih, Hacim
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_original = df_clean[original_cols].copy()

    # B) AI (Mutfak) Verisi: Sadece Oranlar, Yüzdeler, 0-1 arası değerler
    # Ham fiyatları (Open, High vb.) BURAYA ALMIYORUZ.
    ai_cols = [
        'Log_Ret',      # En önemli veri (Değişim oranı)
        'RSI',
        'Dist_SMA_50',
        'Dist_EMA_200',
        'BB_PctB',
        'BB_Width',
        'MACD_Norm',
        'ATR_Pct',
        'Vol_Ratio',
        'Vol_Spike',
        'Hour_Sin',
        'Hour_Cos',
        'Day_Sin',
        'Day_Cos'
    ]
    df_ai = df_clean[ai_cols].copy()

    return df_original, df_ai

def workflow_runner(coin_name,desired_month, desired_timeframes):
    """Tüm süreci yöneten ana fonksiyon."""

    for tf in desired_timeframes:
        # 1. Veriyi Çek
        df_raw = get_crypto_history(f"{coin_name.upper()}/USDT", tf, desired_month)

        # 2. Hesapla ve İkiye Böl
        df_orig, df_ai = prepare_dual_dataframes(df_raw)

        # 3. Kontrol Et (Satır sayıları eşit mi?)
        if len(df_orig) == len(df_ai):
            print(f"✅ Eşleşme Başarılı: İki tabloda da {len(df_orig)} satır var.")
        else:
            print("❌ HATA: Satır sayıları tutmuyor!")

        # 4. Kaydet
        file_orig = f"{coin_name}_{desired_month}Ay_{tf}_ORIGINAL.csv"
        file_ai = f"{coin_name}_{desired_month}Ay_{tf}_AI_Ready.csv"

        df_orig.to_csv(file_orig)
        df_ai.to_csv(file_ai)

        print(f"💾 Kaydedildi:\n  -> {file_orig}\n  -> {file_ai}")
        print("-" * 40)

# --- ÇALIŞTIRMA ---
workflow_runner("ETH",6, ('5m', '15m', '1h'))