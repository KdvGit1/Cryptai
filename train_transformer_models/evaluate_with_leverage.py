import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ai_engine import CryptoDataset, CryptoTransformer
import os

# --- AYARLAR ---
MODEL_CONFIG = {
    'input_dim': 14,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'seq_len': 120
}

LEVERAGE = 20

def backtest_dynamic(model_path, data_path, coin_name="ETH"):
    if not os.path.exists(model_path):
        print(f"❌ Model yok: {model_path}")
        return

    # 1. Model ve Veri Yükleme (Standart)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryptoTransformer(
        input_dim=MODEL_CONFIG['input_dim'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers']
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    dataset = CryptoDataset(data_path, seq_len=MODEL_CONFIG['seq_len'])
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    print(f"🧪 DİNAMİK SİMÜLASYON: {coin_name} | {LEVERAGE}x")

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()
            real_preds = outputs / 100.0
            all_preds.extend(real_preds.cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())

    # --- 2. DİNAMİK SİNYAL ÜRETİMİ (Z-SCORE) ---
    # Modelin "Hepsi Pozitif" biasını kırmak için hareketli ortalama kullanıyoruz.

    preds_series = pd.Series(all_preds)

    # Son 50 mumluk tahminlerin ortalamasını al
    # Eğer model şu anki mum için ortalamadan DAHA YÜKSEK bir değer veriyorsa LONG
    # Ortalamanın ALTINDA kalıyorsa (hala pozitif olsa bile) gücü azalmıştır -> SHORT
    rolling_mean = preds_series.rolling(window=50).mean()
    rolling_std = preds_series.rolling(window=50).std()

    # Z-Score: (Değer - Ortalama) / Standart Sapma
    # Bu bize sinyalin "Normale göre ne kadar güçlü" olduğunu söyler.
    z_scores = (preds_series - rolling_mean) / (rolling_std + 1e-9)

    # 3. SİMÜLASYON
    initial_balance = 1000
    balance = initial_balance
    balance_history = [balance]
    current_position = 0
    trade_count = 0
    commission_rate = 0.0004
    is_liquidated = False

    # Z-Score Eşikleri (Bunlarla oynayabilirsin)
    # +1.0 standart sapmanın üzerindeyse AL
    # -1.0 standart sapmanın altındaysa SAT
    ENTRY_THRESHOLD = 0.5
    EXIT_THRESHOLD = -0.5

    for i in range(len(all_preds)):
        if is_liquidated:
            balance_history.append(0)
            continue

        if i < 50: # İlk 50 mumda ortalama hesaplanamaz, bekle
            balance_history.append(balance)
            continue

        actual = all_actuals[i]
        z_signal = z_scores[i] # Modelin gücü

        # --- KARAR MEKANİZMASI ---
        target_position = current_position

        if z_signal > ENTRY_THRESHOLD:
            target_position = 1 # LONG (Momentum artıyor)
        elif z_signal < EXIT_THRESHOLD:
            target_position = -1 # SHORT (Momentum düşüyor)

        # Maliyet ve PnL
        cost = 0
        if target_position != current_position:
            cost = commission_rate * LEVERAGE
            trade_count += 1
            current_position = target_position

        market_change = current_position * actual * LEVERAGE
        pnl = market_change - cost
        balance = balance * (1 + pnl)

        if balance <= 0:
            balance = 0
            is_liquidated = True

        balance_history.append(balance)

    # 4. Çizim
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Üst Grafik: Bakiye
    ax1.plot(balance_history, color='#00ff00' if balance > initial_balance else '#ff0044')
    ax1.set_title(f"Bakiye Grafiği | İşlem: {trade_count}")
    ax1.grid(True, alpha=0.2)

    final_balance = balance_history[-1]
    profit = ((final_balance - 1000)/1000)*100
    ax1.text(0.02, 0.95, f"SONUÇ: %{profit:.2f}", transform=ax1.transAxes, color='white', fontweight='bold')

    # Alt Grafik: Modelin Tahminleri vs Ortalama
    # Modelin ne düşündüğünü görmek için
    zoom = 200 # Son 200 mumu göster
    ax2.plot(all_preds[-zoom:], label='Model Tahmini', color='yellow', alpha=0.7)
    ax2.plot(rolling_mean.tolist()[-zoom:], label='Hareketli Ort.', color='cyan', linestyle='--')
    ax2.set_title("Model Tahminleri vs Kendi Ortalaması (Son 200 Mum)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- ÇALIŞTIR ---
# Hangi modelin varsa onu koy (1h veya 5m tavsiyemdir)
backtest_dynamic("ETH_TUNED_1h_MODEL.pth", "ETH_6Ay_1h_AI_Ready.csv", "ETH")