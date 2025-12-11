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

def backtest_and_visualize(model_path, data_path, coin_name="ETH"):
    if not os.path.exists(model_path):
        print(f"❌ Model yok: {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"❌ Veri yok: {data_path}")
        return

    # 1. Model Hazırlığı
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

    # 2. Veri Yükleme
    dataset = CryptoDataset(data_path, seq_len=MODEL_CONFIG['seq_len'])
    loader = DataLoader(dataset, batch_size=512, shuffle=False) # Hızlı olsun diye 512

    print(f"🧪 SİMÜLASYON BAŞLIYOR (Akıllı Cüzdan Modu): {coin_name}...")

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()

            # Target Scaling Geri Alma (100'e bölüyoruz)
            real_preds = outputs / 100.0

            all_preds.extend(real_preds.cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())

    # 3. AKILLI SİMÜLASYON (HATAYI DÜZELTTİĞİMİZ YER)
    initial_balance = 1000
    balance = initial_balance
    balance_history = [balance]

    # Şu anki pozisyonumuz (0: Nakit, 1: Long, -1: Short)
    current_position = 0

    # Komisyon Oranı (Binance Futures Taker: %0.04)
    commission_rate = 0.0004

    trade_count = 0

    for i in range(len(all_preds)):
        pred = all_preds[i]     # Modelin tahmini (Örn: +0.02)
        actual = all_actuals[i] # Gerçek değişim (Örn: -0.01)

        # Modelin önerdiği yeni pozisyon
        # Basit eşik: 0'dan büyükse Long, küçükse Short
        target_position = 1 if pred > 0 else -1

        # --- KOMİSYON MANTIĞI ---
        cost = 0
        if target_position != current_position:
            # Pozisyon değiştiyse (Long->Short veya Short->Long) komisyon öde
            cost = commission_rate
            trade_count += 1
            current_position = target_position # Pozisyonu güncelle

        # --- KÂR/ZARAR HESABI ---
        # Elimizdeki pozisyona göre kâr/zarar (Short açtıysak düşüşten kazanırız)
        market_change = current_position * actual

        # Net Değişim = Piyasa Kazancı - Komisyon Maliyeti
        pnl = market_change - cost

        # Bakiyeyi güncelle
        balance = balance * (1 + pnl)
        balance_history.append(balance)

    # 4. Çizim
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # AI Cüzdanı
    ax1.plot(balance_history, color='#00ff00', linewidth=1.5, label='AI Bot (Akıllı Trade)')

    # Buy & Hold (Karşılaştırma)
    # Kümülatif getiri hesabı
    market_trend = [initial_balance]
    for ret in all_actuals:
        market_trend.append(market_trend[-1] * (1 + ret))

    ax1.plot(market_trend, color='cyan', alpha=0.5, linestyle='--', label='Buy & Hold (ETH Tutsa)')

    ax1.axhline(y=initial_balance, color='white', alpha=0.3)
    ax1.set_title(f"{coin_name} Test Sonucu | İşlem Sayısı: {trade_count}")
    ax1.set_ylabel("Bakiye ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Sonuç Yazısı
    final_balance = balance_history[-1]
    profit_pct = ((final_balance - initial_balance) / initial_balance) * 100

    res_color = 'green' if profit_pct > 0 else 'red'
    ax1.text(0.02, 0.95, f"SONUÇ: %{profit_pct:.2f} (${final_balance:.0f})",
             transform=ax1.transAxes, color=res_color, fontsize=14, fontweight='bold')

    plt.show()

# --- ÇALIŞTIR ---
# 6 Aylık modelin ve 6 Aylık ETH verin varsa:
backtest_and_visualize("ETH_TUNED_15m_MODEL.pth", "ETH_6Ay_15m_AI_Ready.csv", "ETH")