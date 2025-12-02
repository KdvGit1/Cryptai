import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from ai_engine import CryptoDataset, CryptoTransformer

# --- AYARLAR ---
# BTC Modelinin mimarisiyle BİREBİR AYNI olmalı
CONFIG = {
    'input_dim': 14,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'seq_len': 120,
    'batch_size': 256,

    # FINE TUNE AYARLARI
    'epochs': 10,          # Çok az epoch (Hızlı adaptasyon)
    'learning_rate': 0.000005 # ÇOK DÜŞÜK HIZ (Bilgiyi silmemek için)
}

def fine_tune_model(pretrained_model_path, target_coin, target_month, timeframe):
    # ETH Verisi
    csv_path = f"{target_coin}_{target_month}Ay_{timeframe}_AI_Ready.csv"
    # Yeni Kayıt İsmi (ETH için özelleşmiş model)
    new_model_name = f"{target_coin}_TUNED_{timeframe}_MODEL.pth"

    print(f"\n🔧 FINE-TUNING BAŞLIYOR: {pretrained_model_path} -> {target_coin} Verisiyle eğitiliyor...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset Hazırla
    dataset = CryptoDataset(csv_path, seq_len=CONFIG['seq_len'])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False)

    # 2. BTC Modelini Yükle
    model = CryptoTransformer(
        input_dim=CONFIG['input_dim'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=0.2
    ).to(device)

    if os.path.exists(pretrained_model_path):
        print("✅ Önceden eğitilmiş BTC modeli yükleniyor...")
        state_dict = torch.load(pretrained_model_path, map_location=device,weights_only=True)
        # DataParallel temizliği
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print(f"❌ HATA: {pretrained_model_path} bulunamadı!")
        return

    # Çoklu GPU varsa
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 3. Eğitim Ayarları (Huber Loss Devam)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate']) # Çok düşük LR

    # 4. Kısa Eğitim Döngüsü
    best_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            scaled_target = by * 100.0
            loss = criterion(output.squeeze(), scaled_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                output = model(bx)
                scaled_target = by * 100.0
                test_loss += criterion(output.squeeze(), scaled_target).item()

        avg_test = test_loss / len(test_loader)

        print(f"Epoch {epoch+1} | Tuned Test Loss: {avg_test:.5f}")

        if avg_test < best_loss:
            best_loss = avg_test
            torch.save(model.state_dict(), new_model_name)

    print(f"🎉 Fine-Tuning Bitti! Yeni Model: {new_model_name}")

# --- ÇALIŞTIR ---
# 1. BTC Modelini Kaynak Göster
# 2. ETH Verisini Hedef Göster (6 Aylık veriyi kullanıyoruz)
#timeframes=('5M','15m','1h')
#for i in timeframes:
#    fine_tune_model(f"BTC_60Ay_{i}_MODEL.pth", "ETH", 6, i)

fine_tune_model(f"BTC_60Ay_15m_MODEL.pth", "ETH", 6, '15m')
