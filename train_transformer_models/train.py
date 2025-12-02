import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

# Motor dosyamızdan gerekli parçaları alıyoruz
from ai_engine import CryptoDataset, CryptoTransformer

# =========================================================
# EĞİTİM FONKSİYONU
# =========================================================
def train_specific_model(coin_name, timeframe, month_period):
    """
    Target Scaling ve Huber Loss ile güçlendirilmiş eğitim fonksiyonu.
    """

    # Dosya İsimleri
    csv_path = f"{coin_name}_{month_period}Ay_{timeframe}_AI_Ready.csv"
    model_save_name = f"{coin_name}_{month_period}Ay_{timeframe}_MODEL.pth"

    print(f"\n{'='*60}")
    print(f"🎯 HEDEF (SCALED): {coin_name} | {timeframe} | {month_period} Aylık Veri")
    print(f"📂 Okunacak: {csv_path}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: {csv_path} bulunamadı! Önce veri çekmelisin.")
        return

    # --- AYARLAR ---
    CONFIG = {
        'seq_len': 120,
        'batch_size': 256,      # Büyük batch (RTX 3060 için uygun)
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'epochs': 50,
        'learning_rate': 0.0005 # Huber Loss ile biraz daha düşük LR iyidir
    }

    # Cihaz Seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Donanım: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok'})")

    # Veri Setini Yükle
    full_dataset = CryptoDataset(csv_path, seq_len=CONFIG['seq_len'])

    # %80 Eğitim, %20 Test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, test_size])

    # DataLoader (CPU worker sayısını artırarak veri akışını hızlandırıyoruz)
    # Windows'ta bazen num_workers hata verebilir, verirse 0 yapın.
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)

    # Modeli İnşa Et
    model = CryptoTransformer(
        input_dim=14, # 14 özellik
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=0.2
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # --- 1. DEĞİŞİKLİK: HUBER LOSS ---
    # Delta=1.0: Hata 1.0'dan küçükse karesini al (Hassas), büyükse düz al (Spike koruması)
    criterion = nn.HuberLoss(delta=1.0)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # --- YENİ EKLENTİ: SCHEDULER ---
    # Eğer "Test Loss" 5 epoch boyunca düşmezse, öğrenme hızını (LR) yarıya indir.
    # Bu, tıkanan modelin kilidini açar.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # EĞİTİM DÖNGÜSÜ
    best_test_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        start_time = time.time()

        # --- TRAIN ---
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            output = model(bx)

            # --- 2. DEĞİŞİKLİK: TARGET SCALING ---
            # Gerçek değeri 100 ile çarpıyoruz.
            # 0.001 -> 0.1 olur. Model bunu daha rahat öğrenir.
            scaled_target = by * 100.0

            # Model çıktısını [Batch, 1] formatından [Batch] formatına getir (squeeze)
            loss = criterion(output.squeeze(), scaled_target)

            loss.backward()

            # Gradyan patlamasını önle (Güvenlik sigortası)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- TEST ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)

                output = model(bx)

                # Test ederken de aynı ölçeği kullanmalıyız ki adil olsun
                scaled_target = by * 100.0

                loss = criterion(output.squeeze(), scaled_target)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        # Scheduler'a rapor ver: "Durum bu, gerekirse hızı düşür"
        scheduler.step(avg_test_loss)

        # --- KAYIT ---
        elapsed = time.time() - start_time
        save_msg = ""

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_name)
            save_msg = "✅ REKOR & KAYIT"

        # Mevcut öğrenme hızını al
        current_lr = optimizer.param_groups[0]['lr']
        # Loss değerlerini terminalde 5 haneli gösterelim
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.5f} | Test: {avg_test_loss:.5f} | LR: {current_lr:.6f} | {save_msg}")

    print(f"🏁 {coin_name} - {timeframe} tamamlandı. En iyi Scaled Loss: {best_test_loss:.5f}")
    print("-" * 60)


# =========================================================
# ANA ÇALIŞTIRICI
# =========================================================
def run_all_trainings():
    coin = "BTC"

    # Eğer 6 aylık veri çektiysen:
    train_specific_model(coin, "5m", 60)
    train_specific_model(coin, "15m", 60)
    train_specific_model(coin, "1h", 60)

if __name__ == "__main__":
    run_all_trainings()