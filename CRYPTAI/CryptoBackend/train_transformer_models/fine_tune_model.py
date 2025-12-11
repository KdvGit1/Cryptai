import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ai_engine dosyasından gerekli sınıfları çekiyoruz
from ai_engine import CryptoDataset, CryptoTransformer

# =========================================================
# GÖRSELLEŞTİRME (Eğitim Sürecini Kaydeder)
# =========================================================
def plot_fine_tune_metrics(train_losses, val_losses, coin_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (Fine-Tune)', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title(f'{coin_name} Transfer Learning (Fine-Tune) Süreci')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Grafik kaydedildi: {save_path}")

# =========================================================
# FINE TUNE FONKSİYONU
# =========================================================
def fine_tune_coin(target_coin, base_model_path, timeframe, month_period):
    """
    target_coin: Eğitmek istediğin yeni coin (Örn: 'ETH')
    base_model_path: BTC modelinin dosya yolu (Örn: 'BTC_36Ay_1h_MODEL.pth')
    """

    # Dosya İsimleri
    csv_path = f"{target_coin}_{month_period}Ay_{timeframe}_AI_Ready.csv"
    new_model_save_name = f"{target_coin}_{month_period}Ay_{timeframe}_FINE_TUNED_MODEL_BASED_{base_model_path.split('_')[0]}.pth" # Yeni isimle kaydediyoruz
    plot_save_name = f"{target_coin}_{month_period}Ay_{timeframe}_FINETUNE.png"

    print(f"\n{'='*60}")
    print(f"🧠 TRANSFER LEARNING: {target_coin} | Baz Model: {base_model_path}")
    print(f"📂 Hedef Veri: {csv_path}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: '{csv_path}' bulunamadı! Önce veriyi indirip hazırlamalısın.")
        return

    if not os.path.exists(base_model_path):
        print(f"❌ HATA: Baz model '{base_model_path}' bulunamadı!")
        return

    # --- AYARLAR (Train.py ile AYNI OLMALI - Mimari Uyuşmazlığı Olmasın) ---
    CONFIG = {
        'seq_len': 120,
        'batch_size': 2048,
        'd_model': 128,       # Train.py'deki ile AYNI olmalı
        'nhead': 4,           # Train.py'deki ile AYNI olmalı
        'num_layers': 2,      # Train.py'deki ile AYNI olmalı

        # --- FINE TUNE İÇİN ÖZEL AYARLAR ---
        'epochs': 50,           # Fine-tune için daha az epoch yeterli (Zaten bilgili)
        'learning_rate': 0.0001, # ÇOK ÖNEMLİ: Normalden 10 kat daha düşük LR kullanıyoruz!
        # Amacımız bilgiyi silmek değil, hafifçe yeni coine uyarlamak.
        'patience': 10           # Daha sabırsız early stopping
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Donanım: {device}")

    # 1. Veri Seti (Yeni Coin)
    full_dataset = CryptoDataset(csv_path, seq_len=CONFIG['seq_len'])
    total_len = len(full_dataset)

    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    train_data = Subset(full_dataset, range(0, train_size))
    val_data = Subset(full_dataset, range(train_size, train_size + val_size))
    test_data = Subset(full_dataset, range(train_size + val_size, total_len))

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    # 2. Modeli Başlat
    model = CryptoTransformer(
        input_dim=14,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        output_dim=1,
        dropout=0.2
    ).to(device)

    # 3. BAZ MODELİN AĞIRLIKLARINI YÜKLE (TRANSFER LEARNING)
    print("📥 BTC Bilgisi Yükleniyor...")
    try:
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print("✅ Başarılı: Model BTC tecrübesiyle başlıyor.")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        print("Mimari parametreleri (d_model, nhead vb.) train.py ile aynı mı?")
        return

    # Optimizer (Düşük Learning Rate ile)
    criterion = nn.HuberLoss(delta=1.0, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    # --- EĞİTİM DÖNGÜSÜ ---
    print("🚀 Fine-Tuning Başlıyor...")
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()

        # TRAIN
        model.train()
        total_train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                output = model(bx).squeeze()
                scaled_target = by * 100.0
                raw_loss = criterion(output, scaled_target)
                # Sample Weighting (Aynı mantık korunuyor)
                sample_weights = 1.0 + (torch.abs(scaled_target) * 5.0)
                weighted_loss = (raw_loss * sample_weights).mean()

            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += weighted_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    output = model(bx).squeeze()
                    scaled_target = by * 100.0
                    loss = criterion(output, scaled_target).mean()
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # KAYIT ve EARLY STOPPING
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), new_model_save_name)
            save_msg = "💾 YENİ COIN MODELİ KAYDEDİLDİ"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:02d} | Time: {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f} | {save_msg}")

        if epochs_no_improve >= CONFIG['patience']:
            print(f"🛑 Erken Durdurma: {target_coin} için model optimize oldu.")
            break

    # FINAL TEST
    print(f"\n{'='*60}")
    print(f"🧪 {target_coin} FİNAL TESTİ (Görülmemiş Veri)")
    model.load_state_dict(torch.load(new_model_save_name))
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            output = model(bx).squeeze()
            scaled_target = by * 100.0
            loss = criterion(output, scaled_target).mean()
            total_test_loss += loss.item()

    final_test_loss = total_test_loss / len(test_loader)
    print(f"🎯 Test Loss: {final_test_loss:.5f}")
    plot_fine_tune_metrics(train_losses, val_losses, target_coin, plot_save_name)

# =========================================================
# ÇALIŞTIRMA BÖLÜMÜ
# =========================================================
if __name__ == "__main__":

    # 1. Hangi BTC modeli temel alınacak? (Hali hazırda eğittiğin model)
    BASE_MODEL = "BTC_36Ay_5m_MODEL.pth"

    # 2. Bu model hangi coinlere uyarlanacak? (Veri setleri hazır olmalı!)
    # Örnek: ETH_36Ay_1h_AI_Ready.csv dosyasının var olduğunu varsayıyoruz.
    target_coins = ["ETH"]

    for coin in target_coins:
        fine_tune_coin(
            target_coin=coin,
            base_model_path=BASE_MODEL,
            timeframe="5m",
            month_period=36
        )