import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend for saving without display

from ai_engine import CryptoDataset, CryptoTransformer

# =========================================================
# GÖRSELLEŞTIRME FONKSİYONU
# =========================================================
def plot_training_metrics(train_losses, test_losses, learning_rates, save_path):
    """
    Eğitim metriklerini grafiklere dönüştürüp kaydeder
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)

    # 1. Train vs Test Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train vs Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Test Loss (Yakınlaştırılmış)
    axes[0, 1].plot(epochs, test_losses, 'r-', linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Test Loss (Detailed)')
    axes[0, 1].grid(True, alpha=0.3)
    best_epoch = test_losses.index(min(test_losses)) + 1
    axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch: {best_epoch}')
    axes[0, 1].legend()

    # 3. Learning Rate
    axes[1, 0].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Loss Improvement
    train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    test_improvement = [(test_losses[0] - loss) / test_losses[0] * 100 for loss in test_losses]
    axes[1, 1].plot(epochs, train_improvement, 'b-', label='Train Improvement', linewidth=2)
    axes[1, 1].plot(epochs, test_improvement, 'r-', label='Test Improvement', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('Loss Improvement Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Grafik kaydedildi: {save_path}")

# =========================================================
# EĞİTİM FONKSİYONU (Güncellenmiş)
# =========================================================
def train_specific_model(coin_name, timeframe, month_period):
    # Dosya İsimlerini Dinamik Oluştur
    csv_path = f"{coin_name}_{month_period}Ay_{timeframe}_AI_Ready.csv"
    model_save_name = f"{coin_name}_{month_period}Ay_{timeframe}_MODEL.pth"
    plot_save_name = f"{coin_name}_{month_period}Ay_{timeframe}_METRICS.png"

    print(f"\n{'='*60}")
    print(f"🎯 EĞİTİM HEDEFİ (Sıralı Split): {coin_name} | {timeframe} | {month_period} Aylık Veri")
    print(f"📂 Veri Dosyası: {csv_path}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: '{csv_path}' bulunamadı!")
        return

    # --- AYARLAR ---
    CONFIG = {
        'seq_len': 120,
        'batch_size': 512,
        'd_model': 512,
        'nhead': 16,
        'num_layers': 6,
        'epochs': 100,
        'learning_rate': 0.001
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Donanım: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok'})")

    # Veri Seti
    full_dataset = CryptoDataset(csv_path, seq_len=CONFIG['seq_len'])

    # --- SIRA TABİİ BÖLME ---
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_data = Subset(full_dataset, range(0, train_size))
    test_data = Subset(full_dataset, range(train_size, len(full_dataset)))

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    # Model
    model = CryptoTransformer(
        input_dim=14,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        output_dim=1,
        dropout=0.4
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_test_loss = float('inf')

    # Metrikleri saklamak için listeler
    train_losses = []
    test_losses = []
    learning_rates = []

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)

            scaled_target = by * 100.0
            loss = criterion(output.squeeze(), scaled_target)

            loss.backward()
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
                scaled_target = by * 100.0
                loss = criterion(output.squeeze(), scaled_target)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        scheduler.step(avg_test_loss)

        # Metrikleri kaydet
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        learning_rates.append(current_lr)

        # --- KAYIT ---
        save_msg = ""
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_name)
            save_msg = "💾 REKOR & KAYIT"

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.5f} | Test: {avg_test_loss:.5f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | {save_msg}")

    # --- GRAFİKLERİ OLUŞTUR VE KAYDET ---
    plot_training_metrics(train_losses, test_losses, learning_rates, plot_save_name)

    print(f"\n🏁 Tamamlandı. En iyi Test Loss: {best_test_loss:.5f}")
    print(f"💾 Model: {model_save_name}")
    print(f"📊 Grafik: {plot_save_name}")
    print("-" * 60)

def run_all_trainings():
    coin = "BTC"
    # Veri setlerine göre burayı güncelle
    train_specific_model(coin, "5m", 36)
    train_specific_model(coin, "15m", 36)
    train_specific_model(coin, "1h", 36)

if __name__ == "__main__":
    run_all_trainings()