import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from ai_engine import CryptoDataset, CryptoTransformer

# =========================================================
# GÖRSELLEŞTİRME FONKSİYONU
# =========================================================
def plot_training_metrics(train_losses, val_losses, learning_rates, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training & Validation Metrics', fontsize=16, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)

    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss (Weighted)', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'orange', label='Val Loss (Actual)', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, val_losses, 'orange', linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss (Detailed)')
    axes[0, 1].grid(True, alpha=0.3)

    if len(val_losses) > 0:
        best_val_idx = val_losses.index(min(val_losses))
        axes[0, 1].axvline(x=best_val_idx + 1, color='g', linestyle='--', label=f'Best Epoch: {best_val_idx + 1}')
        axes[0, 1].legend()

    axes[1, 0].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    if len(train_losses) > 0:
        train_imp = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
        val_imp = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        axes[1, 1].plot(epochs, train_imp, 'b-', label='Train Imp.', linewidth=2)
        axes[1, 1].plot(epochs, val_imp, 'orange', label='Val Imp.', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Improvement Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Grafik kaydedildi: {save_path}")

# =========================================================
# EĞİTİM FONKSİYONU
# =========================================================
def train_specific_model(coin_name, timeframe, month_period):
    csv_path = f"{coin_name}_{month_period}Ay_{timeframe}_AI_Ready.csv"
    model_save_name = f"{coin_name}_{month_period}Ay_{timeframe}_MODEL.pth"
    plot_save_name = f"{coin_name}_{month_period}Ay_{timeframe}_METRICS.png"

    print(f"\n{'='*60}")
    print(f"🎯 EĞİTİM: {coin_name} | {timeframe} | {month_period} Ay")
    print(f"📂 Veri: {csv_path}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: '{csv_path}' bulunamadı!")
        return

    CONFIG = {
        'seq_len': 120,
        'batch_size': 2048,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'epochs': 100,
        'learning_rate': 0.001
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Donanım: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok'})")

    # 1. Veri Seti Hazırlığı
    full_dataset = CryptoDataset(csv_path, seq_len=CONFIG['seq_len'])
    total_len = len(full_dataset)

    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    train_data = Subset(full_dataset, range(0, train_size))
    val_data = Subset(full_dataset, range(train_size, train_size + val_size))
    test_data = Subset(full_dataset, range(train_size + val_size, total_len))

    # Loaderlar
    # Windows'ta bazen num_workers > 0 sorun yaratabilir. Hata alırsan 0 yap.
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    # 2. Model Kurulumu
    model = CryptoTransformer(
        input_dim=14,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        output_dim=1,
        dropout=0.2
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # --- DÜZELTME: GÜNCEL PYTORCH SÖZDİZİMİ ---
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    train_losses, val_losses, learning_rates = [], [], []

    #early stopping isi
    early_stopping_patience = 15
    epochs_no_improve = 0

    # --- EĞİTİM DÖNGÜSÜ ---
    print("🚀 Eğitim Başlıyor...")
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()

        # --- TRAIN PHASE ---
        model.train()
        total_train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # --- DÜZELTME: GÜNCEL PYTORCH SÖZDİZİMİ ---
            with torch.amp.autocast('cuda'):
                output = model(bx).squeeze()
                scaled_target = by * 100.0

                raw_loss = criterion(output, scaled_target)
                sample_weights = 1.0 + (torch.abs(scaled_target) * 5.0)
                weighted_loss = (raw_loss * sample_weights).mean()

            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += weighted_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
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
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)

        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_name)
            save_msg = "💾 KAYIT (Best Val)"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - start_time
        print(f"Ep {epoch+1:02d} | Time: {epoch_time:.1f}s | Train(W): {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f} | {save_msg}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n overfit ihtimalinden kacinmak icin egitim erken bitirildi.Model {early_stopping_patience} epoch boyunca gelismedi.")
            break

    # --- FİNAL TEST ---
    print(f"\n{'='*60}")
    print("🏁 EĞİTİM BİTTİ. FİNAL TEST ÇALIŞTIRILIYOR...")

    model.load_state_dict(torch.load(model_save_name))
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
    print(f"🧪 Final Test Loss (Görülmemiş Veri): {final_test_loss:.5f}")

    plot_training_metrics(train_losses, val_losses, learning_rates, plot_save_name)
    print("-" * 60)

def run_all_trainings():
    coin = "BTC"
    train_specific_model(coin, "5m", 36)
    train_specific_model(coin, "15m", 36)
    train_specific_model(coin, "1h", 36)

if __name__ == "__main__":
    run_all_trainings()