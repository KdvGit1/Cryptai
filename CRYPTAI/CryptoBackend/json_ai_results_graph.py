import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob

def plot_readable_distributions():
    files = glob.glob("live_market_data*.json")

    if not files:
        print("❌ Dosya bulunamadı.")
        return

    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 6))
    if len(files) == 1: axes = [axes]

    for i, filename in enumerate(files):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            predictions = []
            for coin, details in data.items():
                if isinstance(details, dict) and 'ai_prediction' in details:
                    # DEĞİŞİKLİK: Değeri 100 ile çarpıp % formatına getiriyoruz
                    predictions.append(details['ai_prediction'] * 100)
                elif '1h' in details:
                    for tf, info in details.items():
                        if 'ai_prediction' in info:
                            predictions.append(info['ai_prediction'] * 100)

            ax = axes[i]
            if predictions:
                ax.hist(predictions, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)

                # DEĞİŞİKLİK: X eksenini % formatında ve büyük fontla yaz
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f%%'))
                ax.tick_params(axis='x', rotation=45, labelsize=10)

                avg_val = sum(predictions) / len(predictions)
                ax.axvline(avg_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Ort: {avg_val:.3f}%')

                ax.set_title(f"{filename}\n(Ölçek: Yüzde %)", fontsize=12, fontweight='bold')
                ax.set_xlabel('Tahmin Edilen Değişim (%)', fontsize=11)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Veri Yok", ha='center')

        except Exception as e:
            print(f"Hata: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_readable_distributions()