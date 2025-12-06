# Cryptai: Transformer Tabanlı Transfer Learning Kripto Tahmin Sistemi

Bu proje, **Transformer** mimarisini kullanarak kripto para fiyat hareketlerini tahmin eden, **FastAPI** tabanlı otonom bir analiz sistemidir. Sistem, Bitcoin (BTC) üzerinde eğitilen bir "Base Model"i, **Transfer Learning** yöntemiyle diğer altcoin'lere (ETH, SOL vb.) uyarlayarak yüksek doğruluklu tahminler üretir.

## 📂 Proje Mimarisi

* **`ai_engine.py`**: PyTorch tabanlı `CryptoTransformer` model mimarisi ve `CryptoDataset` sınıfı.
* **`train.py`**: Modelin eğitimi, **Mixed Precision (AMP)** kullanımı ve **Huber Loss** ile optimize edilmiş eğitim döngüsü.
* **`fine_tune_model.py`**: Eğitilmiş ana modelin ağırlıklarını alıp, düşük `learning_rate` ile başka coinlere uyarlayan (Transfer Learning) modül.
* **`data_fetcher.py`**: `CCXT` ile veri çekme ve `TA-Lib` kullanarak 14 farklı teknik indikatörün (Feature Engineering) hesaplandığı katman.
* **`exchange_scrapper.py`**: Canlı piyasayı tarayan, modeli çalıştıran ve sonuçları JSON olarak dışarı aktaran inferans motoru.
* **`api_services.py`**: Analiz sonuçlarını ve tarama tetikleyicilerini dış dünyaya açan **FastAPI** servisi.

## 🚀 Özellikler

- **Transformer Encoder Mimarisi:** Zaman serisi verilerindeki uzun vadeli bağımlılıkları yakalar.
- **Transfer Learning & Fine-Tuning:** Her coin için sıfırdan model eğitmek yerine, BTC modelinin "piyasa bilgisini" diğer coinlere aktarır. ETH ile fine tuning yapilmis model ureterek daha oynak piyasalara hazir modeller de egitildi.
- **Gelişmiş Veri İşleme:** RSI, Bollinger Bantları, MACD ve Hacim osilatörleri dahil 14 özellikli girdi matrisi.
- **Otomatik Ölçeklendirme:** Veriler model için normalize edilirken, insanlar için ham değerler saklanır.
- **API Desteği:** `scan_market`, `get_coin_data` gibi endpointler ile frontend entegrasyonuna hazır.

## 🛠️ Kurulum

1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/KdvGit1/Cryptai.git](https://github.com/KdvGit1/Cryptai.git)
