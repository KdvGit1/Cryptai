import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

# ==========================================
# 1. VERİ OKUYUCU SINIFI (DATASET)
# ==========================================
class CryptoDataset(Dataset):
    def __init__(self, csv_file, seq_len=60):
        """
        Kaydedilen AI_Ready.csv dosyasını okur ve modele hazırlar.
        """
        self.df = pd.read_csv(csv_file, index_col=0)

        # GÜVENLİK KONTROLÜ:
        # Eğer 'Date' yanlışlıkla sütun olarak geldiyse veya 'Unnamed: 0' varsa temizle
        cols_to_drop = [c for c in self.df.columns if 'date' in c.lower() or 'unnamed' in c.lower()]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)

        # Hangi sütunları kullanacağımızı ekrana yazalım (Kontrol amaçlı)
        # Buranın 10 tane sayısal özellik olması lazım.
        print(f"Kullanılan Özellikler ({len(self.df.columns)}): {list(self.df.columns)}")

        # Eğer tarih sütunu index değil de normal sütun olarak geldiyse düşür
        if 'Date' in self.df.columns:
            self.df.drop(columns=['Date'], inplace=True)

        # Veriyi PyTorch Tensor'una çevir (float32 formatında)
        # Tablodaki tüm veriler (Log_Ret, RSI, Vol_Ratio vs.) özellik (feature) olarak alınır.
        self.data_matrix = torch.tensor(self.df.values, dtype=torch.float32)

        self.seq_len = seq_len

    def __len__(self):
        # Elimizdeki toplam pencere sayısı
        return len(self.df) - self.seq_len

    def __getitem__(self, index):
        # GİRİŞ (X): index'ten başla, seq_len kadar git (Örn: 60 mumluk kesit)
        x = self.data_matrix[index : index + self.seq_len]

        # HEDEF (Y): Kesitten hemen sonraki mumun "Log_Ret" değeri
        # Log_Ret bizim dosyamızda 0. sütundaydı (CSV'yi kontrol etmiştik)
        y = self.data_matrix[index + self.seq_len, 0]

        return x, y

# ==========================================
# 2. MODEL MİMARİSİ (TRANSFORMER)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Transformer'a zaman kavramını öğreten modül.
    Bunu eklemezsek model 1. mum ile 60. mum arasındaki sıra farkını bilemez.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Logaritmik ölçekte pozisyon matrisi oluşturma (Standart Formül)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Batch boyutunu ekle (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Giriş verisine pozisyon bilgisini ekle
        x = x + self.pe[:, :x.size(1)]
        return x

class CryptoTransformer(nn.Module):
    def __init__(self, input_dim=14, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        """
        input_dim : CSV'deki sütun sayısı (Bizde 10 adet var)
        d_model   : Modelin içindeki nöron sayısı (Zeka kapasitesi)
        nhead     : Multi-Head Attention kafa sayısı (Farklı desenlere odaklanma)
        num_layers: Transformer katman sayısı (Derinlik)
        """
        super(CryptoTransformer, self).__init__()

        # 1. Giriş Katmanı: 10 özelliği genişletip 128'lik vektöre çevirir
        self.feature_embed = nn.Linear(input_dim, d_model)

        # 2. Pozisyon Kodlaması (Zaman algısı)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder (Asıl Beyin)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. Çıkış Katmanı (Tahmin)
        # Transformer'ın çıkışını tek bir sayıya (Log_Ret) indirger.
        # Negatif sayı üretebilmesi için burada ReLU veya Sigmoid YOK!
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [Batch_Size, Seq_Len(60), Features(10)]

        # Özellikleri genişlet
        x = self.feature_embed(x)

        # Zaman bilgisini ekle
        x = self.pos_encoder(x)

        # Transformer katmanlarından geçir
        x = self.transformer_encoder(x)

        # Sadece SON mumun ürettiği bilgiye bakarak tahmin yap
        # (Seq_Len boyutundaki tüm çıktıyı değil, son zaman adımını alıyoruz)
        last_step_output = x[:, -1, :]

        # Sonuca dönüştür
        output = self.decoder(last_step_output)

        return output

# --- TEST BLOĞU ---
if __name__ == "__main__":
    # Kodun hatasız çalıştığını test etmek için (Eğitim değil, sadece kontrol)

    # 1. Dataset Testi (Dosya adını kendi dosyanla değiştirmen gerekebilir)
    try:
        ds = CryptoDataset('BTC_3Ay_15m_AI_Ready.csv', seq_len=60)
        sample_x, sample_y = ds[0]
        print(f"✅ Veri Okundu. Giriş Şekli: {sample_x.shape} (60 mum, 10 özellik)")
        print(f"✅ Hedef Değer: {sample_y} (Sıradaki mumun değişimi)")
    except Exception as e:
        print(f"⚠️ Dosya bulunamadı veya hata: {e}")

    # 2. Model Testi
    # Rastgele veri ile modelin içinden veri geçirelim
    model = CryptoTransformer(input_dim=14, d_model=64, nhead=4, num_layers=2)

    # Batch Size=32 olan sahte bir veri oluştur
    fake_input = torch.randn(32, 60, 10)
    output = model(fake_input)

    print(f"✅ Model Çıktı Şekli: {output.shape} (32 adet tahmin)")
    print("Mimaride sorun yok! 🚀")