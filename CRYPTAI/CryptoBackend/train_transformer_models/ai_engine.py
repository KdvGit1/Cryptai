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

        # Sadece sayısal verileri al (Garanti olsun)
        self.df = self.df.select_dtypes(include=[np.number])

        # Hangi sütunları kullanacağımızı ekrana yazalım (Kontrol amaçlı)
        # Buranın 14 tane sayısal özellik olması lazım.
        # print(f"Kullanılan Özellikler ({len(self.df.columns)}): {list(self.df.columns)}")

        # Veriyi PyTorch Tensor'una çevir (float32 formatında)
        self.data_matrix = torch.tensor(self.df.values, dtype=torch.float32)

        self.seq_len = seq_len

    def __len__(self):
        # Elimizdeki toplam pencere sayısı
        return len(self.df) - self.seq_len

    def __getitem__(self, index):
        # GİRİŞ (X): index'ten başla, seq_len kadar git
        x = self.data_matrix[index : index + self.seq_len]

        # HEDEF (Y): Kesitten hemen sonraki mumun "Log_Ret" değeri
        # Log_Ret bizim dosyamızda 0. sütundaydı
        # DİKKAT: Burada 100 ile çarpma yapmıyoruz. Onu train.py içinde GPU'da yapacağız.
        y = self.data_matrix[index + self.seq_len, 0]

        return x, y

# ==========================================
# 2. MODEL MİMARİSİ (TRANSFORMER)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Transformer'a zaman kavramını öğreten modül.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Logaritmik ölçekte pozisyon matrisi oluşturma
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
    # --- GÜNCELLEME: output_dim parametresi eklendi (Varsayılan: 1) ---
    def __init__(self, input_dim=14, d_model=128, nhead=4, num_layers=2, output_dim=1, dropout=0.1):
        """
        input_dim : CSV'deki sütun sayısı (Bizde 14 adet var)
        d_model   : Modelin içindeki nöron sayısı (Zeka kapasitesi)
        nhead     : Multi-Head Attention kafa sayısı
        num_layers: Transformer katman sayısı
        output_dim: Çıktı sayısı (1 mum tahmini için 1, 5 mum için 5)
        """
        super(CryptoTransformer, self).__init__()

        # 1. Giriş Katmanı
        self.feature_embed = nn.Linear(input_dim, d_model)

        # 2. Pozisyon Kodlaması
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder (Asıl Beyin)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. Çıkış Katmanı (Tahmin)
        # Dinamik çıktı boyutu (output_dim)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: [Batch_Size, Seq_Len, Features]

        # Özellikleri genişlet
        x = self.feature_embed(x)

        # Zaman bilgisini ekle
        x = self.pos_encoder(x)

        # Transformer katmanlarından geçir
        x = self.transformer_encoder(x)

        # Sadece SON mumun ürettiği bilgiye bakarak tahmin yap
        last_step_output = x[:, -1, :]

        # Sonuca dönüştür
        output = self.decoder(last_step_output)

        return output
