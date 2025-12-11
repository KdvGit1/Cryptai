import pandas as pd
import numpy as np

# En son oluşan AI dosyasını oku (Örnek: 15 dakikalık veriyi kontrol edelim)
# Dosya ismini senin oluşturduğunla aynı yaptığına emin ol
df_check = pd.read_csv('BTC_3Ay_1h_AI_Ready.csv', index_col=0)

print("--- GENEL BİLGİ (INFO) ---")
print(df_check.info())

print("\n--- İSTATİSTİKSEL ÖZET (DESCRIBE) ---")
# Bilimsel gösterimi (1.2e-05) kapatıp normal sayı görmek için format ayarı
pd.set_option('display.float_format', lambda x: '%.5f' % x)
print(df_check.describe())

print("\n--- SONSUZ DEĞER KONTROLÜ ---")
# Veri setinde inf (sonsuz) veya -inf var mı?
is_infinite = np.isinf(df_check).sum().sum()
print(f"Toplam Sonsuz Değer Sayısı: {is_infinite}")
if is_infinite > 0:
    print("⚠️ DİKKAT: Veride sonsuz sayılar var, temizlenmeli!")

print("\n--- İLK 5 SATIR ---")
print(df_check.head())