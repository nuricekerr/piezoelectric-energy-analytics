import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Gerçekçi Simülasyon Verisi Üretimi (PVDF ve PZT-5H Rezonans Modellemesi)
np.random.seed(42)
n_samples = 500

# Girdiler: Titreşim Frekansı (Hz) ve İvme Genliği (g)
frequency = np.random.uniform(20, 120, n_samples)  # 20Hz - 120Hz arası mekanik titreşimler
acceleration = np.random.uniform(0.5, 4.5, n_samples)  # 0.5g - 4.5g arası ivmelenme

# Çıktı: Üretilen Gerilim (Volt) 
# Doğrusal olmayan rezonans ve akıllı malzeme davranışını simüle eden fizik tabanlı formül
voltage = (0.08 * frequency) + (1.5 * acceleration) + np.random.normal(0, 0.3, n_samples)

# Veriyi DataFrame yapısına dökme
data = pd.DataFrame({
    'Frequency_Hz': frequency,
    'Acceleration_g': acceleration,
    'Output_Voltage_V': voltage
})

print("=== Mühendislik Veri Seti Analizi (İlk 5 Satır) ===")
print(data.head())

# 2. Makine Öğrenmesi - Voltaj Tahmin Modeli (Regresyon)
X = data[['Frequency_Hz', 'Acceleration_g']]
y = data['Output_Voltage_V']

# Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Yapay Zeka Modelini Eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Model Performans Değerlendirmesi
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n=== Yapay Zeka Model Sonuçları ===")
print(f"Model Başarı Oranı (R2 Score): {r2:.4f} (%{r2*100:.2f} Doğruluk)")
print(f"Ortalama Kare Hata (MSE): {mse:.4f}")

# 3. Grafiksel Görselleştirme (Analiz Çıktısı)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Frequency_Hz'], y=data['Output_Voltage_V'], hue=data['Acceleration_g'], palette='coolwarm')
plt.title('Piezoelectric Energy Harvesting: Voltage vs Frequency & Acceleration')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Generated Voltage (V)')
plt.grid(True)

# Grafiği bilgisayara kaydetme
plt.savefig('piezo_analysis_chart.png')
print("\n[BAŞARILI] 'piezo_analysis_chart.png' grafiği başarıyla oluşturuldu ve kaydedildi.")