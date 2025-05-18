import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Veri setini oku
    veri_yolu = "data/kalp_hastaligi/heart.csv"
    df = pd.read_csv(veri_yolu)
    
    # Veri setinin ilk 5 satırını görüntüle
    print("İlk 5 satır:")
    print(df.head())
    
    # Veri seti hakkında genel bilgiler
    print("\nVeri seti bilgileri:")
    print(f"Satır sayısı: {df.shape[0]}")
    print(f"Sütun sayısı: {df.shape[1]}")
    print("\nSütun açıklamaları:")
    print("- age: Yaş")
    print("- sex: Cinsiyet (1 = erkek, 0 = kadın)")
    print("- cp: Göğüs ağrısı tipi (0-3)")
    print("- trestbps: Dinlenme kan basıncı (mm Hg)")
    print("- chol: Serum kolesterol (mg/dl)")
    print("- fbs: Açlık kan şekeri > 120 mg/dl (1 = evet, 0 = hayır)")
    print("- restecg: Dinlenme elektrokardiyografik sonuçları (0-2)")
    print("- thalach: Maksimum kalp hızı")
    print("- exang: Egzersizle indüklenen anjina (1 = evet, 0 = hayır)")
    print("- oldpeak: ST depresyonu")
    print("- slope: ST segmentinin eğimi (0-2)")
    print("- ca: Floroskopi ile renklendirilmiş ana damar sayısı (0-4)")
    print("- thal: Talyum stres testi (0-3)")
    print("- target: Kalp hastalığı (1 = var, 0 = yok)")
    
    # Veri türlerini göster
    print("\nVeri türleri:")
    print(df.dtypes)
    
    # Temel istatistikleri göster
    print("\nTemel istatistikler:")
    print(df.describe())
    
    # Hedef değişkenin dağılımını görselleştir
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Kalp Hastalığı Dağılımı')
    plt.xlabel('Kalp Hastalığı (1 = var, 0 = yok)')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('images/kalp_hastaligi_dagilimi.png')
    
    # Yaş ve cinsiyet dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='sex', multiple='stack', bins=20)
    plt.title('Yaş ve Cinsiyet Dağılımı')
    plt.xlabel('Yaş')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('images/yas_cinsiyet_dagilimi.png')
    
    print("\nGörselleştirmeler 'images' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 