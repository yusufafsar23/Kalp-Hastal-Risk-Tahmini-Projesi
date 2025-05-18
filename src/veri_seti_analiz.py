import pandas as pd

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
    
    # Sütunların açıklamaları
    sutun_aciklamalari = {
        "age": "Yaş",
        "sex": "Cinsiyet (1 = erkek, 0 = kadın)",
        "cp": "Göğüs ağrısı tipi (0-3)",
        "trestbps": "Dinlenme kan basıncı (mm Hg)",
        "chol": "Serum kolesterol (mg/dl)",
        "fbs": "Açlık kan şekeri > 120 mg/dl (1 = evet, 0 = hayır)",
        "restecg": "Dinlenme elektrokardiyografik sonuçları (0-2)",
        "thalach": "Maksimum kalp hızı",
        "exang": "Egzersizle indüklenen anjina (1 = evet, 0 = hayır)",
        "oldpeak": "ST depresyonu",
        "slope": "ST segmentinin eğimi (0-2)",
        "ca": "Floroskopi ile renklendirilmiş ana damar sayısı (0-4)",
        "thal": "Talyum stres testi (0-3)",
        "target": "Kalp hastalığı (1 = var, 0 = yok)"
    }
    
    print("\nSütun açıklamaları:")
    for sutun, aciklama in sutun_aciklamalari.items():
        print(f"- {sutun}: {aciklama}")
    
    # Veri türlerini göster
    print("\nVeri türleri:")
    print(df.dtypes)
    
    # Temel istatistikleri göster
    print("\nTemel istatistikler:")
    print(df.describe())
    
    # Değer sayıları
    print("\nHedef değişken dağılımı (Kalp hastalığı var/yok):")
    print(df['target'].value_counts())
    
    # Cinsiyet dağılımı
    print("\nCinsiyet dağılımı:")
    print(df['sex'].value_counts())
    
    # Yaş gruplarına göre analiz
    print("\nYaş gruplarına göre kalp hastalığı dağılımı:")
    df['yas_grubu'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '>60'])
    print(df.groupby(['yas_grubu', 'target']).size().unstack())
    
    print("\nAnaliz tamamlandı.")

if __name__ == "__main__":
    main() 