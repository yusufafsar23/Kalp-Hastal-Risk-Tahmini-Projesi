"""
Kalp Hastalığı Veri Seti - Veri Ön İşleme
Bu script, kalp hastalığı veri seti üzerinde veri ön işleme adımlarını uygular.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

# Grafik ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("Kalp Hastalığı Veri Seti - Veri Ön İşleme")
    print("="*50)
    
    # Klasörleri oluştur
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Veri Setinin Yüklenmesi
    print("\n1. Veri Setinin Yüklenmesi")
    print("-"*30)
    df = pd.read_csv('data/kalp_hastaligi/heart.csv')
    print("İlk 5 satır:")
    print(df.head())
    
    # 2. Veri Seti Bilgileri
    print("\n2. Veri Seti Bilgileri")
    print("-"*30)
    print(f"Veri seti boyutu: {df.shape}")
    print("\nVeri tipleri:")
    print(df.dtypes)
    
    # 3. Eksik Değer Kontrolü
    print("\n3. Eksik Değer Kontrolü")
    print("-"*30)
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzdesi = (df.isnull().sum() / len(df)) * 100
    
    eksik_veriler = pd.DataFrame({
        'Eksik Değer Sayısı': eksik_deger,
        'Eksik Değer Yüzdesi': eksik_deger_yuzdesi.round(2)
    })
    print(eksik_veriler)
    
    # 4. Kategorik ve Sayısal Değişkenlerin Belirlenmesi
    print("\n4. Kategorik ve Sayısal Değişkenlerin Belirlenmesi")
    print("-"*30)
    
    # Kategorik ve sayısal değişkenleri belirle
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Hedef değişken
    target = 'target'
    
    print("Kategorik değişkenler:", categorical_cols)
    print("Sayısal değişkenler:", numerical_cols)
    print("Hedef değişken:", target)
    
    # 5. Veri Ön İşleme
    print("\n5. Veri Ön İşleme")
    print("-"*30)
    
    # Özellikler (X) ve hedef değişkeni (y) ayır
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Veri ön işleme adımları için bir pipeline oluştur
    # Sayısal değişkenler için StandardScaler uygula (z-score normalizasyonu)
    # Kategorik değişkenler için OneHotEncoder uygula
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Preprocessor'u uygula
    X_processed = preprocessor.fit_transform(X)
    
    # One-Hot Encoding sonucu oluşan yeni özellik isimlerini belirle
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    # Tüm özellik isimlerini birleştir
    feature_names = list(numerical_cols) + list(cat_feature_names)
    
    # İşlenmiş verileri DataFrame'e dönüştür (sparse matrisi yoğun matrise dönüştür)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    print("İşlenmiş verilerden ilk 5 satır:")
    print(X_processed_df.head())
    
    # 6. Veriyi Eğitim ve Test Setlerine Bölme
    print("\n6. Veriyi Eğitim ve Test Setlerine Bölme")
    print("-"*30)
    
    # Veriyi eğitim ve test setlerine böl (%70 eğitim, %30 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Veri setlerinin boyutlarını göster
    print(f"X_train boyutu: {X_train.shape}")
    print(f"X_test boyutu: {X_test.shape}")
    print(f"y_train boyutu: {y_train.shape}")
    print(f"y_test boyutu: {y_test.shape}")
    
    # 7. İşlenmiş Verileri Kaydet
    print("\n7. İşlenmiş Verileri Kaydetme")
    print("-"*30)
    
    # Veri ön işleme bilgilerini bir sözlükte sakla
    preprocessing_info = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }
    
    # Eğitim ve test verilerini sözlükte sakla
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessing_info': preprocessing_info
    }
    
    # İşlenmiş verileri pickle dosyası olarak kaydet
    with open('data/processed/processed_heart_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Özellik isimlerini CSV dosyası olarak kaydet
    pd.DataFrame(feature_names, columns=['feature_name']).to_csv('data/processed/feature_names.csv', index=False)
    
    # Örnek işlenmiş verileri CSV olarak kaydet
    X_processed_df.head(20).to_csv('data/processed/processed_heart_data_sample.csv', index=False)
    
    print("Veri ön işleme tamamlandı. İşlenmiş veriler 'data/processed/' klasörüne kaydedildi.")
    print("\nVeri Ön İşleme Özeti:")
    print("1. Veri seti yüklendi ve incelendi")
    print("2. Eksik değerler kontrol edildi (eksik değer bulunmadı)")
    print("3. Kategorik ve sayısal değişkenler belirlendi")
    print("4. Sayısal değişkenlere StandardScaler (z-score normalizasyonu) uygulandı")
    print("5. Kategorik değişkenlere OneHotEncoder uygulandı")
    print("6. Veri seti eğitim (%70) ve test (%30) kümelerine ayrıldı, stratifiye örnekleme yapıldı")
    print("7. İşlenmiş veriler kaydedildi")
    print("\nBir sonraki adımda bu veriler kullanılarak makine öğrenmesi modelleri oluşturulacak ve eğitilecektir.")

if __name__ == "__main__":
    main() 