"""
Kalp Hastalığı Veri Seti - Keşifsel Veri Analizi (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Grafikler için ayarlar
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("Kalp Hastalığı Veri Seti - Keşifsel Veri Analizi (EDA)")
    print("="*70)
    
    # Veri setini yükle
    print("\n1. Veri Setinin Yüklenmesi")
    print("-"*30)
    df = pd.read_csv('data/kalp_hastaligi/heart.csv')
    print("İlk 5 satır:")
    print(df.head())
    
    # Veri seti bilgileri
    print("\n2. Veri Seti Bilgileri")
    print("-"*30)
    print(f"Veri seti boyutu: {df.shape}")
    print("\nSütun isimleri:")
    print(df.columns.tolist())
    
    # Veri tipleri
    print("\nVeri tipleri:")
    print(df.dtypes)
    
    # Temel istatistikler
    print("\n3. Temel İstatistikler")
    print("-"*30)
    print(df.describe().round(2))
    
    # Eksik değerler
    print("\n4. Eksik Değer Analizi")
    print("-"*30)
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzdesi = (df.isnull().sum() / len(df)) * 100
    
    eksik_veriler = pd.DataFrame({
        'Eksik Değer Sayısı': eksik_deger,
        'Eksik Değer Yüzdesi': eksik_deger_yuzdesi.round(2)
    })
    print(eksik_veriler)
    
    # Kategorik değişken analizleri
    print("\n5. Kategorik Değişken Analizleri")
    print("-"*30)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    
    for col in categorical_cols:
        print(f"\n{col} için değer dağılımı:")
        print(df[col].value_counts())
    
    # Hedef değişken analizi
    print("\n6. Hedef Değişken Analizi")
    print("-"*30)
    target_yuzde = df['target'].value_counts(normalize=True) * 100
    print(f"Kalp hastalığı olmayan yüzdesi: %{target_yuzde[0]:.2f}")
    print(f"Kalp hastalığı olan yüzdesi: %{target_yuzde[1]:.2f}")
    
    # Görselleştirmeler oluştur ve kaydet
    print("\n7. Görselleştirmeler Oluşturuluyor")
    print("-"*30)
    gorsellestime_olustur(df)
    print("Tüm görselleştirmeler 'images' klasörüne kaydedildi.")
    
    # Yaş gruplarına göre analiz
    print("\n8. Yaş Gruplarına Göre Kalp Hastalığı Analizi")
    print("-"*30)
    df['yas_grubu'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '>60'])
    yas_hastalık = pd.crosstab(df['yas_grubu'], df['target'])
    yas_hastalık.columns = ['Hastalık Yok', 'Hastalık Var']
    print(yas_hastalık)
    
    # Bulgular ve sonuçlar
    print("\n9. Bulgular ve Sonuçlar")
    print("-"*30)
    print("""
    Veri seti analizi sonucunda elde ettiğimiz temel bulgular:

    1. Veri setimizde yaklaşık 1025 hasta bulunmaktadır.
    2. Hedef değişken dağılımı oldukça dengelidir (yaklaşık %49 hastalık yok, %51 hastalık var).
    3. Erkek hastalar kadın hastalara göre daha fazladır.
    4. Göğüs ağrısı tipi ile kalp hastalığı arasında güçlü bir ilişki vardır.
    5. Yaş arttıkça maksimum kalp hızı (thalach) azalmaktadır.
    6. Kalp hastalığı olan hastalarda maksimum kalp hızı genellikle daha düşüktür.
    7. Yaş gruplarına göre incelendiğinde, 40-60 yaş aralığında kalp hastalığı daha sık görülmektedir.
    8. ST depresyonu (oldpeak) değeri kalp hastalığı ile pozitif korelasyona sahiptir.
    """)


def gorsellestime_olustur(df):
    """Tüm görselleştirmeleri oluşturur ve kaydeder"""
    
    # Klasörü oluştur
    import os
    os.makedirs('images', exist_ok=True)
    
    # Korelasyon hesaplaması için verileri kopyala
    df_corr = df.copy()
    
    # 1. Hedef değişken dağılımı
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='target', data=df, hue='target', palette='Blues', legend=False)
    plt.title('Kalp Hastalığı Dağılımı', fontsize=16)
    plt.xlabel('Kalp Hastalığı (1 = Var, 0 = Yok)', fontsize=12)
    plt.ylabel('Hasta Sayısı', fontsize=12)
    
    # Çubukların üzerine değer ekle
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5, f'{height}', ha="center")
    
    plt.savefig('images/kalp_hastaligi_dagilimi.png')
    plt.close()
    
    # 2. Cinsiyet dağılımı
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x='sex', data=df, hue='sex', palette='viridis', legend=False)
    plt.title('Cinsiyete Göre Dağılım', fontsize=14)
    plt.xlabel('Cinsiyet (1 = Erkek, 0 = Kadın)', fontsize=12)
    plt.ylabel('Kişi Sayısı', fontsize=12)
    
    # Çubukların üzerine değer ekle
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height + 5, f'{height}', ha="center")
    
    # Cinsiyete göre hastalık dağılımı
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x='sex', hue='target', data=df, palette='viridis')
    plt.title('Cinsiyete Göre Kalp Hastalığı Dağılımı', fontsize=14)
    plt.xlabel('Cinsiyet (1 = Erkek, 0 = Kadın)', fontsize=12)
    plt.ylabel('Kişi Sayısı', fontsize=12)
    plt.legend(['Hastalık Yok', 'Hastalık Var'])
    
    plt.tight_layout()
    plt.savefig('images/cinsiyet_dagilimi.png')
    plt.close()
    
    # 3. Yaş ve hastalık ilişkisi
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20)
    plt.title('Yaşa Göre Kalp Hastalığı Dağılımı', fontsize=16)
    plt.xlabel('Yaş', fontsize=12)
    plt.ylabel('Kişi Sayısı', fontsize=12)
    plt.legend(['Hastalık Yok', 'Hastalık Var'])
    plt.savefig('images/yas_hastalık_dagilimi.png')
    plt.close()
    
    # 4. Yaş gruplarına göre analiz
    df_temp = df.copy()
    df_temp['yas_grubu'] = pd.cut(df_temp['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '>60'])
    yas_hastalık = pd.crosstab(df_temp['yas_grubu'], df_temp['target'])
    yas_hastalık.columns = ['Hastalık Yok', 'Hastalık Var']
    
    plt.figure(figsize=(10, 6))
    yas_hastalık.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Yaş Gruplarına Göre Kalp Hastalığı Dağılımı', fontsize=16)
    plt.xlabel('Yaş Grubu', fontsize=12)
    plt.ylabel('Kişi Sayısı', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('images/yas_grubu_hastalık.png')
    plt.close()
    
    # 5. Korelasyon matrisi
    correlation_matrix = df_corr.corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Korelasyon Matrisi', fontsize=18)
    plt.savefig('images/korelasyon_matrisi.png')
    plt.close()
    
    # 6. Hedef değişken ile korelasyon
    hedef_korelasyon = correlation_matrix['target'].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    hedef_korelasyon.drop('target').plot(kind='bar', colormap='viridis')
    plt.title('Değişkenlerin Kalp Hastalığı ile Korelasyonu', fontsize=16)
    plt.xlabel('Değişkenler', fontsize=12)
    plt.ylabel('Korelasyon Katsayısı', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('images/hedef_korelasyon.png')
    plt.close()
    
    # 7. Yaş ve Maksimum Kalp Hızı İlişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='viridis', s=80, alpha=0.7)
    plt.title('Yaş ve Maksimum Kalp Hızı İlişkisi', fontsize=16)
    plt.xlabel('Yaş', fontsize=12)
    plt.ylabel('Maksimum Kalp Hızı', fontsize=12)
    plt.legend(['Hastalık Yok', 'Hastalık Var'])
    plt.grid(alpha=0.3)
    plt.savefig('images/yas_kalp_hizi.png')
    plt.close()
    
    # 8. Göğüs ağrısı tipi ve hastalık ilişkisi
    plt.figure(figsize=(10, 6))
    cp_target = pd.crosstab(df['cp'], df['target'])
    cp_target.columns = ['Hastalık Yok', 'Hastalık Var']
    cp_target.plot(kind='bar', stacked=True, colormap='Blues')
    plt.title('Göğüs Ağrısı Tipine Göre Kalp Hastalığı Dağılımı', fontsize=16)
    plt.xlabel('Göğüs Ağrısı Tipi', fontsize=12)
    plt.ylabel('Kişi Sayısı', fontsize=12)
    plt.xticks(ticks=range(4), labels=['Tipik Anjina', 'Atipik Anjina', 'Anjinal Olmayan', 'Asemptomatik'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('images/gogus_agrisi_hastalık.png')
    plt.close()
    
    # 9. Sayısal değişkenlerin dağılımı
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'{col} Dağılımı', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frekans', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('images/sayisal_degiskenler.png')
    plt.close()
    
    # 10. Kutu grafikleri ile aykırı değer analizi
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='target', y=col, data=df, palette='viridis')
        plt.title(f'{col} - Hastalık İlişkisi', fontsize=14)
        plt.xlabel('Kalp Hastalığı (1 = Var, 0 = Yok)', fontsize=12)
        plt.ylabel(col, fontsize=12)
        
    plt.tight_layout()
    plt.savefig('images/kutu_grafikleri.png')
    plt.close()


if __name__ == "__main__":
    main() 