import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# images klasörünü oluştur
os.makedirs('images', exist_ok=True)

# Veri setini oku
df = pd.read_csv('data/kalp_hastaligi/heart.csv')

print("Veri seti başarıyla yüklendi!")
print(f"Veri seti boyutu: {df.shape}")

# Temel istatistikler
print("\nTemel istatistikler:")
print(df.describe())

# Eksik değerleri kontrol et
eksik_deger = df.isnull().sum()
eksik_deger_yuzdesi = (df.isnull().sum() / len(df)) * 100

print("\nEksik değerlerin kontrolü:")
print(pd.DataFrame({
    'Eksik Değer Sayısı': eksik_deger,
    'Eksik Değer Yüzdesi': eksik_deger_yuzdesi
}))

# Grafik ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

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

plt.savefig('images/1_hedef_degisken_dagilimi.png', bbox_inches='tight', dpi=300)
plt.close()

# Yüzde olarak
target_yuzde = df['target'].value_counts(normalize=True) * 100
print(f"\nKalp hastalığı olmayan yüzdesi: %{target_yuzde[0]:.2f}")
print(f"Kalp hastalığı olan yüzdesi: %{target_yuzde[1]:.2f}")

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
plt.savefig('images/2_cinsiyet_dagilimi.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Sayısal değişkenlerin dağılımı
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'{col} Dağılımı', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frekans', fontsize=12)
    
plt.tight_layout()
plt.savefig('images/3_sayisal_degisken_dagilimi.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Yaş analizi
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20)
plt.title('Yaşa Göre Kalp Hastalığı Dağılımı', fontsize=16)
plt.xlabel('Yaş', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)
plt.legend(['Hastalık Yok', 'Hastalık Var'])
plt.savefig('images/4_yas_dagilimi.png', bbox_inches='tight', dpi=300)
plt.close()

# Yaş gruplarına göre kalp hastalığı analizi
df['yas_grubu'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '>60'])
yas_hastalık = pd.crosstab(df['yas_grubu'], df['target'])
yas_hastalık.columns = ['Hastalık Yok', 'Hastalık Var']

# Çapraz tablo
print("\nYaş gruplarına göre kalp hastalığı dağılımı:")
print(yas_hastalık)

# Görselleştirme
plt.figure(figsize=(10, 6))
yas_hastalık.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Yaş Gruplarına Göre Kalp Hastalığı Dağılımı', fontsize=16)
plt.xlabel('Yaş Grubu', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('images/5_yas_grubu_dagilimi.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Korelasyon matrisi
df_numeric = df.drop('yas_grubu', axis=1)  # Kategorik sütunu kaldır
correlation_matrix = df_numeric.corr()

# Korelasyon ısı haritası
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi', fontsize=18)
plt.savefig('images/6_korelasyon_matrisi.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Hedef değişken ile korelasyon
hedef_korelasyon = correlation_matrix['target'].sort_values(ascending=False)
print("\nHedef değişken ile korelasyonlar:")
print(hedef_korelasyon)

# Hedef değişken ile korelasyon grafiği
plt.figure(figsize=(12, 8))
hedef_korelasyon.drop('target').plot(kind='bar', colormap='viridis')
plt.title('Değişkenlerin Kalp Hastalığı ile Korelasyonu', fontsize=16)
plt.xlabel('Değişkenler', fontsize=12)
plt.ylabel('Korelasyon Katsayısı', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('images/7_hedef_korelasyon.png', bbox_inches='tight', dpi=300)
plt.close()

# 7. Yaş ve Maksimum Kalp Hızı İlişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='viridis', s=80, alpha=0.7)
plt.title('Yaş ve Maksimum Kalp Hızı İlişkisi', fontsize=16)
plt.xlabel('Yaş', fontsize=12)
plt.ylabel('Maksimum Kalp Hızı', fontsize=12)
plt.legend(['Hastalık Yok', 'Hastalık Var'])
plt.grid(alpha=0.3)
plt.savefig('images/8_yas_kalp_hizi.png', bbox_inches='tight', dpi=300)
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
plt.savefig('images/9_gogus_agrisi.png', bbox_inches='tight', dpi=300)
plt.close()

# 9. Kolesterol ve kan basıncı ilişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='chol', y='trestbps', hue='target', data=df, palette='viridis', s=80, alpha=0.7)
plt.title('Kolesterol ve Kan Basıncı İlişkisi', fontsize=16)
plt.xlabel('Kolesterol (mg/dl)', fontsize=12)
plt.ylabel('Dinlenme Kan Basıncı (mm Hg)', fontsize=12)
plt.legend(['Hastalık Yok', 'Hastalık Var'])
plt.grid(alpha=0.3)
plt.savefig('images/10_kolesterol_kan_basinci.png', bbox_inches='tight', dpi=300)
plt.close()

# 10. Pair plot
selected_features = ['age', 'thalach', 'chol', 'oldpeak', 'target']
sns.pairplot(df[selected_features], hue='target', palette='viridis')
plt.suptitle('Seçili Değişkenlerin Pair Plot', y=1.02, fontsize=16)
plt.savefig('images/11_pair_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# 11. Kutu grafikleri
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='target', y=col, data=df, hue='target', palette='viridis', legend=False)
    plt.title(f'{col} - Hastalık İlişkisi', fontsize=14)
    plt.xlabel('Kalp Hastalığı (1 = Var, 0 = Yok)', fontsize=12)
    plt.ylabel(col, fontsize=12)
    
plt.tight_layout()
plt.savefig('images/12_kutu_grafikleri.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nTüm grafikler başarıyla oluşturuldu ve 'images/' klasörüne kaydedildi!") 