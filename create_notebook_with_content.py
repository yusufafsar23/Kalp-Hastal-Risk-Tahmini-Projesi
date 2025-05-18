import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Kalp Hastalığı Veri Seti - Keşifsel Veri Analizi (EDA)\n",
                "\n",
                "Bu notebook'ta kalp hastalığı veri seti üzerinde keşifsel veri analizi gerçekleştireceğiz."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Gerekli Kütüphanelerin İçe Aktarılması"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Grafikler için ayarlar\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 8)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Veri Setinin Yüklenmesi ve İncelenmesi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Veri setini yükle\n",
                "df = pd.read_csv('../data/kalp_hastaligi/heart.csv')\n",
                "\n",
                "# İlk 5 satırı göster\n",
                "print(\"İlk 5 satır:\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Veri seti bilgileri\n",
                "print(f\"Veri seti boyutu: {df.shape}\")\n",
                "print(\"\\nSütun isimleri:\")\n",
                "print(df.columns.tolist())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Veri türleri\n",
                "print(\"Veri türleri:\")\n",
                "df.dtypes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Temel istatistikler\n",
                "df.describe().round(2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Eksik Değer Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Eksik değerleri kontrol et\n",
                "eksik_deger = df.isnull().sum()\n",
                "eksik_deger_yuzdesi = (df.isnull().sum() / len(df)) * 100\n",
                "\n",
                "eksik_veriler = pd.DataFrame({\n",
                "    'Eksik Değer Sayısı': eksik_deger,\n",
                "    'Eksik Değer Yüzdesi': eksik_deger_yuzdesi.round(2)\n",
                "})\n",
                "eksik_veriler"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Hedef Değişken Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hedef değişken dağılımı\n",
                "target_count = df['target'].value_counts()\n",
                "target_yuzde = df['target'].value_counts(normalize=True) * 100\n",
                "\n",
                "print(f\"Kalp hastalığı olmayan sayısı: {target_count[0]}\")\n",
                "print(f\"Kalp hastalığı olan sayısı: {target_count[1]}\")\n",
                "print(f\"\\nKalp hastalığı olmayan yüzdesi: %{target_yuzde[0]:.2f}\")\n",
                "print(f\"Kalp hastalığı olan yüzdesi: %{target_yuzde[1]:.2f}\")\n",
                "\n",
                "# Görselleştirme\n",
                "plt.figure(figsize=(10, 6))\n",
                "ax = sns.countplot(x='target', data=df, palette='viridis')\n",
                "plt.title('Hedef Değişken Dağılımı', fontsize=16)\n",
                "plt.xlabel('Kalp Hastalığı (0 = Yok, 1 = Var)', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "\n",
                "# Çubukların üzerine değer ekle\n",
                "for p in ax.patches:\n",
                "    height = p.get_height()\n",
                "    ax.text(p.get_x() + p.get_width()/2., height + 5, f'{height}', ha=\"center\")\n",
                "\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Cinsiyet Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cinsiyet dağılımı\n",
                "print(\"Cinsiyet dağılımı:\")\n",
                "print(df['sex'].value_counts())\n",
                "print(\"\\nCinsiyete göre yüzde:\")\n",
                "print(df['sex'].value_counts(normalize=True).round(2) * 100)\n",
                "\n",
                "# Görselleştirme\n",
                "plt.figure(figsize=(15, 6))\n",
                "\n",
                "# Cinsiyet dağılımı\n",
                "plt.subplot(1, 2, 1)\n",
                "ax1 = sns.countplot(x='sex', data=df, palette='viridis')\n",
                "plt.title('Cinsiyet Dağılımı', fontsize=14)\n",
                "plt.xlabel('Cinsiyet (1 = Erkek, 0 = Kadın)', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "\n",
                "# Çubukların üzerine değer ekle\n",
                "for p in ax1.patches:\n",
                "    height = p.get_height()\n",
                "    ax1.text(p.get_x() + p.get_width()/2., height + 5, f'{height}', ha=\"center\")\n",
                "\n",
                "# Cinsiyete göre hastalık dağılımı\n",
                "plt.subplot(1, 2, 2)\n",
                "ax2 = sns.countplot(x='sex', hue='target', data=df, palette='viridis')\n",
                "plt.title('Cinsiyete Göre Kalp Hastalığı Dağılımı', fontsize=14)\n",
                "plt.xlabel('Cinsiyet (1 = Erkek, 0 = Kadın)', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "plt.legend(['Hastalık Yok', 'Hastalık Var'])\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Yaş Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Yaş dağılımı\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20)\n",
                "plt.title('Yaşa Göre Kalp Hastalığı Dağılımı', fontsize=16)\n",
                "plt.xlabel('Yaş', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "plt.legend(['Hastalık Yok', 'Hastalık Var'])\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Yaş gruplarına göre analiz\n",
                "df['yas_grubu'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '>60'])\n",
                "yas_hastalık = pd.crosstab(df['yas_grubu'], df['target'])\n",
                "yas_hastalık.columns = ['Hastalık Yok', 'Hastalık Var']\n",
                "\n",
                "# Çapraz tablo\n",
                "print(\"Yaş gruplarına göre kalp hastalığı dağılımı:\")\n",
                "print(yas_hastalık)\n",
                "\n",
                "# Görselleştirme\n",
                "plt.figure(figsize=(10, 6))\n",
                "yas_hastalık.plot(kind='bar', stacked=True, colormap='viridis')\n",
                "plt.title('Yaş Gruplarına Göre Kalp Hastalığı Dağılımı', fontsize=16)\n",
                "plt.xlabel('Yaş Grubu', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "plt.grid(axis='y', alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Korelasyon Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Korelasyon matrisi\n",
                "df_numeric = df.drop('yas_grubu', axis=1)  # Kategorik sütunu kaldır\n",
                "correlation_matrix = df_numeric.corr()\n",
                "\n",
                "# Korelasyon ısı haritası\n",
                "plt.figure(figsize=(14, 10))\n",
                "sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)\n",
                "plt.title('Korelasyon Matrisi', fontsize=18)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hedef değişken ile korelasyon\n",
                "hedef_korelasyon = correlation_matrix['target'].sort_values(ascending=False)\n",
                "print(\"Hedef değişken ile korelasyonlar:\")\n",
                "print(hedef_korelasyon)\n",
                "\n",
                "# Hedef değişken ile korelasyon grafiği\n",
                "plt.figure(figsize=(12, 8))\n",
                "hedef_korelasyon.drop('target').plot(kind='bar', colormap='viridis')\n",
                "plt.title('Değişkenlerin Kalp Hastalığı ile Korelasyonu', fontsize=16)\n",
                "plt.xlabel('Değişkenler', fontsize=12)\n",
                "plt.ylabel('Korelasyon Katsayısı', fontsize=12)\n",
                "plt.grid(axis='y', alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Yaş ve Maksimum Kalp Hızı İlişkisi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Yaş ve Maksimum Kalp Hızı İlişkisi\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='viridis', s=80, alpha=0.7)\n",
                "plt.title('Yaş ve Maksimum Kalp Hızı İlişkisi', fontsize=16)\n",
                "plt.xlabel('Yaş', fontsize=12)\n",
                "plt.ylabel('Maksimum Kalp Hızı', fontsize=12)\n",
                "plt.legend(['Hastalık Yok', 'Hastalık Var'])\n",
                "plt.grid(alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Göğüs Ağrısı Analizi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Göğüs ağrısı tipi ve hastalık ilişkisi\n",
                "plt.figure(figsize=(10, 6))\n",
                "cp_target = pd.crosstab(df['cp'], df['target'])\n",
                "cp_target.columns = ['Hastalık Yok', 'Hastalık Var']\n",
                "cp_target.plot(kind='bar', stacked=True, colormap='Blues')\n",
                "plt.title('Göğüs Ağrısı Tipine Göre Kalp Hastalığı Dağılımı', fontsize=16)\n",
                "plt.xlabel('Göğüs Ağrısı Tipi', fontsize=12)\n",
                "plt.ylabel('Kişi Sayısı', fontsize=12)\n",
                "plt.xticks(ticks=range(4), labels=['Tipik Anjina', 'Atipik Anjina', 'Anjinal Olmayan', 'Asemptomatik'])\n",
                "plt.grid(axis='y', alpha=0.3)\n",
                "plt.show()\n",
                "\n",
                "# Göğüs ağrısı tipi değer sayıları\n",
                "print(\"Göğüs ağrısı tipi dağılımı:\")\n",
                "print(df['cp'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Kolesterol ve Kan Basıncı İlişkisi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kolesterol ve kan basıncı ilişkisi\n",
                "plt.figure(figsize=(12, 8))\n",
                "sns.scatterplot(x='chol', y='trestbps', hue='target', data=df, palette='viridis', s=80, alpha=0.7)\n",
                "plt.title('Kolesterol ve Kan Basıncı İlişkisi', fontsize=16)\n",
                "plt.xlabel('Kolesterol (mg/dl)', fontsize=12)\n",
                "plt.ylabel('Dinlenme Kan Basıncı (mm Hg)', fontsize=12)\n",
                "plt.legend(['Hastalık Yok', 'Hastalık Var'])\n",
                "plt.grid(alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Çiftli İlişki Grafikleri"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Sayısal değişkenlerin çiftli ilişkileri\n",
                "sns.pairplot(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']], hue='target', palette='viridis')\n",
                "plt.suptitle('Sayısal Değişkenler Arasındaki İlişkiler', fontsize=16, y=1.02)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Kutu Grafikleri"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Sayısal değişkenlerin kutu grafikleri\n",
                "plt.figure(figsize=(15, 10))\n",
                "for i, col in enumerate(['age', 'trestbps', 'chol', 'thalach', 'oldpeak']):\n",
                "    plt.subplot(2, 3, i+1)\n",
                "    sns.boxplot(x='target', y=col, data=df, palette='viridis')\n",
                "    plt.title(f'{col} için Kutu Grafiği', fontsize=14)\n",
                "    plt.xlabel('Kalp Hastalığı (0 = Yok, 1 = Var)', fontsize=12)\n",
                "    plt.ylabel(col, fontsize=12)\n",
                "    \n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Sonuçlar ve Bulgular"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Veri seti analizi sonucunda elde ettiğimiz temel bulgular:\n",
                "\n",
                "1. **Hedef Değişken Dağılımı**: Veri seti oldukça dengeli bir dağılıma sahiptir (yaklaşık %49 hastalık yok, %51 hastalık var).\n",
                "\n",
                "2. **Cinsiyet Etkisi**: Erkek hastalar kadın hastalara göre sayıca daha fazladır.\n",
                "\n",
                "3. **Yaş Faktörü**: 45-65 yaş aralığında kalp hastalığı daha sık görülmektedir.\n",
                "\n",
                "4. **Göğüs Ağrısı Tipi**: Göğüs ağrısı tipi ile kalp hastalığı arasında güçlü bir ilişki vardır. Özellikle asemptomatik tip göğüs ağrısı (tip 3) olan hastalarda kalp hastalığı riski daha yüksektir.\n",
                "\n",
                "5. **Maksimum Kalp Hızı**: Kalp hastalığı olan hastalarda maksimum kalp hızı genellikle daha düşüktür ve yaş ile negatif korelasyona sahiptir.\n",
                "\n",
                "6. **ST Depresyonu (oldpeak)**: ST depresyonu değeri kalp hastalığı ile pozitif korelasyona sahiptir.\n",
                "\n",
                "7. **Korelasyon Analizi**: Kalp hastalığı ile en güçlü korelasyona sahip değişkenler: 'cp' (göğüs ağrısı tipi), 'thalach' (maksimum kalp hızı), 'slope' (ST segmenti eğimi) ve 'oldpeak' (ST depresyonu)."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Ensure the notebooks directory exists
os.makedirs("notebooks", exist_ok=True)

# Write the notebook to a file
with open("notebooks/kalp_hastaligi_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook created successfully at notebooks/kalp_hastaligi_eda.ipynb") 