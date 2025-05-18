"""
Kalp Hastalığı Tahmini - Model Eğitimi ve Değerlendirme
Bu script, hem model eğitimi hem de değerlendirme adımlarını içerir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score
)

# Grafik ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("Kalp Hastalığı Tahmini - Model Eğitimi ve Değerlendirme")
    print("="*70)
    
    # Klasörleri oluştur
    os.makedirs('images/model_evaluation', exist_ok=True)
    
    # 1. Veri Setini Yükle ve Ön İşle
    print("\n1. Veri Setini Yükleme ve Ön İşleme")
    print("-"*30)
    
    # Veri setini oku
    df = pd.read_csv('data/kalp_hastaligi/heart.csv')
    print(f"Veri seti boyutu: {df.shape}")
    
    # Kategorik ve sayısal değişkenleri belirleme
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    target = 'target'
    
    # Özellikleri ve hedefi ayır
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Ön işleme pipeline'ı oluştur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Verileri ön işle
    X_processed = preprocessor.fit_transform(X)
    
    # Eğitim ve test setlerine böl
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # 2. Model Seçimi ve Çapraz Doğrulama
    print("\n2. Model Seçimi ve Çapraz Doğrulama")
    print("-"*30)
    
    # Modelleri tanımla
    models = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000, random_state=42),
        'Karar Ağacı': DecisionTreeClassifier(random_state=42),
        'Rastgele Orman': RandomForestClassifier(random_state=42)
    }
    
    # Her modeli çapraz doğrulama ile değerlendir
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name} değerlendiriliyor...")
        
        # 5-katlı çapraz doğrulama
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_results[model_name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"Ortalama Doğruluk: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # En iyi modeli seç
    best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean_score'])
    best_cv_score = cv_results[best_model_name]['mean_score']
    
    print(f"\nEn iyi model: {best_model_name} (CV Doğruluk: {best_cv_score:.4f})")
    
    # 3. En İyi Modeli Eğit
    print("\n3. En İyi Modeli Eğitme")
    print("-"*30)
    
    # En iyi modeli seç ve eğit
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    # 4. Test Verisi Üzerinde Değerlendirme
    print("\n4. Test Verisi Üzerinde Değerlendirme")
    print("-"*30)
    
    # Test verisinde tahmin yap
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
    
    # 5. Model Performans Metrikleri
    print("\n5. Model Performans Metrikleri")
    print("-"*30)
    
    # Ana metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    print(f"AUC Skoru: {auc_score:.4f}")
    
    # 6. Karmaşıklık Matrisi
    print("\n6. Karmaşıklık Matrisi (Confusion Matrix)")
    print("-"*30)
    
    # Karmaşıklık matrisi oluştur
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    print("Karmaşıklık Matrisi:")
    print(conf_matrix)
    
    print(f"\nDoğru Negatif (TN): {tn}")
    print(f"Yanlış Pozitif (FP): {fp}")
    print(f"Yanlış Negatif (FN): {fn}")
    print(f"Doğru Pozitif (TP): {tp}")
    
    # Ek metrikler
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    
    print(f"\nÖzgüllük (Specificity): {specificity:.4f}")
    print(f"Yanlış Pozitif Oranı (FPR): {false_positive_rate:.4f}")
    
    # 7. Sınıflandırma Raporu
    print("\n7. Sınıflandırma Raporu (Classification Report)")
    print("-"*30)
    
    print(classification_report(y_test, y_pred))
    
    # 8. Görselleştirmeler
    print("\n8. Model Performansı Görselleştirmeleri")
    print("-"*30)
    
    # 8.1. Karmaşıklık Matrisi Görselleştirmesi
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negatif (0)', 'Pozitif (1)'],
                yticklabels=['Negatif (0)', 'Pozitif (1)'])
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.title(f'{best_model_name} - Karmaşıklık Matrisi', fontsize=14)
    
    # Ek bilgileri grafiğe ekle
    plt.figtext(0.5, 0.01, 
                f"Doğruluk: {accuracy:.4f} | Kesinlik: {precision:.4f} | Duyarlılık: {recall:.4f} | F1: {f1:.4f}", 
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/confusion_matrix.png')
    plt.close()
    
    # 8.2. ROC Eğrisi
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Eğrisi (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Rastgele Tahmin')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (1 - Özgüllük)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (Duyarlılık)', fontsize=12)
    plt.title(f'{best_model_name} - ROC Eğrisi', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/model_evaluation/roc_curve.png')
    plt.close()
    
    # 8.3. Performans Metrikleri Grafiği
    plt.figure(figsize=(10, 6))
    metrics = ['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1 Skoru', 'AUC', 'Özgüllük']
    values = [accuracy, precision, recall, f1, auc_score, specificity]
    
    bars = plt.bar(metrics, values, color='blue', alpha=0.7)
    
    plt.title(f'{best_model_name} - Performans Metrikleri', fontsize=16)
    plt.ylabel('Skor', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Çubukların üzerine değerleri ekle
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/performance_metrics.png')
    plt.close()
    
    # 9. Değerlendirme Sonuçları
    print("\n9. Değerlendirme Sonuçları Özeti")
    print("-"*30)
    
    print("Model Değerlendirme Sonuçları:")
    print(f"- En iyi model: {best_model_name}")
    print(f"- Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"- Kesinlik (Precision): {precision:.4f}")
    print(f"- Duyarlılık (Recall): {recall:.4f}")
    print(f"- F1 Skoru: {f1:.4f}")
    print(f"- AUC Skoru: {auc_score:.4f}")
    
    print("\nModel eğitimi ve değerlendirmesi tamamlandı.")
    print("Görselleştirmeler 'images/model_evaluation/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 