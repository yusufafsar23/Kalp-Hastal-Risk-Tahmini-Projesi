"""
Kalp Hastalığı Tahmini - Model Eğitimi ve Hiperparametre Optimizasyonu
Bu script, kalp hastalığı veri seti üzerinde çeşitli makine öğrenmesi modellerini eğitir,
çapraz doğrulama ile değerlendirir ve en iyi modelin hiperparametrelerini optimize eder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Grafik ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("Kalp Hastalığı Tahmini - Model Eğitimi ve Hiperparametre Optimizasyonu")
    print("="*70)
    
    # Klasörleri oluştur
    os.makedirs('models', exist_ok=True)
    os.makedirs('images/model_performance', exist_ok=True)
    
    # 1. İşlenmiş Verileri Yükle
    print("\n1. İşlenmiş Verileri Yükleme")
    print("-"*30)
    
    try:
        with open('data/processed/processed_heart_data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        preprocessing_info = data['preprocessing_info']
        
        print(f"Eğitim veri seti boyutu: {X_train.shape}")
        print(f"Test veri seti boyutu: {X_test.shape}")
        print("Veriler başarıyla yüklendi.")
    except:
        print("İşlenmiş veri bulunamadı! Önce veri_on_isleme.py çalıştırılmalıdır.")
        return
    
    # 2. Model Seçimi
    print("\n2. Model Seçimi")
    print("-"*30)
    
    # Modelleri tanımla
    models = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000, random_state=42),
        'Karar Ağacı': DecisionTreeClassifier(random_state=42),
        'Rastgele Orman': RandomForestClassifier(random_state=42)
    }
    
    print("Değerlendirilecek modeller:")
    for i, model_name in enumerate(models.keys(), 1):
        print(f"{i}. {model_name}")
    
    # 3. Çapraz Doğrulama ile Model Performanslarını Değerlendir
    print("\n3. Çapraz Doğrulama ile Model Performanslarını Değerlendirme")
    print("-"*30)
    
    cv_results = {}
    cv_scores = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name} değerlendiriliyor...")
        
        # 5-katlı çapraz doğrulama ile modeli değerlendir
        start_time = time.time()
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        end_time = time.time()
        
        # Sonuçları kaydet
        cv_results[model_name] = {
            'cv_score_mean': cv_score.mean(),
            'cv_score_std': cv_score.std(),
            'cv_scores': cv_score,
            'time': end_time - start_time
        }
        
        cv_scores[model_name] = cv_score.mean()
        
        print(f"Ortalama Doğruluk: {cv_score.mean():.4f} (±{cv_score.std():.4f})")
        print(f"CV Skorları: {cv_score}")
        print(f"İşlem Süresi: {end_time - start_time:.2f} saniye")
    
    # En iyi modeli bul
    best_model_name = max(cv_scores, key=cv_scores.get)
    best_cv_score = cv_scores[best_model_name]
    
    print(f"\nEn iyi model: {best_model_name} (CV Doğruluk: {best_cv_score:.4f})")
    
    # 4. Çapraz Doğrulama Sonuçlarını Görselleştir
    print("\n4. Çapraz Doğrulama Sonuçlarını Görselleştirme")
    print("-"*30)
    
    # Modellerin çapraz doğrulama skorlarını gösteren çubuk grafik
    plt.figure(figsize=(10, 6))
    cv_means = [cv_results[model]['cv_score_mean'] for model in models.keys()]
    cv_stds = [cv_results[model]['cv_score_std'] for model in models.keys()]
    model_names = list(models.keys())
    
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=10, alpha=0.7)
    
    # En iyi modelin çubuğunu vurgula
    for i, model in enumerate(model_names):
        if model == best_model_name:
            bars[i].set_color('green')
    
    plt.title('Model Karşılaştırma - 5-Katlı Çapraz Doğrulama', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Doğruluk Skoru', fontsize=14)
    plt.ylim(0.7, 1.0)  # Y eksenini 0.7-1.0 arasında sınırla
    plt.grid(axis='y', alpha=0.3)
    
    # Çubukların üzerine değerleri ekle
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/model_performance/model_comparison_cv.png')
    plt.close()
    
    # 5. Hiperparametre Optimizasyonu
    print("\n5. Hiperparametre Optimizasyonu")
    print("-"*30)
    
    # En iyi modelin hiperparametre optimizasyonu
    best_base_model = models[best_model_name]
    
    # Model tipine göre hiperparametre arama uzayını belirle
    if best_model_name == 'Lojistik Regresyon':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        }
        # L1 penalty yalnızca liblinear ve saga solver'ları ile çalışır
        # Elasticnet penalty yalnızca saga solver ile çalışır
        # Newton-cg ve lbfgs solver'ları L1 penalty ile çalışmaz
        
    elif best_model_name == 'Karar Ağacı':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
    elif best_model_name == 'Rastgele Orman':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    print(f"{best_model_name} için hiperparametre optimizasyonu yapılıyor...")
    
    # RandomizedSearchCV ile hiperparametre optimizasyonu yap (GridSearchCV yerine daha hızlı)
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=best_base_model,
        param_distributions=param_grid,
        n_iter=20,  # Rastgele 20 farklı kombinasyon dene
        cv=5,        # 5-katlı çapraz doğrulama
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,   # Tüm CPU çekirdeklerini kullan
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    
    print(f"\nEn iyi hiperparametreler:\n{random_search.best_params_}")
    print(f"En iyi çapraz doğrulama skoru: {random_search.best_score_:.4f}")
    print(f"Optimizasyon süresi: {end_time - start_time:.2f} saniye")
    
    # Optimizasyon sonuçlarını görselleştir
    plt.figure(figsize=(12, 6))
    
    # Sonuçları DataFrame'e dönüştür
    results = pd.DataFrame(random_search.cv_results_)
    
    # Her iterasyonun doğruluk skorunu çiz
    plt.plot(results['mean_test_score'], 'o-', color='blue', alpha=0.7, markersize=8)
    plt.axhline(y=random_search.best_score_, color='r', linestyle='--', alpha=0.7, 
                label=f'En İyi Skor: {random_search.best_score_:.4f}')
    
    plt.title(f'{best_model_name} Hiperparametre Optimizasyonu', fontsize=16)
    plt.xlabel('İterasyon', fontsize=14)
    plt.ylabel('Doğruluk Skoru', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/model_performance/hyperparameter_optimization.png')
    plt.close()
    
    # 6. En İyi Model ile Test Verisi Üzerinde Değerlendirme
    print("\n6. En İyi Model ile Test Verisi Üzerinde Değerlendirme")
    print("-"*30)
    
    # En iyi hiperparametrelerle modeli eğit
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Test verisi üzerinde tahmin yap
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
    
    # Model performansını değerlendir
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Test verisi üzerinde model performansı:")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Hassasiyet (Recall): {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    print(f"AUC Skoru: {auc:.4f}")
    
    print("\nKarmaşıklık Matrisi:")
    print(conf_matrix)
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # 7. Model Performansını Görselleştir
    print("\n7. Model Performansını Görselleştirme")
    print("-"*30)
    
    # Karmaşıklık matrisi görselleştirmesi
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negatif (0)', 'Pozitif (1)'],
                yticklabels=['Negatif (0)', 'Pozitif (1)'])
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.title(f'{best_model_name} - Karmaşıklık Matrisi', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/model_performance/confusion_matrix.png')
    plt.close()
    
    # ROC eğrisi
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Eğrisi (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı', fontsize=12)
    plt.title(f'{best_model_name} - ROC Eğrisi', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/model_performance/roc_curve.png')
    plt.close()
    
    # Önemli metrikleri gösteren çubuk grafik
    plt.figure(figsize=(10, 6))
    metrics = ['Doğruluk', 'Kesinlik', 'Hassasiyet', 'F1 Skoru', 'AUC']
    values = [accuracy, precision, recall, f1, auc]
    
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
    plt.savefig('images/model_performance/performance_metrics.png')
    plt.close()
    
    # 8. En İyi Modeli Kaydet
    print("\n8. En İyi Modeli Kaydetme")
    print("-"*30)
    
    # Eğitilmiş modeli kaydet
    model_info = {
        'model': best_model,
        'model_name': best_model_name,
        'best_params': random_search.best_params_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix,
        'preprocessing_info': preprocessing_info
    }
    
    with open('models/best_heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"En iyi model kaydedildi: models/best_heart_disease_model.pkl")
    print("\nModel eğitimi ve hiperparametre optimizasyonu tamamlandı.")
    print(f"\nEn İyi Model: {best_model_name}")
    print(f"En İyi Hiperparametreler: {random_search.best_params_}")
    print(f"Test Doğruluğu: {accuracy:.4f}")

if __name__ == "__main__":
    main() 