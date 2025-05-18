"""
Kalp Hastalığı Tahmini - Model Değerlendirme
Bu script, eğitilmiş ve optimize edilmiş kalp hastalığı tahmin modelini değerlendirir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score
)

# Grafik ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("Kalp Hastalığı Tahmini - Model Değerlendirme")
    print("="*50)
    
    # Klasörleri oluştur
    os.makedirs('images/model_evaluation', exist_ok=True)
    
    # 1. Kaydedilmiş Modeli Yükle
    print("\n1. Kaydedilmiş Modeli Yükleme")
    print("-"*30)
    
    try:
        with open('models/best_heart_disease_model.pkl', 'rb') as f:
            model_info = pickle.load(f)
            
        # Model bilgilerini çıkar
        model = model_info['model']
        model_name = model_info['model_name']
        best_params = model_info['best_params']
        accuracy = model_info['accuracy']
        precision = model_info['precision']
        recall = model_info['recall']
        f1 = model_info['f1']
        auc = model_info['auc']
        conf_matrix = model_info['confusion_matrix']
        
        print(f"Model: {model_name}")
        print(f"En iyi parametreler: {best_params}")
        print("Model başarıyla yüklendi.")
    except:
        print("Kaydedilmiş model bulunamadı! Önce model_egitimi.py çalıştırılmalıdır.")
        return
    
    # 2. İşlenmiş Test Verilerini Yükle
    print("\n2. İşlenmiş Test Verilerini Yükleme")
    print("-"*30)
    
    try:
        with open('data/processed/processed_heart_data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"Test veri seti boyutu: {X_test.shape}")
        print("Test verileri başarıyla yüklendi.")
    except:
        print("İşlenmiş veri bulunamadı! Önce veri_on_isleme.py çalıştırılmalıdır.")
        return
    
    # 3. Test Verisi Üzerinde Tahmin Yap
    print("\n3. Test Verisi Üzerinde Tahmin Yapma")
    print("-"*30)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
    
    # 4. Model Performans Metriklerini Hesapla
    print("\n4. Model Performans Metrikleri")
    print("-"*30)
    
    # Ana metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Sonuçları göster
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    print(f"AUC Skoru: {auc_score:.4f}")
    print(f"Ortalama Kesinlik (AP) Skoru: {avg_precision:.4f}")
    
    # 5. Karmaşıklık Matrisi
    print("\n5. Karmaşıklık Matrisi (Confusion Matrix)")
    print("-"*30)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Karmaşıklık matrisinden değerleri çıkar
    tn, fp, fn, tp = conf_matrix.ravel()
    
    print(f"Karmaşıklık Matrisi:")
    print(f"[[{tn}, {fp}]")
    print(f" [{fn}, {tp}]]")
    
    print(f"\nDoğru Negatif (TN): {tn}")
    print(f"Yanlış Pozitif (FP): {fp}")
    print(f"Yanlış Negatif (FN): {fn}")
    print(f"Doğru Pozitif (TP): {tp}")
    
    # Ek metrikler
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    negative_predictive_value = tn / (tn + fn)
    
    print(f"\nÖzgüllük (Specificity): {specificity:.4f}")
    print(f"Yanlış Pozitif Oranı (FPR): {false_positive_rate:.4f}")
    print(f"Negatif Tahmin Değeri (NPV): {negative_predictive_value:.4f}")
    
    # 6. Sınıflandırma Raporu
    print("\n6. Sınıflandırma Raporu (Classification Report)")
    print("-"*30)
    
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    
    # 7. Görselleştirmeler
    print("\n7. Model Performansı Görselleştirmeleri")
    print("-"*30)
    
    # 7.1. Karmaşıklık Matrisi Görselleştirmesi
    plt.figure(figsize=(8, 7))
    
    # Daha detaylı bir heatmap oluştur
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negatif (0)', 'Pozitif (1)'],
                yticklabels=['Negatif (0)', 'Pozitif (1)'])
    
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.title(f'{model_name} - Karmaşıklık Matrisi', fontsize=14)
    
    # Ek bilgileri grafiğe ekle
    plt.figtext(0.5, 0.01, 
               f"Doğruluk: {accuracy:.4f} | Kesinlik: {precision:.4f} | Duyarlılık: {recall:.4f} | F1: {f1:.4f}", 
               ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/confusion_matrix_detailed.png')
    plt.close()
    
    # 7.2. Performans Metrikleri Çubuğu
    plt.figure(figsize=(12, 6))
    
    metrics = ['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1 Skoru', 'AUC', 'AP', 'Özgüllük']
    values = [accuracy, precision, recall, f1, auc_score, avg_precision, specificity]
    
    # Metrik tiplerine göre renklendirme
    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#3498db']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    
    plt.title(f'{model_name} - Performans Metrikleri', fontsize=16)
    plt.ylabel('Skor', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Çubukların üzerine değerleri ekle
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/performance_metrics_detailed.png')
    plt.close()
    
    # 7.3. ROC Eğrisi
    plt.figure(figsize=(10, 8))
    
    # ROC eğrisi için gerekli hesaplamalar
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Grafiği çiz
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Eğrisi (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Rastgele Tahmin')
    
    # Grafiği güzelleştir
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (1 - Özgüllük)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (Duyarlılık)', fontsize=12)
    plt.title(f'{model_name} - ROC Eğrisi', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/roc_curve_detailed.png')
    plt.close()
    
    # 7.4. Precision-Recall Eğrisi
    plt.figure(figsize=(10, 8))
    
    # PR eğrisi için gerekli hesaplamalar
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Grafiği çiz
    plt.plot(recalls, precisions, color='green', lw=2, 
             label=f'Precision-Recall Eğrisi (AP = {avg_precision:.4f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='gray', linestyle='--', 
                label=f'Rastgele Tahmin (Sınıf Dağılımı = {sum(y_test)/len(y_test):.4f})')
    
    # Grafiği güzelleştir
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Duyarlılık (Recall)', fontsize=12)
    plt.ylabel('Kesinlik (Precision)', fontsize=12)
    plt.title(f'{model_name} - Precision-Recall Eğrisi', fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/model_evaluation/precision_recall_curve.png')
    plt.close()
    
    # 8. Değerlendirme Sonuçları
    print("\n8. Değerlendirme Sonuçları Özeti")
    print("-"*30)
    
    # Başarı değerlendirmesi
    print("Model Değerlendirme Sonuçları:")
    print(f"- Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"- Kesinlik (Precision): {precision:.4f}")
    print(f"- Duyarlılık (Recall): {recall:.4f}")
    print(f"- F1 Skoru: {f1:.4f}")
    print(f"- AUC Skoru: {auc_score:.4f}")
    
    # Eşik değeri analizi
    print("\nEşik değeri analizi:")
    thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds_to_check:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
        precision_threshold = precision_score(y_test, y_pred_threshold)
        recall_threshold = recall_score(y_test, y_pred_threshold)
        f1_threshold = f1_score(y_test, y_pred_threshold)
        
        print(f"\nEşik değeri {threshold} için performans:")
        print(f"- Doğruluk: {accuracy_threshold:.4f}")
        print(f"- Kesinlik: {precision_threshold:.4f}")
        print(f"- Duyarlılık: {recall_threshold:.4f}")
        print(f"- F1 Skoru: {f1_threshold:.4f}")
    
    # Sonuç
    print("\nModel değerlendirmesi tamamlandı. Görselleştirmeler 'images/model_evaluation/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 