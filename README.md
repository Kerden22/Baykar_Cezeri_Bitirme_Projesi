# Meme Kanseri Teşhis Sistemi - Yapay Zeka Uzmanlık Projesi

----------

Bu proje, Sanayi ve Teknoloji Bakanlığı’nın Millî Teknoloji Hamlesi vizyonuyla yürütülen Yapay Zekâ Uzmanlık Programı kapsamında, Baykar – Cezeri iştiraki tarafından sağlanan bitirme projesi çerçevesinde hazırlanmıştır.

----------

## 📊 Proje Amacı

Bu proje, meme kanseri hastalığının tespiti ve sınıflandırılması için klasik makine öğrenmesi ve derin öğrenme modellerini kullanarak yapay zekâ tabanlı bir karar destek sistemi geliştirmeyi hedefler. Wisconsin Breast Cancer (Diagnostic) veri seti kullanılmıştır.

----------

## 🔧 Kullanılan Teknolojiler

-   **Python 3.x**
    
-   **Jupyter Notebook** (Veri analizi ve model eğitimi)
    
-   **Streamlit** (Web tabanlı arayüz geliştirme)
    
-   **TensorFlow & Keras** (Derin öğrenme modeli)
    
-   **Scikit-Learn** (Makine öğrenmesi modelleri ve metrikler)
    
-   **XGBoost, CatBoost** (Boosting modelleri)
    
-   **ChromaDB + Google Gemini 2.5 (RAG)** (LLM tabanlı Chatbot)
    
-   **Seaborn & Matplotlib** (Veri görselleştirme)
    
-   **Joblib, JSON, Pickle** (Model kayıt ve yükleme)
    

----------

## 🔢 Proje Dosya Yapısı

```
BAYKAR - CEZERI
├── __pycache__/
├── .venv/                  # Sanal ortam
├── catboost_info/          # CatBoost çalışma dosyaları
├── chroma_db/              # RAG vektör veri tabanı
├── data/
│   ├── data.csv            # Ana veri seti
│   ├── feature_order.csv   # Özellik sırası
│   ├── iyi.csv             # İyi huylu örnek
│   └── kotu.csv            # Kötü huylu örnek
├── docs/
│   ├── meme-kanseri-rehberi.pdf  # Chatbot referans dokümanı
├── notebooks/
│   └── Proje.ipynb         # Tüm model eğitim notebook'u
├── results/                # Model değerlendirme görselleri
├── .env                    # Ortam değişkenleri
├── app.py                  # Streamlit uygulama kodu
├── breast_mlp.h5           # Eğitilmiş MLP modeli
├── Rapor                   # Proje raporu
├── requirements.txt        # Gerekli paketler
├── scaler.pkl              # StandardScaler objesi
├── streamlit_rag.py        # Chatbot kodları
├── threshold.json          # Eşik değeri

```

----------

## 📃 Proje Akışı

### 1) **Veri Analizi (Notebook Aşaması)**

-   Veri temizleme ve ön işleme yapıldı.
    
-   Eksik veriler kontrol edildi.
    
-   EDA çalışmaları gerçekleştirildi (heatmap, PCA, histogram vb.).
    

### 2) **Makine Öğrenmesi Modelleri**

-   Random Forest, Logistic Regression, SVM, KNN, Naive Bayes, XGBoost ve CatBoost eğitildi.
    
-   Modellerin doğruluk oranları karşılaştırıldı.
    
-   Random Forest için GridSearchCV ile hiperparametre optimizasyonu yapıldı.
    

### 3) **Derin Öğrenme (MLP) Modeli**

-   Derin öğrenme modeli eğitildi.
    
-   Eşik değeri precision-recall curve ile optimize edildi.
    
-   Model, `breast_mlp.h5` olarak kaydedildi.
    

### 4) **Model Artefaktları Kaydedildi**

-   scaler.pkl, threshold.json, feature_order.csv çıktıları oluşturuldu.
    

### 5) **Streamlit Web Arayüzü Geliştirildi**

-   Özellik girme ve CSV yükleme ile tahmin alma sistemi yapıldı.
    
-   Veri analizi ve model değerlendirme sekmeleri eklendi.
    
-   Bootstrap tarzı özelleştirilmiş tasarım uygulandı.
    

### 6) **RAG + Gemini 2.5 Tabanlı Chatbot Entegrasyonu**

-   PDF tabanlı bilgi kaynağı ile sadece meme kanseri hakkında bilgi sunan chatbot geliştirildi.
    
-   ChromaDB ile vektör tabanlı bilgi alma sağlandı.
    

### 7) **Detaylı Proje Raporu Yazıldı**

-   Projenin tüm aşamaları belge halinde sunuldu.
    

----------

## 📅 Kurulum ve Çalıştırma

### Ortamı Kurun

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Ortam Değişkenlerini Ayarlayın

-   `.env` dosyası içine Google Gemini API anahtarları ve gerekli değişkenler yazılmalı.

```bash
GOOGLE_API_KEY="<Gemini API Key>"
```
    

### Uygulamayı Çalıştırın

```bash
streamlit run app.py
```

----------

## 🔝 Kullanım Örnekleri

-   **Veri Girişi:** data/ altındaki iyi.csv ve kotu.csv dosyalarını Streamlit arayüzüne yükleyerek tahmin alabilirsiniz.
    
-   **Chatbot Kullanımı:** Meme kanseri hakkında bilgi almak için chatbot sekmesinden sorular sorabilirsiniz.
    
-   **Model Performansı:** Model değerlendirme sekmesinde F1 ve confusion matrix görselleri sunulmaktadır.
    

----------

## 🔹 Geliştirici

## **Mahmut Kerem Erden**  
## [k.erden03@gmail.com](mailto:k.erden03@gmail.com)

----------

