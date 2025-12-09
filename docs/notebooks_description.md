### 01_eda.ipynb – Exploratory Data Analysis (EDA)

Bu notebook’ta veri setinin detaylı keşfi yapılır.

İçerik:

Veri yükleme ve temel kontroller

Eksik değer analizi

Sınıf dağılımları

Sayısal değişkenlerin istatistikleri

Korelasyon matrisi

Kutu grafikleri, histogramlar, dağılım grafikleri

Hedef değişken analizleri

Amaç:
Veriyi anlamak, olası sorunları tespit etmek ve modelleme sürecine hazırlık yapmak.

### 02_baseline.ipynb – Baseline Model Oluşturma

Bu notebook minimal veri işleme ve basit bir model kurarak performansın taban seviyesini belirler.

İçerik:

Basit preprocessing

Train-test split

Logistic Regression / Decision Tree gibi temel modeller

İlk accuracy, precision, recall sonuçları

İlk confusion matrix

Amaç:
İyileştirmeleri karşılaştırmak için referans bir başlangıç performansı oluşturmak.

### 03_feature_engineering.ipynb – Feature Engineering

Bu notebook özellik çıkarımı ve dönüştürme tekniklerini içerir.

İçerik:

Eksik değer doldurma stratejileri

Kategori encoding (One-Hot, Target Encoding, vb.)

Standartlaştırma / normalize etme

Yeni değişken üretme

Outlier işlemleri

PCA gibi boyut indirgeme işlemleri

Amaç:
Model performansını artıracak daha güçlü veri temsilleri elde etmek.

### 04_model_optimization.ipynb – Model Optimization & Tuning

Bu notebook model iyileştirme ve hiperparametre optimizasyonunu içerir.

İçerik:

GridSearchCV / RandomizedSearchCV

XGBoost, LightGBM, RandomForest gibi gelişmiş modeller

Cross-validation sonuçları

En iyi parametre setinin seçilmesi

Feature importance grafikleri

Amaç:
En yüksek performans veren modeli elde etmek.

### 05_evaluation.ipynb – Model Değerlendirme

Bu notebook modelin derinlemesine değerlendirilmesini yapar.

İçerik:

Accuracy / F1 / Recall / Precision

ROC-AUC grafiği

Confusion matrix

SHAP değerleri ile model açıklanabilirliği

Error analysis (yanlış sınıflanan örnekler)

Amaç:
Modelin güçlü ve zayıf yanlarını belirlemek, doğruluğu yorumlamak ve modelin güvenilirliğini ölçmek.

### 06_pipeline.ipynb – Pipeline & Deployment Hazırlığı

Bu notebook tüm süreci kapsayan bir pipeline oluşturur.

İçerik:

Feature engineering + model + evaluation entegrasyonu

sklearn Pipeline yapısı

Modelin kaydedilmesi (joblib.pkl)

API/Streamlit için örnek giriş-çıkış testleri

Amaç:
Üretim ortamında kullanılabilir tek parça bir makine öğrenmesi pipeline’ı oluşturmak.