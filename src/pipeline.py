import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.utils import feature_engineering
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH

def run_pipeline():
    # 1️⃣ Ham veriyi oku
    df = pd.read_csv(RAW_DATA_PATH + 'application_train.csv')
    
    # 2️⃣ Feature engineering uygula
    df_fe = feature_engineering(df)
    
    # 3️⃣ Hedef ve feature ayrımı
    X = df_fe.drop(columns=['TARGET'])
    y = df_fe['TARGET']
    
    # 4️⃣ Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5️⃣ Model eğitimi
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6️⃣ Validasyon skorunu hesapla
    y_pred_proba = model.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_proba)
    print(f"Validation ROC-AUC: {auc_score:.4f}")
    
    # 7️⃣ Modeli kaydet
    joblib.dump(model, MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    run_pipeline()
