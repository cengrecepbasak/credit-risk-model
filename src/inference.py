import joblib
import pandas as pd
from src.utils import feature_engineering
from src.config import MODEL_PATH, DEFAULT_THRESHOLD

def predict(df):
    model = joblib.load(MODEL_PATH)
    df_fe = feature_engineering(df)
    preds_proba = model.predict_proba(df_fe)[:,1]
    preds_class = (preds_proba > DEFAULT_THRESHOLD).astype(int)
    df['Predicted_Class'] = preds_class
    df['Predicted_Probability'] = preds_proba
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/application_test.csv')
    result = predict(df)
    result.to_csv('data/processed/application_test_pred.csv', index=False)
    print("Tahmin dosyası kaydedildi: application_test_pred.csv")
