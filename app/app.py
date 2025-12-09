# app.py - Streamlit Arayüzü ile Credit Risk Prediction
import streamlit as st
import pandas as pd
import joblib
import os

# Sayfa başlığı
st.title("Credit Risk Prediction")
st.write("Müşterinin kredi geri ödememe (default) riskini tahmin edebilirsiniz.")

# Model yolu
model_path = 'models/final_pipeline_model.pkl'
if not os.path.exists(model_path):
    st.error("Model bulunamadı. Lütfen pipeline notebook ile modeli oluşturun.")
    st.stop()

# Modeli yükle
model = joblib.load(model_path)

# Kullanıcıdan CSV yüklemesini iste
uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type="csv")

if uploaded_file is not None:
    # Veriyi oku
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:"
, df.head())

    # Feature engineering (pipeline fonksiyonunu kopyaladık)
    def feature_engineering(df):
        df_fe = df.copy()
        df_fe['CREDIT_INCOME_RATIO'] = df_fe['AMT_CREDIT'] / (df_fe['AMT_INCOME_TOTAL'] + 1)
        df_fe['ANNUITY_INCOME_RATIO'] = df_fe['AMT_ANNUITY'] / (df_fe['AMT_INCOME_TOTAL'] + 1)
        df_fe['CREDIT_ANNUITY_RATIO'] = df_fe['AMT_CREDIT'] / (df_fe['AMT_ANNUITY'] + 1)
        ext_cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
        df_fe[ext_cols] = df_fe[ext_cols].fillna(df_fe[ext_cols].median())
        df_fe['EXT_SOURCES_MEAN'] = df_fe[ext_cols].mean(axis=1)
        df_fe['EXT_SOURCES_MIN'] = df_fe[ext_cols].min(axis=1)
        df_fe['EXT_SOURCES_MAX'] = df_fe[ext_cols].max(axis=1)
        doc_cols = [col for col in df_fe.columns if 'FLAG_DOCUMENT' in col]
        df_fe['DOCUMENT_COUNT'] = df_fe[doc_cols].sum(axis=1)
        df_fe['CHILDREN_RATIO'] = df_fe['CNT_CHILDREN'] / (df_fe['CNT_FAM_MEMBERS'] + 1)
        df_fe['EMPLOYED_YEARS'] = df_fe['DAYS_EMPLOYED'].apply(lambda x: 0 if x==365243 else x) / -365
        cat_cols = df_fe.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_fe[col] = df_fe[col].astype(str).astype('category').cat.codes
        df_fe = df_fe.fillna(df_fe.median(numeric_only=True))
        return df_fe

    # Feature engineering uygula
    df_fe = feature_engineering(df)

    # Tahmin al
    preds_proba = model.predict_proba(df_fe)[:,1]
    preds_class = (preds_proba > 0.5).astype(int)

    # Sonuçları göster
    df['Predicted_Class'] = preds_class
    df['Predicted_Probability'] = preds_proba
    st.write("Tahmin sonuçları:", df[['Predicted_Class','Predicted_Probability']])

    # İndirilebilir CSV oluştur
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Tahminleri indir", data=csv, file_name='predictions.csv', mime='text/csv')
