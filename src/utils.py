import pandas as pd

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
