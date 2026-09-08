import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb


NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'avg_monthly_spend', 'num_services', 'has_streaming']

CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'tenure_bucket'
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def build_pipeline(model=None) -> Pipeline:
    if model is None:
        scale_pos_weight = 2.7  # approximate default; train.py overrides this
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_COLS),
        ('cat', categorical_transformer, CATEGORICAL_COLS),
    ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
