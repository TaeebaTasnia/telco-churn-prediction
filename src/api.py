import sys
sys.path.insert(0, '.')

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import add_features


# Load pipeline at startup (module level)
pipeline = joblib.load('models/churn_pipeline.pkl')

THRESHOLD = 0.35

app = FastAPI()


class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/predict')
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df)
    prob = pipeline.predict_proba(df)[0][1]
    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": "Yes" if prob >= THRESHOLD else "No"
    }
