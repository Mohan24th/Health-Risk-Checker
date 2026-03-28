import joblib
import pandas as pd
import numpy as np

from app.model.preprocessing import OutlierClipper

# Load model
model = joblib.load("app/model/model.pkl")


def predict(data: dict):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }