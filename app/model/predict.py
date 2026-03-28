import joblib
import pandas as pd

# Load model
model = joblib.load("app/model/model.pkl")

def generate_explanation(data: dict):
    reasons = []

    if data["glucose_level"] > 140:
        reasons.append("High glucose level")

    if data["bmi"] > 30:
        reasons.append("High BMI (Obese)")
    elif data["bmi"] > 25 and data["bmi"] <= 30:
        reasons.append("Overweight")
    else:
        reasons.append("Normal BMI")


    if data["blood_pressure"] > 140:
        reasons.append("High blood pressure")

    if data["age"] > 50:
        reasons.append("Higher age risk")

    if data["smoking_status"].lower() == "smoker":
        reasons.append("Smoking habit")

    return reasons


def predict(data: dict):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    confidence = float(max(probability))

    risk = "High" if prediction == 1 else "Low"

    reasons = generate_explanation(data)

    return {
        "prediction": int(prediction),
        "risk": risk,
        "confidence": round(confidence, 2),
        "reasons": reasons
    }