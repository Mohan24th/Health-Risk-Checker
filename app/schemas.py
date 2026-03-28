from pydantic import BaseModel

class HealthInput(BaseModel):
    glucose_level: float
    blood_pressure: float
    cholesterol: float
    bmi: float
    age: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    smoking_status: str