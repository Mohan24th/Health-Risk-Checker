from pydantic import BaseModel

class HealthInput(BaseModel):
    age: int
    bmi: float
    glucose_level: float
    blood_pressure: float
    smoking_status: str