from fastapi import FastAPI
from app.schemas import HealthInput
from app.model.predict import predict

app = FastAPI(title="Health Risk Prediction API")


@app.get("/")
def home():
    return {"message": "ML API is running "}


@app.post("/predict")
def get_prediction(input: HealthInput):
    result = predict(input.dict())
    return result