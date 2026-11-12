from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Load the trained model once when the API starts
MODEL_PATH = Path(__file__).parent / "models" / "best_random_forest.joblib"
model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="California Housing Prediction API",
    description="Predict California area-level median housing values.",
    version="1.0.0"
)


class HousingInput(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(ge=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)


@app.get("/")
def home():
    return {
        "message": "Welcome to my ML API!",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/validate")
def validate_housing(data: HousingInput):
    return {
        "message": "Input data is valid!",
        "received_data": data.model_dump()
    }


@app.post("/predict")
def predict_housing(data: HousingInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    # The model predicts values in units of $100,000
    predicted_price = float(prediction * 100000)

    return {
        "predicted_price_usd": round(predicted_price, 2),
        "currency": "USD"
    }