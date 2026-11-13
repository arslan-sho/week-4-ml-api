
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "models" / "best_random_forest.joblib"


# Load the model when the application starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="California Housing Prediction API",
    description="Predict California area-level median housing values.",
    version="1.0.0",
    lifespan=lifespan
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
def health_check(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/validate")
def validate_housing(data: HousingInput):
    return {
        "message": "Input data is valid!",
        "received_data": data.model_dump()
    }


@app.post("/predict")
def predict_housing(data: HousingInput, request: Request):
    model = request.app.state.model
    input_data = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(input_data)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc

    # The model predicts values in units of $100,000
    predicted_price = float(prediction * 100000)

    return {
        "predicted_price_usd": round(predicted_price, 2),
        "currency": "USD"
    }
