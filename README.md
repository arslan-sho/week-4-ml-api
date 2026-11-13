# California Housing Prediction API

A machine learning API that predicts area-level median housing values in California using a trained Random Forest model.

## Features

- FastAPI prediction endpoint
- Input validation with Pydantic
- Saved Scikit-learn model
- Model loading during application startup
- Health-check endpoint
- Automated tests with pytest
- Interactive API documentation with Swagger UI
- Optional Docker containerization

## Tech Stack

- Python 3.12
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Pydantic
- Pytest
- Docker (optional)

## Project Structure

```text
week-4-ml-api/
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── models/
│   └── best_random_forest.joblib
└── tests/
    └── test_api.py
```

## Run Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn main:app --reload
```

The API will run locally at:

http://127.0.0.1:8000

### 4. Open API documentation

Swagger UI:

http://127.0.0.1:8000/docs

### 5. Check API health

Open:

http://127.0.0.1:8000/health

A healthy response confirms that the API is running and the trained model is loaded.

## Run Tests

With the virtual environment activated, run:

```bash
python -m pytest -v
```

## Run with Docker (Optional)

Build the image:

```bash
sudo docker build -t california-housing-api .
```

Start the container:

```bash
sudo docker run -d --name california-housing-api-container -p 8000:8000 california-housing-api
```

Open the documentation:

http://127.0.0.1:8000/docs

Stop the container:

```bash
sudo docker stop california-housing-api-container
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health and model status |
| POST | `/validate` | Validate housing input |
| POST | `/predict` | Predict median housing value |

## Prediction Input Example

```json
{
  "MedInc": 8.3,
  "HouseAge": 25,
  "AveRooms": 5.2,
  "AveBedrms": 1.1,
  "Population": 1000,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

## Important Note

The model predicts area-level median housing values, not the market value of an individual property. Predictions are estimates and should not be treated as formal property appraisals.
