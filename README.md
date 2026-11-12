# California Housing Prediction API

A machine learning API that predicts area-level median housing values in California using a trained Random Forest model.

## Features

- FastAPI prediction endpoint
- Input validation with Pydantic
- Saved Scikit-learn model
- Automated tests with pytest
- Docker containerization
- Interactive API documentation with Swagger UI

## Tech Stack

- Python 3.12
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Pytest
- Docker

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

Activate your virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Open the documentation:

http://127.0.0.1:8000/docs

## Run Tests

```bash
python -m pytest -v
```

## Run with Docker

Build the image:

```bash
sudo docker build -t california-housing-api .
```

Start the container:

```bash
sudo docker run -d --name california-housing-api-container -p 8000:8000 california-housing-api
```

Open:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
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
