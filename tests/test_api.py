from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_predict_with_valid_input():
    sample_data = {
        "MedInc": 8.3,
        "HouseAge": 25,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert response.json()["currency"] == "USD"
    assert response.json()["predicted_price_usd"] > 0


def test_predict_with_invalid_input():
    sample_data = {
        "MedInc": -5,
        "HouseAge": 25,
        "AveRooms": 5.2,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 422