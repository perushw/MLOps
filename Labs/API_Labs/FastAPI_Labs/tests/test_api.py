import pytest
from fastapi.testclient import TestClient
from main import app

ADELIE = {"island": "Torgersen", "sex": "male", "bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}
CHINSTRAP = {"island": "Dream", "sex": "female", "bill_length_mm": 46.5, "bill_depth_mm": 17.9, "flipper_length_mm": 192, "body_mass_g": 3500}
GENTOO = {"island": "Biscoe", "sex": "female", "bill_length_mm": 46.1, "bill_depth_mm": 13.2, "flipper_length_mm": 211, "body_mass_g": 4500}


# the with block runs the app's lifespan, so the model is loaded before the tests run
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    info = response.json()
    assert info["algorithm"] == "RandomForestClassifier"
    assert info["classes"] == ["Adelie", "Chinstrap", "Gentoo"]
    assert info["test_accuracy"] > 0.9


def test_predict_one(client):
    response = client.post("/predict", json=ADELIE)
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "Adelie"
    assert body["confidence"] == max(body["probabilities"].values())
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_predict_batch(client):
    response = client.post("/predict/batch", json=[ADELIE, CHINSTRAP, GENTOO])
    assert response.status_code == 200
    assert [r["species"] for r in response.json()] == ["Adelie", "Chinstrap", "Gentoo"]


def test_unknown_island_is_rejected(client):
    response = client.post("/predict", json={**ADELIE, "island": "Hawaii"})
    assert response.status_code == 422


def test_negative_measurement_is_rejected(client):
    response = client.post("/predict", json={**ADELIE, "body_mass_g": -5})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    penguin = dict(ADELIE)
    del penguin["sex"]
    response = client.post("/predict", json=penguin)
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422
