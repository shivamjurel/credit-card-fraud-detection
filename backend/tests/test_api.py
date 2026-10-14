"""
API tests. Run from the backend/ folder:
    pytest -v

The expected predictions are NOT hardcoded: the inputs are real transactions
from each dataset's held-out test set (models/spark/samples.json) and the REAL
trained PySpark models produce the predictions.
(The first test waits ~15 s while Spark starts.)
"""

import json

import config

SAMPLES = json.loads(config.SAMPLES_PATH.read_text())
CONTEXT_FIELDS = {"actual_class", "merchant", "city", "state"}


def to_request_body(sample: dict) -> dict:
    """Remove the label and display-only fields - the API only receives model inputs."""
    return {key: value for key, value in sample.items() if key not in CONTEXT_FIELDS}


def test_health_reports_models_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models"] == {"ulb": True, "sparkov": True}


def test_empty_history(client):
    """Test 5: a fresh database has no transactions."""
    assert client.get("/transactions").json() == []
    stats = client.get("/dashboard").json()
    assert stats["total_transactions"] == 0
    assert stats["fraud_percentage"] == 0.0


def test_normal_transaction_is_legitimate(client):
    """Test 1: real legitimate transactions from both test sets."""
    for dataset in ("sparkov", "ulb"):
        response = client.post(f"/predict/{dataset}", json=to_request_body(SAMPLES[dataset]["legitimate"][0]))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"] == "Legitimate"
        assert body["risk_level"] in {"Low", "Medium"}
        assert body["fraud_probability"] < body["risk_thresholds"]["high"]


def test_suspicious_transaction_is_fraud(client):
    """Test 2: real fraudulent transactions (detected in evaluation) from both test sets."""
    for dataset in ("sparkov", "ulb"):
        response = client.post(f"/predict/{dataset}", json=to_request_body(SAMPLES[dataset]["fraud"][0]))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"] == "Fraud"
        assert body["risk_level"] == "High"


def test_negative_amount_is_rejected(client):
    """Test 3: invalid amount -> HTTP 422 validation error."""
    body = to_request_body(SAMPLES["sparkov"]["legitimate"][0])
    body["amount"] = -50
    response = client.post("/predict/sparkov", json=body)
    assert response.status_code == 422
    assert any(error["field"] == "amount" for error in response.json()["errors"])


def test_invalid_category_and_missing_field_are_rejected(client):
    body = to_request_body(SAMPLES["sparkov"]["legitimate"][0])
    body["category"] = "casino"
    del body["age"]
    response = client.post("/predict/sparkov", json=body)
    assert response.status_code == 422
    fields = {error["field"] for error in response.json()["errors"]}
    assert {"category", "age"} <= fields


def test_non_numeric_value_is_rejected(client):
    body = to_request_body(SAMPLES["ulb"]["legitimate"][0])
    body["v3"] = "abc"
    assert client.post("/predict/ulb", json=body).status_code == 422


def test_history_detail_dashboard_and_delete(client):
    transactions = client.get("/transactions").json()
    assert len(transactions) == 4  # the four successful predictions above
    assert {t["dataset"] for t in transactions} == {"ulb", "sparkov"}

    sparkov_only = client.get("/transactions", params={"dataset": "sparkov"}).json()
    assert len(sparkov_only) == 2

    first_id = transactions[0]["id"]
    detail = client.get(f"/transactions/{first_id}").json()
    assert detail["features"] and detail["risk_thresholds"]

    stats = client.get("/dashboard").json()
    assert stats["total_transactions"] == 4
    assert stats["fraud_transactions"] == 2
    assert stats["by_dataset"] == {"sparkov": 2, "ulb": 2}
    assert client.get("/dashboard", params={"dataset": "ulb"}).json()["total_transactions"] == 2

    assert client.delete(f"/transactions/{first_id}").status_code == 200
    assert client.get(f"/transactions/{first_id}").status_code == 404
    assert client.delete(f"/transactions/{first_id}").status_code == 404


def test_model_info_lists_five_models_per_dataset(client):
    info = client.get("/model-info").json()
    for dataset in ("ulb", "sparkov"):
        assert len(info["datasets"][dataset]["models"]) == 5
        assert info["datasets"][dataset]["selected_model"]
