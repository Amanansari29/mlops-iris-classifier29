"""Tests for the container prediction HTTP API."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.serve import FEATURE_ORDER, MAX_REQUEST_BYTES, PredictionHandler


@pytest.fixture
def api_server(monkeypatch):
    iris = load_iris(as_frame=True)
    features = iris.data.copy()
    features.columns = FEATURE_ORDER[:4]
    features["sepal_area"] = (
        features["sepal length (cm)"] * features["sepal width (cm)"]
    )
    features["petal_area"] = (
        features["petal length (cm)"] * features["petal width (cm)"]
    )
    features["sepal_to_petal_length_ratio"] = (
        features["sepal length (cm)"] / features["petal length (cm)"].replace(0, np.nan)
    )
    features = features.fillna(features.median())
    encoder = LabelEncoder()
    labels = encoder.fit_transform(iris.target_names[iris.target.to_numpy()])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(features, labels)
    monkeypatch.setattr(PredictionHandler, "model", model)
    monkeypatch.setattr(PredictionHandler, "label_encoder", encoder)

    server = ThreadingHTTPServer(("127.0.0.1", 0), PredictionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _sample_payload():
    return {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
        "sepal_area": 17.85,
        "petal_area": 0.28,
        "sepal_to_petal_length_ratio": 5.1 / 1.4,
    }


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    with response:
        return response.status, json.loads(response.read())


def test_health_endpoint_returns_healthy(api_server):
    with urllib.request.urlopen(f"{api_server}/health") as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "healthy"}


def test_unknown_get_path_returns_not_found(api_server):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{api_server}/unknown")

    assert error.value.code == 404


def test_prediction_returns_known_class_and_probabilities(api_server):
    status, body = _post_json(f"{api_server}/predict", _sample_payload())

    assert status == 200
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}
    assert len(body["probabilities"]) == 3
    assert all(0 <= probability <= 1 for probability in body["probabilities"])
    assert sum(body["probabilities"]) == pytest.approx(1.0)


def test_prediction_rejects_missing_features(api_server):
    status, body = _post_json(
        f"{api_server}/predict",
        {"sepal length (cm)": 5.1},
    )

    assert status == 400
    assert body["error"] == "Missing features"
    assert "petal width (cm)" in body["missing"]


def test_prediction_rejects_non_numeric_features(api_server):
    payload = _sample_payload()
    payload["petal width (cm)"] = "0.2"

    status, body = _post_json(f"{api_server}/predict", payload)

    assert status == 400
    assert body["error"] == "Feature values must be finite numbers"


def test_prediction_rejects_non_finite_features(api_server):
    payload = _sample_payload()
    payload["petal width (cm)"] = float("inf")

    status, body = _post_json(f"{api_server}/predict", payload)

    assert status == 400
    assert body["error"] == "Feature values must be finite numbers"


def test_prediction_rejects_unrepresentably_large_feature(api_server):
    payload = _sample_payload()
    payload["petal width (cm)"] = 10**400

    status, body = _post_json(f"{api_server}/predict", payload)

    assert status == 400
    assert body["error"] == "Feature values must be finite numbers"


def test_prediction_rejects_non_object_json(api_server):
    request = urllib.request.Request(
        f"{api_server}/predict",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)

    assert error.value.code == 400
    assert json.loads(error.value.read())["error"] == "JSON body must be an object"


def test_prediction_rejects_oversized_body(api_server):
    request = urllib.request.Request(
        f"{api_server}/predict",
        data=b" " * (MAX_REQUEST_BYTES + 1),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)

    assert error.value.code == 413
