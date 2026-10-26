"""Smoke tests for model training and prediction output shape."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(5, 1, size=(30, 4)), columns=FEATURE_COLS
    )
    labels = rng.integers(0, 3, size=30)
    return features, labels


def test_model_trains_without_error(synthetic_dataset):
    features, labels = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    model.fit(features, labels)

    assert callable(model.predict)


def test_model_predict_output_shape(synthetic_dataset):
    features, labels = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(features, labels)

    predictions = model.predict(features)

    assert predictions.shape == (len(features),)


def test_model_predict_proba_sums_to_one(synthetic_dataset):
    features, labels = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(features, labels)

    probabilities = model.predict_proba(features)

    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
