"""Quality gate for model predictions on the bundled Iris dataset."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MIN_ACCEPTABLE_ACCURACY = 0.90
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


@pytest.fixture(scope="module")
def trained_model_and_test_set():
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = iris.target_names[df["species"].to_numpy()]
    df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"] / df["petal length (cm)"].replace(0, pd.NA)
    )

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["species"])
    features = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    model.fit(x_train, y_train)
    return model, x_test, y_test, label_encoder


def test_predictions_are_within_known_label_set(trained_model_and_test_set):
    model, x_test, _, label_encoder = trained_model_and_test_set

    predictions = model.predict(x_test)

    assert set(predictions).issubset(set(range(len(label_encoder.classes_))))


def test_model_meets_minimum_accuracy_threshold(trained_model_and_test_set):
    model, x_test, y_test, _ = trained_model_and_test_set

    accuracy = model.score(x_test, y_test)

    assert accuracy >= MIN_ACCEPTABLE_ACCURACY, (
        f"Model accuracy {accuracy:.4f} fell below the minimum acceptable "
        f"threshold of {MIN_ACCEPTABLE_ACCURACY}"
    )
