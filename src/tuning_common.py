"""Shared dataset and MLflow configuration for hyperparameter tuning."""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
TARGET_COL = "species"
EXPERIMENT_NAME = "iris-hyperparameter-tuning"
DEFAULT_DATA_PATH = Path("data/processed/iris_features.csv")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH):
    """Load features and split them into reproducible stratified train/test sets."""
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median())
    y = LabelEncoder().fit_transform(df[TARGET_COL])
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def configure_mlflow() -> None:
    """Use the configured tracking server, defaulting to the practical's local UI."""
    import mlflow

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    mlflow.set_experiment(EXPERIMENT_NAME)
