"""Train and save the Iris classifier and label encoder for container serving."""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("data/processed/iris_features.csv")
MODEL_DIR = Path("models")
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

logger = logging.getLogger(__name__)


def train_and_save_model(
    data_path: Path = DATA_PATH,
    model_dir: Path = MODEL_DIR,
) -> tuple[RandomForestClassifier, LabelEncoder]:
    """Train the classifier on the processed Iris dataset and save its artifacts."""
    df = pd.read_csv(data_path)
    features = df[FEATURE_COLS].copy()
    features = features.fillna(features.median())

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["species"])
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(features, labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "final_model.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
    logger.info(
        "Trained on %d rows; classes: %s; saved artifacts under %s",
        len(df),
        list(label_encoder.classes_),
        model_dir,
    )
    return model, label_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_and_save_model(args.data, args.model_dir)


if __name__ == "__main__":
    main()
