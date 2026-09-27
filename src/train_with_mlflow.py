"""Train multiple Iris models and log each run to MLflow."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = Path("data/processed/iris_features.csv")
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
EXPERIMENT_NAME = "iris-classification-baseline"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])
    return X, y


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def log_confusion_matrix(model_name: str, y_true, y_pred, labels) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    output_path = Path("artifacts") / f"confusion_matrix_{model_name}.png"
    output_path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test, labels):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    params = {"model_type": model_name}
    if hasattr(model, "n_estimators"):
        params["n_estimators"] = model.n_estimators
    if hasattr(model, "max_depth"):
        params["max_depth"] = model.max_depth
    if hasattr(model, "max_iter"):
        params["max_iter"] = model.max_iter
    if hasattr(model, "C"):
        params["C"] = model.C

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_artifact_path = log_confusion_matrix(model_name, y_test, predictions, labels)
        mlflow.log_artifact(model_artifact_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"\nRun: {run.info.run_id}")
        print(f"Model: {model_name}")
        print(metrics)

    return metrics


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    label_values = sorted(set(y))
    model_configs = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=200, random_state=42, solver="lbfgs"),
        ),
        (
            "random_forest_shallow",
            RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        ),
        (
            "random_forest_deep",
            RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
        ),
    ]

    results = []
    for model_name, model in model_configs:
        metrics = train_and_log_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels=label_values,
        )
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results)
    print("\nMODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    best_model = results_df.loc[results_df["f1_macro"].idxmax()]
    print("\nBest model based on F1 score:")
    print(best_model.to_string())


if __name__ == "__main__":
    main()
