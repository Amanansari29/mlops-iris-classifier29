"""Load the best registered model from the MLflow Model Registry."""

import os

import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
MODEL_URI = "models:/iris-classifier-prod/Staging"


def main():
    print("=" * 60)
    print("LOADING REGISTERED MODEL")
    print("=" * 60)
    print("Model URI:", MODEL_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)
    print("\nModel loaded successfully!")
    print("Model type:", type(model))
    print("\nLoaded Model:")
    print(model)


if __name__ == "__main__":
    main()
