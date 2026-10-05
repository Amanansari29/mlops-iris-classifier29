"""Establish a cross-validated Decision Tree baseline for tuning."""

import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from tuning_common import DEFAULT_DATA_PATH, configure_mlflow, load_dataset


def run_baseline(data_path: str = str(DEFAULT_DATA_PATH)) -> float:
    configure_mlflow()
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    model = DecisionTreeClassifier(random_state=42)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
    )

    with mlflow.start_run(run_name="baseline_decision_tree"):
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier_default",
                "cv_folds": 5,
                "total_fits": 5,
                "random_state": 42,
            }
        )
        mlflow.log_metrics(
            {
                "cv_f1_macro_mean": float(cv_scores.mean()),
                "cv_f1_macro_std": float(cv_scores.std()),
                "test_accuracy": float(model.fit(X_train, y_train).score(X_test, y_test)),
            }
        )

    print(
        f"Baseline CV f1_macro: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )
    print(f"Baseline test accuracy: {model.score(X_test, y_test):.4f}")
    return float(cv_scores.mean())


if __name__ == "__main__":
    run_baseline()
