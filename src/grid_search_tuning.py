"""Exhaustively tune a Random Forest over a fixed hyperparameter grid."""

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid

from tuning_common import DEFAULT_DATA_PATH, configure_mlflow, load_dataset


PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
CV_FOLDS = 5
ARTIFACT_NAME = "grid_search_all_candidates.csv"


def run_grid_search(data_path: str = str(DEFAULT_DATA_PATH)):
    configure_mlflow()
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    total_combinations = len(ParameterGrid(PARAM_GRID))
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=False,
    )

    with mlflow.start_run(run_name="grid_search_random_forest"):
        mlflow.log_params(
            {
                "search_type": "GridSearchCV",
                "model_type": "RandomForestClassifier",
                "total_combinations": total_combinations,
                "cv_folds": CV_FOLDS,
                "total_fits": total_combinations * CV_FOLDS,
                "random_state": 42,
            }
        )
        grid.fit(X_train, y_train)
        test_accuracy = grid.best_estimator_.score(X_test, y_test)
        mlflow.log_metric("best_cv_f1_macro", float(grid.best_score_))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in grid.best_params_.items()
            }
        )

        results_df = pd.DataFrame(grid.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].copy()
        results_df["params"] = results_df["params"].map(
            lambda params: json.dumps(params, sort_keys=True)
        )
        results_df = results_df.sort_values(
            ["rank_test_score", "params"], kind="stable"
        )
        output_path = Path.cwd() / ARTIFACT_NAME
        results_df.to_csv(output_path, index=False)
        mlflow.log_artifact(str(output_path))

    candidate_count = len(grid.cv_results_["params"])
    total_fits = candidate_count * CV_FOLDS
    print(
        f"Grid Search evaluated {candidate_count} combinations x "
        f"{CV_FOLDS} folds = {total_fits} total fits"
    )
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return float(grid.best_score_), grid.best_params_


if __name__ == "__main__":
    run_grid_search()
