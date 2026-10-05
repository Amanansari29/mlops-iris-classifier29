"""Tune a Random Forest by sampling the same space as the grid search."""

import json
from pathlib import Path

import mlflow
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from tuning_common import DEFAULT_DATA_PATH, configure_mlflow, load_dataset


PARAM_DIST = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 15),
    "max_features": ["sqrt", "log2"],
}
N_ITER = 30
CV_FOLDS = 5
ARTIFACT_NAME = "random_search_all_candidates.csv"


def run_random_search(data_path: str = str(DEFAULT_DATA_PATH)):
    configure_mlflow()
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
    )

    with mlflow.start_run(run_name="random_search_random_forest"):
        mlflow.log_params(
            {
                "search_type": "RandomizedSearchCV",
                "model_type": "RandomForestClassifier",
                "n_iter": N_ITER,
                "cv_folds": CV_FOLDS,
                "total_fits": N_ITER * CV_FOLDS,
                "random_state": 42,
            }
        )
        search.fit(X_train, y_train)
        test_accuracy = search.best_estimator_.score(X_test, y_test)
        mlflow.log_metric("best_cv_f1_macro", float(search.best_score_))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in search.best_params_.items()
            }
        )

        results_df = pd.DataFrame(search.cv_results_)[
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

    candidate_count = len(search.cv_results_["params"])
    print(
        f"Random Search evaluated {candidate_count} combinations x "
        f"{CV_FOLDS} folds = {candidate_count * CV_FOLDS} total fits"
    )
    print(f"Best params: {search.best_params_}")
    print(f"Best CV f1_macro: {search.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return float(search.best_score_), search.best_params_


if __name__ == "__main__":
    run_random_search()
