"""Compare the baseline, grid-search, and random-search MLflow runs."""

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from tuning_common import configure_mlflow


EXPECTED_RUNS = {
    "baseline_decision_tree",
    "grid_search_random_forest",
    "random_search_random_forest",
}


def compare_results() -> pd.DataFrame:
    configure_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name("iris-hyperparameter-tuning")
    if experiment is None:
        raise RuntimeError(
            "Experiment 'iris-hyperparameter-tuning' not found. "
            "Run the baseline and both tuning scripts first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
        max_results=1000,
    )
    latest_by_name = {}
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")
        if run_name in EXPECTED_RUNS:
            latest_by_name[run_name] = run

    missing_runs = EXPECTED_RUNS - latest_by_name.keys()
    if missing_runs:
        raise RuntimeError(
            "Missing expected tuning runs: "
            f"{', '.join(sorted(missing_runs))}. Run the missing scripts first."
        )

    rows = []
    for run_name in (
        "baseline_decision_tree",
        "grid_search_random_forest",
        "random_search_random_forest",
    ):
        run = latest_by_name[run_name]
        row = {
            "run": run_name,
            "model_type": run.data.params.get("model_type"),
            "cv_f1_macro_mean": run.data.metrics.get("cv_f1_macro_mean"),
            "best_cv_f1_macro": run.data.metrics.get("best_cv_f1_macro"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
            "total_fits": run.data.params.get("total_fits"),
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    results["cv_f1_macro"] = results["best_cv_f1_macro"].fillna(
        results["cv_f1_macro_mean"]
    )
    baseline_score = results.loc[
        results["run"] == "baseline_decision_tree", "cv_f1_macro"
    ].iloc[0]
    grid_score = results.loc[
        results["run"] == "grid_search_random_forest", "cv_f1_macro"
    ].iloc[0]
    random_score = results.loc[
        results["run"] == "random_search_random_forest", "cv_f1_macro"
    ].iloc[0]

    print("HYPERPARAMETER TUNING COMPARISON")
    print("=" * 88)
    print(
        results[
            [
                "run",
                "model_type",
                "cv_f1_macro",
                "test_accuracy",
                "total_fits",
            ]
        ].to_string(index=False)
    )
    print("\nCV F1 improvement over baseline:")
    print(f"Grid Search:   {grid_score - baseline_score:+.4f}")
    print(f"Random Search: {random_score - baseline_score:+.4f}")
    print(f"Random vs Grid: {random_score - grid_score:+.4f}")
    return results


if __name__ == "__main__":
    compare_results()
