"""Retrieve point-in-time-correct features for all Iris source rows."""

import pandas as pd
from feast import FeatureStore


FEATURES = [
    "iris_measurements:sepal length (cm)",
    "iris_measurements:sepal width (cm)",
    "iris_measurements:petal length (cm)",
    "iris_measurements:petal width (cm)",
    "iris_engineered_features:sepal_area",
    "iris_engineered_features:petal_area",
    "iris_engineered_features:sepal_to_petal_length_ratio",
    "iris_engineered_features:petal_length_bin",
]


def main() -> None:
    source_df = pd.read_parquet("data/iris_features.parquet")
    entity_df = source_df[["sample_id", "event_timestamp"]]
    store = FeatureStore(repo_path=".")
    training_df = store.get_historical_features(
        entity_df=entity_df, features=FEATURES
    ).to_df()
    feature_columns = [column for column in training_df.columns if column not in {
        "sample_id", "event_timestamp"
    }]
    print(f"Retrieved {len(training_df)} rows")
    print(f"Feature columns: {feature_columns}")
    print(f"Null feature values: {int(training_df[feature_columns].isna().sum().sum())}")


if __name__ == "__main__":
    main()
