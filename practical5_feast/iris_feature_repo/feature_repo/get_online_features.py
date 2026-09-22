"""Retrieve the registered Iris features from the online store."""

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
    store = FeatureStore(repo_path=".")
    result = store.get_online_features(
        features=FEATURES, entity_rows=[{"sample_id": 1}]
    )
    result_df = pd.DataFrame(result.to_dict())
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
