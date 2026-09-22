"""Demonstrate reuse through the registered Feast feature service."""

from feast import FeatureStore


def main() -> None:
    store = FeatureStore(repo_path=".")
    feature_service = store.get_feature_service("iris_feature_service")
    result = store.get_online_features(
        features=feature_service, entity_rows=[{"sample_id": 1}]
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
