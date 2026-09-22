# Feature Store Experiment Results

## Implementation

Experiment 5 is implemented in
[`practical5_feast/iris_feature_repo/feature_repo`](../practical5_feast/iris_feature_repo/feature_repo).
The repository uses Feast's local provider, a SQLite online store, and a
Parquet offline source generated from
`data/processed/iris_features.csv`.

The single `iris_features_source` feeds two registered feature views:

- `iris_measurements`: the four original Iris measurements.
- `iris_engineered_features`: `sepal_area`, `petal_area`,
  `sepal_to_petal_length_ratio`, and `petal_length_bin`.

Both views are included in the `iris_feature_service`, allowing another model
or service to consume the same definitions without reimplementing feature
engineering.

## Execution results

- Source preparation completed with **149 rows** and 12 columns.
- `feast apply` created one entity, two feature views, and one feature service.
- `feast materialize-incremental` completed successfully.
- Both feature views reported `AVAILABLE_ONLINE`.
- Online retrieval for `sample_id=1` returned all eight requested features.
- Historical retrieval returned **149 rows** with **zero null feature values**.
- Feature-service retrieval returned all registered measurement and engineered
  features for `sample_id=1`.

The source contains 149 rows, so the historical result is 149 rows rather than
the 150-row figure in the original worksheet's verification text.

## Benefits demonstrated

### Training-serving consistency

The same Parquet source and feature-view schema populate both the offline
historical retrieval path and the online SQLite store. This keeps training and
serving aligned and avoids independently reimplementing the transformations.

### Point-in-time correctness

Historical retrieval joins each entity row at its `event_timestamp`, ensuring
that only feature values available at that time are used. This prevents future
feature values from leaking into training data.

### Reusability and governance

The feature service exposes a named, versioned set of features that can be
reused by classification, clustering, or other consumers. The feature
definitions in `features.py` remain the single source of truth.
