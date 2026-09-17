## Pipeline Stages
| Stage | Purpose | Input | Output |
|---|---|---|---|
| Collect | Obtain raw Iris data | sklearn Iris dataset | `iris_raw.csv` |
| Preprocess | Clean data | `iris_raw.csv` | `iris_preprocessed.csv` |
| Feature Engineering | Create useful features | `iris_preprocessed.csv` | `iris_features.csv` |
| Validate | Check schema, nulls and ranges | `iris_features.csv` | Validation result |
