# Hyperparameter Tuning Analysis

This practical compares a default Decision Tree baseline with exhaustive Grid
Search and budgeted Random Search for a Random Forest. All approaches use the
same stratified 80/20 train/test split (`random_state=42`) and macro F1 with
five-fold cross-validation on the training data. The held-out test set is used
only for the final accuracy measurement.

## Search configurations

The Grid Search evaluates:

- `n_estimators`: 50, 100, 200
- `max_depth`: 3, 5, 10, unlimited
- `min_samples_split`: 2, 5, 10
- `max_features`: `sqrt`, `log2`

That is 3 × 4 × 3 × 2 = 72 configurations, or 360 five-fold model fits.
Random Search samples 30 configurations from the same search space, for 150
fits. The seed is fixed at 42 for reproducibility.

Each script logs a single top-level MLflow run in
`iris-hyperparameter-tuning`. Search runs include the complete ranked
candidate table as a CSV artifact, including parameters, mean and standard
deviation of the cross-validation score, and rank. This keeps the experiment
easy to compare as three runs while retaining each evaluated candidate's
results.

## Measured results

Run the three scripts and then `python src/compare_tuning_results.py` to obtain
the recorded values from the configured MLflow tracking server. Scores depend
on the dataset and installed scikit-learn version; the comparison script is
the source of truth for the current run. The original practical reported:

| Approach | CV macro F1 | Test accuracy | Fits |
|---|---:|---:|---:|
| Default Decision Tree baseline | 0.9663 | 0.9000 | 5 |
| Random Forest Grid Search | 0.9663 | 0.9667 | 360 |
| Random Forest Random Search | 0.9663 | 0.9667 | 150 |

On the current 149-row feature dataset and installed scikit-learn version,
the best cross-validation score ties the Decision Tree baseline rather than
exceeding it. Both tuned Random Forests improve held-out test accuracy from
0.9000 to 0.9667. Random Search ties Grid Search on mean CV macro F1 while
requiring less than half as many fits. This does not guarantee that tuning
will improve cross-validation performance; the comparison script reports the
measured results. Re-run the experiment when data, dependencies, or search
settings change.

## Run commands

From the repository root, start the configured MLflow UI/server in one
terminal, then in a second terminal activate the project's environment, set
`MLFLOW_TRACKING_URI` to the server URL, and run:

```bash
python src/baseline_model.py
python src/grid_search_tuning.py
python src/random_search_tuning.py
python src/compare_tuning_results.py
```

The search candidate CSVs are written to the repository root as well as logged
as MLflow artifacts.

## Viva-voce questions

### 1. Why establish a baseline before tuning?

A baseline provides a reference for whether added model complexity and
optimization actually improve the result. If a tuned Random Forest scores
worse than the simpler Decision Tree, investigate the features, evaluation
setup, search space, and possible overfitting rather than assuming complexity
is beneficial.

### 2. Why can Random Search perform similarly with fewer fits?

Usually only some hyperparameters strongly affect the score. Grid Search
spends evaluations on every value in every dimension, while Random Search
samples combinations and can explore the influential dimensions efficiently.
Grid Search is useful when the space is small, exhaustive reproducibility is
required, or interactions warrant evaluating every combination.

### 3. Why not use the test set to pick hyperparameters?

Selecting configurations based on the test set leaks information from that
set into model selection. The best score then becomes optimistically biased
because the search may select a configuration that fits test-set noise.
Cross-validation on the training set is used for selection; the held-out test
set is reserved for final evaluation.
