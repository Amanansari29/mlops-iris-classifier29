# Testing Strategy

Run the suite from the repository root after installing `requirements.txt`:

```bash
python -m pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

Pytest discovers `test_*.py` under `tests/`. The command prints per-file
coverage and writes `test-results.xml` in JUnit format for CI systems. Coverage
and JUnit outputs are generated locally and are not committed.

## Test layers

| Test file | Layer | Coverage |
|---|---|---|
| `tests/test_preprocess.py` | Data processing | Duplicate removal, median imputation, missing targets, empty input, and the command-line entrypoint |
| `tests/test_features.py` | Data processing | Derived areas, length bins, zero-length ratio handling, and the command-line entrypoint |
| `tests/test_validate.py` | Data validation | Valid rows, missing columns, nulls, invalid labels, out-of-range measurements, and the command-line entrypoint |
| `tests/test_training_pipeline.py` | Pipeline smoke tests | Model fitting, prediction shape, and probability invariants on a small synthetic dataset |
| `tests/test_model_quality.py` | Model quality | Valid class predictions and a minimum held-out accuracy of 0.90 |

The model-quality test uses scikit-learn's bundled Iris dataset and reproduces
the pipeline's area and ratio features. It therefore does not require DVC
artifacts or a generated `data/processed/iris_features.csv` file, making the
suite runnable in a clean checkout.

## Measured coverage

The full suite passed **21 tests**. In the local validation run, total coverage
was **21%** across all of `src/`. Most of the uncovered statements belong to
separate training, tuning, and MLflow scripts that this pipeline-focused suite
does not execute. The covered pipeline modules reached:

| Module | Coverage |
|---|---:|
| `src/pipeline/preprocess.py` | 100% |
| `src/pipeline/features.py` | 100% |
| `src/pipeline/validate.py` | 100% |

Coverage percentages can vary if source files or test scope change; rerun the
command above to get the current report. A failed assertion or a model accuracy
below 0.90 fails the test run, while the JUnit XML provides structured results
for CI reporting.
