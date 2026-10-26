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
does not execute. Each of the three data-pipeline modules reached 100% coverage,
exceeding the 85% target:

| Module | Coverage |
|---|---:|
| `src/pipeline/preprocess.py` | 100% |
| `src/pipeline/features.py` | 100% |
| `src/pipeline/validate.py` | 100% |

Coverage percentages can vary if source files or test scope change; rerun the
command above to get the current report.

## Verification steps

Run `python -m pytest -v` to list every test and its result. The current suite
contains 21 tests (the earlier 17-test example predates added entrypoint and
null/CLI-failure checks). Relevant current test names include:

```text
tests/test_preprocess.py::test_preprocess_removes_duplicates PASSED
tests/test_preprocess.py::test_preprocess_imputes_missing_numeric_values PASSED
tests/test_features.py::test_sepal_and_petal_features_are_computed_correctly PASSED
tests/test_validate.py::test_validate_fails_on_out_of_range_value PASSED
tests/test_training_pipeline.py::test_model_predict_proba_sums_to_one PASSED
tests/test_model_quality.py::test_model_meets_minimum_accuracy_threshold PASSED
============================== 21 passed ==============================
```

Run the command at the top of this document to generate the coverage table and
`test-results.xml`. The report should contain one `<testsuite>` element and 21
`<testcase>` elements. The three pipeline modules should each be above 85%.

The quality gate was checked with `MIN_ACCEPTABLE_ACCURACY` temporarily raised
to `0.99`: the quality test failed as intended because the measured held-out
accuracy is `0.9667`. Restore the normal threshold of `0.90` after this check.
A regression that brings accuracy below `0.90` also fails the test run.
