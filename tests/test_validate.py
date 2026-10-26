"""Tests for data schema and statistical validation."""

import runpy
import sys

import pandas as pd
import pytest

from src.pipeline.validate import DataValidationError, validate

VALID_ROW = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "species": "setosa",
    "sepal_area": 17.85,
    "petal_area": 0.28,
    "sepal_to_petal_length_ratio": 3.64,
    "petal_length_bin": "short",
}


def _write_data(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_passes_on_clean_data(tmp_path):
    path = _write_data(tmp_path, pd.DataFrame([VALID_ROW, VALID_ROW]))

    result = validate(path)

    assert len(result) == 2


def test_validate_fails_on_missing_column(tmp_path):
    df = pd.DataFrame([VALID_ROW]).drop(columns=["sepal_area"])
    path = _write_data(tmp_path, df)

    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(path)


def test_validate_fails_on_invalid_species(tmp_path):
    row = VALID_ROW.copy()
    row["species"] = "unicorn"
    path = _write_data(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="Unexpected species values"):
        validate(path)


def test_validate_fails_on_out_of_range_value(tmp_path):
    row = VALID_ROW.copy()
    row["sepal length (cm)"] = 50.0
    path = _write_data(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="out of expected range"):
        validate(path)


def test_validate_fails_on_null_values(tmp_path):
    row = VALID_ROW.copy()
    row["petal width (cm)"] = None
    path = _write_data(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="Unexpected null values"):
        validate(path)


def test_validate_command_line_entrypoint(tmp_path, monkeypatch):
    path = _write_data(tmp_path, pd.DataFrame([VALID_ROW]))
    monkeypatch.setattr(sys, "argv", ["validate.py", "--input", path])

    runpy.run_path("src/pipeline/validate.py", run_name="__main__")


def test_validate_command_line_exits_on_invalid_data(tmp_path, monkeypatch):
    row = VALID_ROW.copy()
    row["species"] = "unicorn"
    path = _write_data(tmp_path, pd.DataFrame([row]))
    monkeypatch.setattr(sys, "argv", ["validate.py", "--input", path])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path("src/pipeline/validate.py", run_name="__main__")

    assert exc_info.value.code == 1
