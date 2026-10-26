"""Tests for preprocessing behavior and edge cases."""

import runpy
import sys

import pandas as pd
import pytest

from src.pipeline.preprocess import preprocess

NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def valid_raw_df(tmp_path):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
            "petal length (cm)": [1.4, 1.4, 1.3],
            "petal width (cm)": [0.2, 0.2, 0.2],
            "species": ["setosa", "setosa", "setosa"],
            "collected_at": ["2026-01-01"] * 3,
        }
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_preprocess_removes_duplicates(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)
    df_with_duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    input_path = tmp_path / "raw_with_duplicate.csv"
    output_path = tmp_path / "processed.csv"
    df_with_duplicate.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 3
    assert output_path.exists()


def test_preprocess_imputes_missing_numeric_values(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "sepal length (cm)"] = None
    input_path = tmp_path / "raw_with_missing.csv"
    output_path = tmp_path / "processed.csv"
    df.to_csv(input_path, index=False)
    expected_median = df["sepal length (cm)"].median()

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isna().sum() == 0
    assert result.loc[0, "sepal length (cm)"] == pytest.approx(expected_median)


def test_preprocess_drops_rows_with_missing_target(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)
    df.loc[0, "species"] = None
    input_path = tmp_path / "raw_missing_target.csv"
    output_path = tmp_path / "processed.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2
    assert result["species"].notna().all()


def test_preprocess_handles_empty_dataframe(tmp_path):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(columns=NUMERIC_COLS + ["species", "collected_at"]).to_csv(
        input_path, index=False
    )

    result = preprocess(str(input_path), str(output_path))

    assert result.empty
    assert output_path.exists()


def test_preprocess_command_line_entrypoint(tmp_path, valid_raw_df, monkeypatch):
    output_path = tmp_path / "processed.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess.py",
            "--input",
            str(valid_raw_df),
            "--output",
            str(output_path),
        ],
    )

    runpy.run_path("src/pipeline/preprocess.py", run_name="__main__")

    assert output_path.exists()
