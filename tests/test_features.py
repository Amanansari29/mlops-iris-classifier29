"""Tests for feature engineering."""

import runpy
import sys

import pandas as pd

from src.pipeline.features import engineer_features


def test_sepal_and_petal_features_are_computed_correctly(tmp_path):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.0],
            "sepal width (cm)": [4.0],
            "petal length (cm)": [2.0],
            "petal width (cm)": [1.0],
            "species": ["setosa"],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert result.loc[0, "sepal_area"] == 20.0
    assert result.loc[0, "petal_area"] == 2.0
    assert result.loc[0, "sepal_to_petal_length_ratio"] == 2.5
    assert output_path.exists()


def test_petal_length_bin_assigns_expected_category(tmp_path):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.0, 5.0, 5.0],
            "sepal width (cm)": [3.0, 3.0, 3.0],
            "petal length (cm)": [1.0, 3.0, 6.0],
            "petal width (cm)": [0.2, 1.0, 2.0],
            "species": ["setosa", "versicolor", "virginica"],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert list(result["petal_length_bin"]) == ["short", "medium", "long"]


def test_zero_petal_length_produces_missing_ratio(tmp_path):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.0],
            "sepal width (cm)": [3.0],
            "petal length (cm)": [0.0],
            "petal width (cm)": [0.2],
            "species": ["setosa"],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert pd.isna(result.loc[0, "sepal_to_petal_length_ratio"])


def test_features_command_line_entrypoint(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.0],
            "sepal width (cm)": [4.0],
            "petal length (cm)": [2.0],
            "petal width (cm)": [1.0],
            "species": ["setosa"],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "features.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    runpy.run_path("src/pipeline/features.py", run_name="__main__")

    assert output_path.exists()
