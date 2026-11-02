"""Tests for preparing the model artifacts used by the Docker image."""

import joblib
import pandas as pd

from src.train_final_model import FEATURE_COLS, train_and_save_model


def test_training_saves_loadable_model_and_label_encoder(tmp_path):
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 6.0, 6.5],
            "sepal width (cm)": [3.5, 2.8, 3.0],
            "petal length (cm)": [1.4, 4.5, 5.5],
            "petal width (cm)": [0.2, 1.3, 2.0],
            "sepal_area": [17.85, 16.8, 19.5],
            "petal_area": [0.28, 5.85, 11.0],
            "sepal_to_petal_length_ratio": [5.1 / 1.4, 6.0 / 4.5, 6.5 / 5.5],
            "species": ["setosa", "versicolor", "virginica"],
        }
    )
    data_path = tmp_path / "iris_features.csv"
    model_dir = tmp_path / "models"
    df.to_csv(data_path, index=False)

    model, encoder = train_and_save_model(data_path, model_dir)

    loaded_model = joblib.load(model_dir / "final_model.joblib")
    loaded_encoder = joblib.load(model_dir / "label_encoder.joblib")
    predictions = loaded_model.predict(df[FEATURE_COLS])

    assert set(encoder.classes_) == {"setosa", "versicolor", "virginica"}
    assert list(loaded_encoder.classes_) == list(encoder.classes_)
    assert set(loaded_encoder.inverse_transform(predictions)).issubset(
        set(encoder.classes_)
    )
    assert model.n_estimators == 100
