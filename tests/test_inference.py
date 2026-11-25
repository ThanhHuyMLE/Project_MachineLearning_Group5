from __future__ import annotations

import json

import numpy as np
import pandas as pd

from house_price.artifacts import MANIFEST_FILENAME, save_inference_artifacts
from house_price.models import MLPTorchRegressorModel
from house_price.predict import main, predict_frame
from house_price.preprocessing import (
    build_preprocessor,
    output_feature_names,
    prepare_feature_frame,
)


def _raw_houses(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "date": ["20150101T000000"] * n,
            "bedrooms": rng.integers(1, 6, n).astype(float),
            "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5], n),
            "sqft_living": rng.integers(600, 3500, n),
            "sqft_lot": rng.integers(1200, 15000, n),
            "floors": rng.choice([1.0, 1.5, 2.0], n),
            "waterfront": rng.integers(0, 2, n),
            "view": rng.integers(0, 5, n),
            "condition": rng.integers(1, 6, n),
            "grade": rng.integers(5, 11, n),
            "sqft_above": rng.integers(500, 2800, n),
            "sqft_basement": rng.integers(0, 800, n),
            "yr_built": rng.integers(1950, 2015, n),
            "yr_renovated": np.zeros(n, dtype=int),
            "zipcode": rng.choice([98001, 98002, 98101], n),
            "lat": rng.uniform(47.1, 47.8, n),
            "long": rng.uniform(-122.5, -121.3, n),
            "sqft_living15": rng.integers(600, 3500, n),
            "sqft_lot15": rng.integers(1200, 15000, n),
        }
    )
    frame["price"] = 100_000 + 180 * frame["sqft_living"] + 20_000 * frame["grade"]
    return frame


def _trained_artifacts(tmp_path):
    raw = _raw_houses()
    features = prepare_feature_frame(raw)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.transform(features)

    model = MLPTorchRegressorModel(
        hidden_layer_sizes=(4,),
        epochs=2,
        batch_size=8,
        save_path=tmp_path / "mlp_torch_parameters.pt",
    )
    model.build_model(input_size=transformed.shape[1])
    model.fit(transformed, raw["price"].to_numpy())

    data = {
        "preprocessor": preprocessor,
        "input_columns": list(features.columns),
        "feature_columns": output_feature_names(preprocessor),
    }
    save_inference_artifacts(data, tmp_path)
    return raw


def test_saved_artifacts_predict_unlabeled_rows(tmp_path):
    raw = _trained_artifacts(tmp_path)
    new_houses = raw.iloc[:3].drop(columns=["price"])

    result = predict_frame(new_houses, tmp_path)

    assert list(result.columns[:-1]) == list(new_houses.columns)
    assert result["predicted_price"].shape == (3,)
    assert np.isfinite(result["predicted_price"]).all()
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["model_type"] == "MLPTorchRegressorModel"


def test_predict_cli_writes_csv(tmp_path):
    raw = _trained_artifacts(tmp_path)
    input_path = tmp_path / "new_houses.csv"
    output_path = tmp_path / "predictions.csv"
    raw.iloc[:2].drop(columns=["price"]).to_csv(input_path, index=False)

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--artifact-dir",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    saved = pd.read_csv(output_path)
    assert len(saved) == 2
    assert "predicted_price" in saved.columns
