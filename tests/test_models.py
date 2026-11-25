"""Kiểm tra hợp đồng và hoạt động của các lớp mô hình."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import numpy as np
import pytest

from house_price.models import (
    DecisionTreeRegressorModel,
    LinearRegressorModel,
    MLPTorchRegressorModel,
    RandomForestRegressorModel,
)
from house_price.train_evaluate import MODEL_NAMES, train_and_evaluate


def test_predict_before_fit_raises():
    model = LinearRegressorModel()
    model.build_model()

    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(np.zeros((2, 18)))


def test_fit_before_build_raises():
    model = LinearRegressorModel()

    with pytest.raises(RuntimeError, match="Chưa build_model"):
        model.fit(np.zeros((2, 18)), np.zeros(2))


def test_linear_regression_fits_and_predicts_on_synthetic_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 18))
    y = X[:, 0] * 3.0 + 5.0

    model = LinearRegressorModel()
    model.build_model()
    model.fit(X, y)

    predictions = model.predict(X[:5])

    assert predictions.shape == (5,)
    assert np.allclose(predictions, y[:5], atol=1e-6)


def _build_contract_model(model_class, tmp_path):
    """Tạo mô hình với cấu hình nhỏ, đủ để chạy nhanh trên dữ liệu synthetic."""
    if model_class is MLPTorchRegressorModel:
        return MLPTorchRegressorModel(
            epochs=5,
            batch_size=16,
            save_path=tmp_path / "mlp_torch_parameters.pt",
        )
    return model_class()


@pytest.mark.parametrize(
    "model_class",
    [
        MLPTorchRegressorModel,
        LinearRegressorModel,
        DecisionTreeRegressorModel,
        RandomForestRegressorModel,
    ],
)
def test_all_official_model_classes_build_fit_and_predict_on_synthetic_data(model_class, tmp_path):
    """Hợp đồng chung: cả 4 lớp mô hình đều build/fit/predict được."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10

    model = _build_contract_model(model_class, tmp_path)
    if model_class is MLPTorchRegressorModel:
        model.build_model(input_size=X.shape[1])
    else:
        model.build_model()
    model.fit(X[:50], y[:50])

    predictions = model.predict(X[50:])

    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


def test_train_and_evaluate_returns_four_official_models_on_synthetic_data(tmp_path):
    """train_and_evaluate trả đủ 4 mô hình chính thức."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6))
    true_w = np.array([3.5, -2.0, 1.2, 0.8, -1.5, 2.2])
    y = X @ true_w + 10.0 + rng.normal(scale=0.1, size=80)
    data = {
        "X_train": X[:50],
        "y_train": y[:50],
        "X_val": X[50:65],
        "y_val": y[50:65],
        "X_test": X[65:],
        "y_test": y[65:],
    }

    table, predictions = train_and_evaluate(data, tmp_path, evaluation_split="test")

    assert len(table) == 4
    assert set(table["model"]) == set(MODEL_NAMES)
    assert set(predictions) == set(MODEL_NAMES)
    assert (tmp_path / "mlp_torch_parameters.pt").is_file()
    assert (tmp_path / "linear_regression_parameters.json").is_file()
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        assert (tmp_path / f"actual_vs_predicted_{slug}.png").is_file()
