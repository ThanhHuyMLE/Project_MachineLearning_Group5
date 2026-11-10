"""Kiểm tra hợp đồng và hoạt động của các lớp mô hình."""

from __future__ import annotations

import json

import numpy as np
import pytest

from house_price.models import (
    DecisionTreeRegressorModel,
    LinearRegressorModel,
    MLPRegressorModel,
    RandomForestRegressorModel,
)


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


def test_handwritten_mlp_trains_predicts_and_saves_parameters(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10

    parameter_path = tmp_path / "outputs" / "mlp_parameters.json"
    model = MLPRegressorModel(
        epochs=5,
        batch_size=16,
        save_path=parameter_path,
    )
    model.build_model(input_size=X.shape[1])
    model.fit(X, y)

    predictions = model.predict(X[:7])

    assert predictions.shape == (7,)
    assert np.isfinite(predictions).all()
    assert len(model.history_["loss"]) == 5
    assert parameter_path.is_file()

    saved = json.loads(parameter_path.read_text(encoding="utf-8"))
    assert saved["model_type"] == "MLPRegressorModel"
    assert saved["input_size"] == 5
    assert len(saved["weights"]) == 3
    assert len(saved["biases"]) == 3


def test_handwritten_mlp_requires_input_size_to_build():
    model = MLPRegressorModel()

    with pytest.raises(TypeError):
        model.build_model()


def test_other_unimplemented_models_raise_not_implemented():
    for model_class in (
        DecisionTreeRegressorModel,
        RandomForestRegressorModel,
    ):
        model = model_class()

        with pytest.raises(NotImplementedError):
            model.build_model()