"""Kiểm tra hợp đồng class mô hình: thứ tự phương thức và báo lỗi."""

from __future__ import annotations

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


def test_unimplemented_models_raise_not_implemented():
    for cls in (MLPRegressorModel, DecisionTreeRegressorModel, RandomForestRegressorModel):
        model = cls()
        with pytest.raises(NotImplementedError):
            model.build_model()
