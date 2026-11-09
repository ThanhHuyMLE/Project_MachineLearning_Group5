"""Kiểm tra chỉ số Hồi quy và bảng so sánh mô hình."""

from __future__ import annotations

import numpy as np
import pytest

from house_price.evaluation import regression_metrics
from house_price.model_comparison import compare_models


def test_perfect_predictions_score_zero_error_and_r2_one():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_metrics_on_known_values():
    metrics = regression_metrics([100.0, 200.0], [110.0, 190.0])
    assert metrics["mse"] == 100.0
    assert metrics["rmse"] == 10.0
    assert metrics["mae"] == 10.0
    assert metrics["mape"] == pytest.approx(7.5)


def test_mape_is_none_when_true_value_is_zero():
    metrics = regression_metrics([0.0, 200.0], [10.0, 190.0])
    assert metrics["mape"] is None


def test_compare_models_sorts_by_mse_ascending():
    table = compare_models(
        {
            "MLP Regression": {"mse": 9.0, "rmse": 3.0, "mae": 2.0, "r2": 0.5},
            "Linear Regression": {"mse": 4.0, "rmse": 2.0, "mae": 1.0, "r2": 0.8},
        }
    )
    assert list(table["model"]) == ["Linear Regression", "MLP Regression"]
    assert list(table.columns) == ["model", "mse", "rmse", "mae", "r2"]


def test_r2_can_be_negative_for_bad_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 10.0, 10.0])
    assert regression_metrics(y_true, y_pred)["r2"] < 0
