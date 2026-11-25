"""Kiểm tra Linear Regression viết tay bằng batch gradient descent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from house_price.models import LinearRegressorModel


def _synthetic_data(seed=42, n_samples=200, noise=0.1):
    """Dữ liệu synthetic y = X w + b + nhiễu, w và b đã biết trước."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 3))
    true_beta = np.array([2.0, -1.5, 0.5])
    y = X @ true_beta + 4.0 + rng.normal(scale=noise, size=n_samples)
    return X, y, true_beta, 4.0


def _fitted_model(X, y, **kwargs):
    model = LinearRegressorModel(**kwargs)
    model.build_model()
    model.fit(X, y)
    return model


def test_default_hyperparameters():
    model = LinearRegressorModel()
    assert model.learning_rate == 0.1
    assert model.max_epochs == 1000
    assert model.tol == 1e-6
    with pytest.raises(ValueError, match="learning_rate"):
        LinearRegressorModel(learning_rate=0)
    with pytest.raises(ValueError, match="max_epochs"):
        LinearRegressorModel(max_epochs=0)
    with pytest.raises(ValueError, match="tol"):
        LinearRegressorModel(tol=0)


def test_recovers_known_coefficients():
    X, y, true_beta, true_intercept = _synthetic_data()
    model = _fitted_model(X, y)

    assert np.allclose(model.coef_, true_beta, atol=0.05)
    assert model.intercept_ == pytest.approx(true_intercept, abs=0.05)
    assert model.converged_ is True
    assert 1 <= model.n_iter_ <= model.max_epochs
    assert len(model.history_["loss"]) == model.n_iter_


def test_loss_decreases_monotonically():
    X, y, _, _ = _synthetic_data()
    model = _fitted_model(X, y)
    losses = model.history_["loss"]

    assert len(losses) >= 2
    # Gradient descent với learning rate ổn định làm loss giảm đơn điệu;
    # dung sai 1e-9 chỉ để trừ sai số làm tròn của float.
    assert np.all(np.diff(losses) <= 1e-9)
    assert losses[-1] < losses[0]


def test_matches_lstsq_reference():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    true_beta = np.array([1.5, -2.0, 0.5, 3.0, -0.25])
    y = X @ true_beta + 4.0 + rng.normal(scale=0.1, size=100)
    model = _fitted_model(X, y, tol=1e-8)

    # Nghiệm tham chiếu độc lập bằng lstsq; chỉ dùng trong TEST, không
    # dùng trong model (model dùng gradient descent viết tay).
    design = np.column_stack([np.ones(len(X)), X])
    reference, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    assert model.intercept_ == pytest.approx(reference[0], abs=1e-3)
    assert np.allclose(model.coef_, reference[1:], atol=1e-3)


def test_predict_before_fit_and_bad_shapes_raise():
    X, y, _, _ = _synthetic_data()
    model = LinearRegressorModel()
    model.build_model()
    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="chưa fit"):
        model.export_params("params.json")

    # y ngắn hơn X một dòng.
    with pytest.raises(ValueError, match="dòng"):
        model.fit(X, y[:-1])
    # X 3 chiều.
    with pytest.raises(ValueError, match="2 chiều"):
        model.fit(np.zeros((2, 2, 2)), np.zeros(2))

    model.fit(X, y)
    # Predict với số đặc trưng khác lúc fit.
    with pytest.raises(ValueError, match="đặc trưng"):
        model.predict(np.zeros((2, 5)))


def test_export_params_writes_readable_json(tmp_path):
    X, y, _, _ = _synthetic_data(n_samples=50)
    model = _fitted_model(X, y)
    path = model.export_params(tmp_path / "params.json", feature_names=["a", "b", "c"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "LinearRegression"
    assert "gradient descent" in payload["method"].lower()
    assert payload["n_features"] == 3
    assert payload["intercept"] == pytest.approx(model.intercept_)
    assert [item["feature"] for item in payload["coefficients"]] == ["a", "b", "c"]
    assert [item["weight"] for item in payload["coefficients"]] == pytest.approx(list(model.coef_))
    assert payload["learning_rate"] == model.learning_rate
    assert payload["n_epochs"] == model.n_iter_
    assert payload["final_loss"] == pytest.approx(model.history_["loss"][-1])
    assert payload["converged"] == model.converged_

    with pytest.raises(ValueError, match="tên đặc trưng"):
        model.export_params(tmp_path / "khac.json", feature_names=["chi-mot-ten"])


def test_json_round_trip_preserves_predictions(tmp_path):
    X, y, _, _ = _synthetic_data(n_samples=80)
    model = _fitted_model(X, y)

    path = model.export_json(tmp_path / "linear.json", feature_names=["a", "b", "c"])
    loaded = LinearRegressorModel.load_json(path)

    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_build_model_resets_fitted_state():
    X, y, _, _ = _synthetic_data(n_samples=20)
    model = _fitted_model(X, y)
    model.build_model()
    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(X[:1])


def test_model_file_uses_no_sklearn_estimator():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "house_price"
        / "models"
        / "linear_regression.py"
    ).read_text(encoding="utf-8")
    assert "sklearn" not in source
    assert "LinearRegression(" not in source
