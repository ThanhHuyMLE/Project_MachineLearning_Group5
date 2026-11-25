"""Kiểm tra RandomForestRegressorModel viết tay.

Dữ liệu synthetic nhỏ, có nhiễu, quan hệ phi tuyến nhẹ để cây CART
viết tay học được và để các cây bootstrap khác nhau thấy rõ.
"""

from __future__ import annotations

import numpy as np
import pytest

from house_price.models.random_forest import RandomForestRegressorModel


def _make_noisy_data(seed: int = 0, n_samples: int = 120, n_features: int = 8):
    """Dữ liệu synthetic: y phụ thuộc mạnh vào 3 đặc trưng đầu + nhiễu."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    noise = rng.normal(scale=1.0, size=n_samples)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] ** 2 + noise
    return X, y


def _fit_forest(**kwargs):
    X, y = _make_noisy_data()
    forest = RandomForestRegressorModel(**kwargs)
    forest.build_model()
    forest.fit(X, y)
    return forest, X, y


def _tree_predictions(forest, X):
    return np.column_stack([tree.predict(X) for tree in forest.trees_])


def test_bootstrap_trees_differ():
    """Các cây bootstrap thực sự khác nhau, không phải bản sao của nhau."""
    forest, X, _ = _fit_forest(n_estimators=15, max_depth=6)

    assert len(forest.trees_) == 15
    tree_predictions = _tree_predictions(forest, X)

    # Phương sai giữa các cây trên cùng một điểm dữ liệu phải > 0:
    # nếu mọi cây học trên cùng dữ liệu thì chúng cho cùng dự đoán.
    assert tree_predictions.var(axis=1).mean() > 0
    assert not np.all(tree_predictions == tree_predictions[:, [0]])


def test_feature_sampling_sqrt_runs_and_differs_from_all_features():
    """max_features='sqrt' chạy được và cho rừng khác với max_features=None."""
    forest_sqrt, X, _ = _fit_forest(n_estimators=10, max_depth=4, max_features="sqrt")
    forest_all, _, _ = _fit_forest(n_estimators=10, max_depth=4, max_features=None)

    # Introspection (khi cây lưu lại tham số): mọi cây đều nhận "sqrt".
    if hasattr(forest_sqrt.trees_[0], "max_features"):
        assert all(tree.max_features == "sqrt" for tree in forest_sqrt.trees_)
    if hasattr(forest_all.trees_[0], "max_features"):
        assert all(tree.max_features is None for tree in forest_all.trees_)

    # Lấy mẫu đặc trưng làm thay đổi cây nên dự đoán của hai rừng khác nhau.
    assert not np.allclose(forest_sqrt.predict(X), forest_all.predict(X))


def test_predict_is_mean_of_individual_trees():
    """predict của rừng bằng đúng trung bình predict của từng cây."""
    forest, X, _ = _fit_forest(n_estimators=10, max_depth=4)

    expected = _tree_predictions(forest, X).mean(axis=1)

    assert np.allclose(forest.predict(X), expected, rtol=0, atol=0)


def test_reproducible_with_same_seed_and_different_with_other_seed():
    """Cùng seed cho cùng dự đoán; khác seed cho dự đoán khác nhau."""
    forest_a, X, _ = _fit_forest(n_estimators=10, max_depth=4, random_state=42)
    forest_b, _, _ = _fit_forest(n_estimators=10, max_depth=4, random_state=42)
    forest_c, _, _ = _fit_forest(n_estimators=10, max_depth=4, random_state=7)

    assert np.array_equal(forest_a.predict(X), forest_b.predict(X))
    assert not np.allclose(forest_a.predict(X), forest_c.predict(X))


def test_predict_before_fit_raises_runtime_error():
    """predict trước fit phải báo lỗi rõ ràng."""
    forest = RandomForestRegressorModel(n_estimators=5)
    forest.build_model()

    with pytest.raises(RuntimeError, match="chưa fit"):
        forest.predict(np.zeros((2, 8)))


def test_fit_before_build_raises_runtime_error():
    """fit trước build_model phải báo lỗi rõ ràng."""
    X, y = _make_noisy_data()
    forest = RandomForestRegressorModel(n_estimators=5)

    with pytest.raises(RuntimeError, match="Chưa build_model"):
        forest.fit(X, y)


def test_fit_predict_shapes_on_synthetic_data():
    """fit/predict đúng shape trên dữ liệu synthetic, dự đoán hữu hạn."""
    forest, X, y = _fit_forest(n_estimators=10, max_depth=4)

    predictions = forest.predict(X[:5])

    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    # Rừng học được tín hiệu (không chỉ trả hằng số), sai số nhỏ hơn
    # phương sai của y trên cùng tập train.
    assert np.mean((forest.predict(X) - y) ** 2) < np.var(y)


def test_invalid_shapes_raise_value_error():
    """Input sai shape phải báo ValueError, không fail âm thầm."""
    X, y = _make_noisy_data()

    forest = RandomForestRegressorModel(n_estimators=5)
    forest.build_model()
    with pytest.raises(ValueError):
        forest.fit(X.ravel(), y)
    with pytest.raises(ValueError):
        forest.fit(X, y[:10])
    with pytest.raises(ValueError):
        forest.fit(X, y.reshape(-1, 2))

    forest.fit(X, y)
    with pytest.raises(ValueError):
        forest.predict(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        forest.predict(np.zeros(8))


def test_build_model_resets_fitted_state():
    """build_model lại phải xóa trạng thái đã fit trước đó."""
    forest, X, _ = _fit_forest(n_estimators=5, max_depth=3)
    assert forest.predict(X[:2]).shape == (2,)

    forest.build_model()

    assert forest.trees_ == []
    with pytest.raises(RuntimeError, match="chưa fit"):
        forest.predict(X[:2])


def test_json_round_trip_preserves_predictions(tmp_path):
    forest, X, _ = _fit_forest(n_estimators=5, max_depth=4)

    path = forest.export_json(tmp_path / "random_forest.json")
    loaded = RandomForestRegressorModel.load_json(path)

    np.testing.assert_array_equal(loaded.predict(X), forest.predict(X))
    assert len(loaded.trees_) == 5
