"""Kiểm tra Decision Tree Regression viết tay (CART, Stage 04)."""

from __future__ import annotations

import numpy as np
import pytest

from house_price.models.decision_tree import DecisionTreeRegressorModel


def _tree_depth(node: dict) -> int:
    """Độ sâu của cây: lá ở độ sâu 0, gốc sâu nhất tính từ gốc."""
    if node["is_leaf"]:
        return node["depth"]
    return max(_tree_depth(node["left"]), _tree_depth(node["right"]))


def _leaf_sizes(node: dict) -> list[int]:
    """Số mẫu của mọi lá trong cây."""
    if node["is_leaf"]:
        return [node["n_samples"]]
    return _leaf_sizes(node["left"]) + _leaf_sizes(node["right"])


def _fit_model(X, y, **kwargs):
    model = DecisionTreeRegressorModel(**kwargs)
    model.build_model()
    model.fit(X, y)
    return model


def test_simple_split_finds_threshold_and_leaf_means():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [8.0], [9.0], [10.0], [11.0]])
    y = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0])

    model = _fit_model(X, y)

    assert model.tree_["is_leaf"] is False
    assert model.tree_["feature"] == 0
    assert model.tree_["threshold"] == pytest.approx(5.5)
    assert model.tree_["left"]["value"] == pytest.approx(1.0)
    assert model.tree_["right"]["value"] == pytest.approx(10.0)
    assert np.allclose(model.predict(np.array([[0.5], [9.0]])), [1.0, 10.0])


def test_sample_exactly_on_threshold_goes_left():
    # Ngưỡng tìm được là 5.5 (trung điểm của 3 và 8); mẫu x = 5.5 đúng
    # bằng ngưỡng phải đi nhánh trái theo quy ước x <= threshold.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [8.0], [9.0], [10.0], [11.0]])
    y = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0])

    model = _fit_model(X, y)

    assert model.tree_["threshold"] == pytest.approx(5.5)
    assert model.predict(np.array([[5.5]])) == pytest.approx([1.0])
    # Ngay trên ngưỡng thì đi nhánh phải.
    assert model.predict(np.array([[5.500001]])) == pytest.approx([10.0])


def test_binary_feature_with_two_distinct_values():
    # Đặc trưng nhị phân (dạng one-hot 0/1): chỉ có một ngưỡng ứng viên
    # là 0.5, cây phải chia đúng tại đó.
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([2.0, 2.0, 2.0, 8.0, 8.0, 8.0])

    model = _fit_model(X, y)

    assert model.tree_["is_leaf"] is False
    assert model.tree_["threshold"] == pytest.approx(0.5)
    assert np.allclose(model.predict(np.array([[0.0], [1.0]])), [2.0, 8.0])


def test_binary_one_hot_columns_split_correctly():
    # Hai cột one-hot bù nhau: cây chỉ cần chia trên một cột là tách
    # được hai nhóm, lá dự đoán đúng trung bình từng nhóm.
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([5.0, 5.0, 9.0, 9.0])

    model = _fit_model(X, y)

    assert model.tree_["is_leaf"] is False
    assert model.tree_["threshold"] == pytest.approx(0.5)
    assert np.allclose(model.predict(X), y)


def test_max_depth_is_respected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + X[:, 2]

    model = _fit_model(X, y, max_depth=2)

    assert _tree_depth(model.tree_) == 2
    assert np.isfinite(model.predict(X)).all()


def test_constant_target_gives_single_leaf():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = np.full(20, 7.5)

    model = _fit_model(X, y)

    assert model.tree_["is_leaf"] is True
    assert model.tree_["value"] == pytest.approx(7.5)
    assert np.allclose(model.predict(X), 7.5)


def test_min_samples_leaf_is_respected():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] * 5.0 + rng.normal(size=40)

    model = _fit_model(X, y, min_samples_leaf=5)

    assert min(_leaf_sizes(model.tree_)) >= 5


def test_same_seed_gives_same_predictions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(size=50) * 0.1

    first = _fit_model(X, y, max_depth=4, max_features="sqrt", random_state=7)
    second = _fit_model(X, y, max_depth=4, max_features="sqrt", random_state=7)

    assert np.array_equal(first.predict(X), second.predict(X))


def test_max_features_sqrt_runs_and_sampling_is_effective():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 9))
    y = 4.0 * X[:, 0] + rng.normal(size=60) * 0.1

    model = _fit_model(X, y, max_depth=3, max_features="sqrt")

    predictions = model.predict(X)
    assert predictions.shape == (60,)
    assert np.isfinite(predictions).all()

    # Lấy mẫu đặc trưng có hiệu lực: khi mỗi nút chỉ xét 1 trong 4
    # đặc trưng, đặc trưng mang thông tin (cột 0) không phải lúc nào
    # cũng được chọn làm gốc.
    X_small = rng.normal(size=(40, 4))
    y_small = 5.0 * X_small[:, 0]
    root_features = set()
    for seed in range(50):
        sampled = _fit_model(X_small, y_small, max_depth=2, max_features=1, random_state=seed)
        root_features.add(sampled.tree_["feature"])
    assert len(root_features) > 1
    assert 0 in root_features


def test_predict_before_fit_and_fit_before_build_raise():
    model = DecisionTreeRegressorModel()
    model.build_model()

    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(np.zeros((2, 3)))

    unbuilt = DecisionTreeRegressorModel()
    with pytest.raises(RuntimeError, match="Chưa build_model"):
        unbuilt.fit(np.zeros((2, 3)), np.zeros(2))

    # build_model lại sau khi fit sẽ reset trạng thái đã học.
    fitted = _fit_model(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 0.0, 1.0, 1.0]))
    fitted.build_model()
    with pytest.raises(RuntimeError, match="chưa fit"):
        fitted.predict(np.zeros((2, 1)))


def test_predict_with_wrong_shape_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = _fit_model(X, y)

    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        model.predict(np.zeros(1))


def test_json_round_trip_preserves_predictions(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 4))
    y = 2.5 * X[:, 0] - X[:, 2] ** 2
    model = _fit_model(X, y, max_depth=5)

    path = model.export_json(tmp_path / "decision_tree.json", feature_names=["a", "b", "c", "d"])
    loaded = DecisionTreeRegressorModel.load_json(path)

    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    np.testing.assert_array_equal(loaded.feature_importances_, model.feature_importances_)
