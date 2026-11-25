"""Kiểm tra MLP PyTorch: kiến trúc, huấn luyện, lưu/nạp và lặp lại được."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import numpy as np
import pytest
import torch
from torch import nn

from house_price.evaluation import regression_metrics
from house_price.models.mlp_torch import MLPTorchRegressorModel


def _synthetic_data(n=80, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10
    return X, y


def test_torch_mlp_has_expected_architecture():
    """Kiến trúc phải là Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)."""
    model = MLPTorchRegressorModel()
    model.build_model(input_size=88)

    linears = [m for m in model.network.net if isinstance(m, nn.Linear)]
    relus = [m for m in model.network.net if isinstance(m, nn.ReLU)]

    assert len(linears) == 3
    assert [tuple(layer.weight.shape) for layer in linears] == [(32, 88), (16, 32), (1, 16)]
    assert len(relus) == 2
    assert model.network.net[-1] is linears[-1]


def test_torch_mlp_trains_loss_decreases_and_saves(tmp_path):
    """Huấn luyện vài epoch: loss giảm, dự đoán hữu hạn, file .pt được lưu."""
    X, y = _synthetic_data()
    save_path = tmp_path / "mlp_torch_parameters.pt"

    model = MLPTorchRegressorModel(epochs=5, batch_size=16, save_path=save_path)
    model.build_model(input_size=X.shape[1])
    model.fit(X, y)

    predictions = model.predict(X[:7])

    assert predictions.shape == (7,)
    assert np.isfinite(predictions).all()
    assert len(model.history_["loss"]) == 5
    assert model.history_["loss"][-1] < model.history_["loss"][0]
    assert save_path.is_file()


def test_torch_mlp_tracks_validation_loss(tmp_path):
    """Fit kèm X_val/y_val thì history_ có val_loss đủ số epoch."""
    X, y = _synthetic_data(n=80)
    model = MLPTorchRegressorModel(
        epochs=5,
        batch_size=16,
        save_path=tmp_path / "mlp_torch_parameters.pt",
    )
    model.build_model(input_size=X.shape[1])
    model.fit(X[:60], y[:60], X_val=X[60:], y_val=y[60:])

    assert len(model.history_["loss"]) == 5
    assert len(model.history_["val_loss"]) == 5
    assert np.isfinite(model.history_["val_loss"]).all()


def test_torch_mlp_save_and_load_round_trip(tmp_path):
    """Nạp lại mô hình đã lưu phải cho cùng dự đoán."""
    X, y = _synthetic_data()
    save_path = tmp_path / "mlp_torch_parameters.pt"

    model = MLPTorchRegressorModel(epochs=5, batch_size=16, save_path=save_path)
    model.build_model(input_size=X.shape[1])
    model.fit(X, y)
    before = model.predict(X[:10])

    loaded = MLPTorchRegressorModel.load(save_path)
    after = loaded.predict(X[:10])

    assert np.allclose(before, after)
    assert loaded.hidden_layer_sizes == (32, 16)
    assert loaded._input_size == X.shape[1]


def test_torch_mlp_reproducible_with_same_seed(tmp_path):
    """Cùng seed cho cùng dự đoán; khác seed cho trọng số khởi tạo khác."""
    X, y = _synthetic_data()

    first = MLPTorchRegressorModel(epochs=5, batch_size=16, save_path=tmp_path / "a.pt")
    first.build_model(input_size=X.shape[1])
    first.fit(X, y)

    second = MLPTorchRegressorModel(epochs=5, batch_size=16, save_path=tmp_path / "b.pt")
    second.build_model(input_size=X.shape[1])
    second.fit(X, y)

    assert np.allclose(first.predict(X[:10]), second.predict(X[:10]))

    third = MLPTorchRegressorModel(
        epochs=5, batch_size=16, random_state=7, save_path=tmp_path / "c.pt"
    )
    third.build_model(input_size=X.shape[1])
    first_weights = list(first.network.parameters())
    third_weights = list(third.network.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(first_weights, third_weights))


def test_torch_mlp_metrics_reasonable_on_synthetic_data(tmp_path):
    """Trên dữ liệu tuyến tính đơn giản, R2 phải cao khi đủ số epoch.

    Adam lr=0.001 cần đủ số bước để hội tụ (đã kiểm chứng: 500 epoch
    cho R2 0,96-0,99 trên 4 seed khác nhau); 30 epoch là chưa đủ cho
    bản PyTorch nên không dùng làm ngưỡng ở đây.
    """
    X, y = _synthetic_data(n=200, seed=0)
    model = MLPTorchRegressorModel(
        epochs=500, batch_size=32, save_path=tmp_path / "mlp_torch_parameters.pt"
    )
    model.build_model(input_size=X.shape[1])
    model.fit(X[:150], y[:150])

    metrics = regression_metrics(y[150:], model.predict(X[150:]))

    assert metrics["r2"] > 0.9
    assert np.isfinite(metrics["rmse"])


def test_torch_mlp_rejects_invalid_parameters():
    """Tham số không hợp lệ phải lỗi rõ ràng."""
    with pytest.raises(ValueError, match="hidden_layer_sizes"):
        MLPTorchRegressorModel(hidden_layer_sizes=())
    with pytest.raises(ValueError, match="learning_rate"):
        MLPTorchRegressorModel(learning_rate=0)
    with pytest.raises(ValueError, match="epochs"):
        MLPTorchRegressorModel(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        MLPTorchRegressorModel(batch_size=0)

    model = MLPTorchRegressorModel()
    with pytest.raises(ValueError, match="input_size"):
        model.build_model(input_size=0)


def test_torch_mlp_requires_build_before_fit_and_fit_before_predict():
    """Chưa build thì không fit được; chưa fit thì không predict được."""
    model = MLPTorchRegressorModel()
    with pytest.raises(RuntimeError, match="Chưa build_model"):
        model.fit(np.zeros((2, 5)), np.zeros(2))

    model.build_model(input_size=5)
    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(np.zeros((2, 5)))

    with pytest.raises(TypeError):
        MLPTorchRegressorModel().build_model()
