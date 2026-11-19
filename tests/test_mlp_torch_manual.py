"""Kiểm tra MLP PyTorch viết tay (``MLPTorchManualModel``, bản giáo dục).

Các test dùng dữ liệu synthetic nhỏ, seed cố định, mỗi test chạy độc
lập với nhau:

1. Gradient check: so sánh gradient giải tích từ ``_backward`` với
   gradient số bằng finite difference trung tâm, bao phủ toàn bộ
   trọng số (weights) VÀ bias của MỌI layer, thực hiện tại điểm tránh
   biên không khả vi của ReLU (kiểm tra trực tiếp ``min |z|`` ở các
   hidden layer trước khi so sánh).
2. Học được mapping tuyến tính đơn giản trên dữ liệu synthetic.
3. Training path không dùng autograd: quét source cấm gọi backward
   của autograd / ``autograd.grad`` / ``torch.optim``, và fit vẫn chạy
   được bên trong ``torch.no_grad()``.
4. Shape/dtype/device của tham số, lỗi khi chưa fit, sai input_size.
5. Một bước Adam viết tay di chuyển tham số ngược hướng gradient.
6. Round-trip save/load cho cùng dự đoán.
7. Gradient check in float64: same central-difference protocol on a
   two-hidden-layer net, isolating hand-written math errors from
   float32 rounding (see test_manual_gradient_float64_central_difference).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import numpy as np
import pytest
import torch

from house_price.evaluation import regression_metrics
from house_price.models.mlp_torch_manual import MLPTorchManualModel, mse_loss

# Ngưỡng sai số tương đối tối đa cho gradient check. Với tham số
# float32 và eps = 1e-3, sai số làm tròn của finite difference trung
# tâm cỡ 1e-3, nên ngưỡng 1e-2 là dung sai rộng rãi nhưng vẫn đủ chặt
# để bắt các lỗi đạo hàm thường gặp (ví dụ thiếu hệ số 2/n của MSE sẽ
# lệch cỡ O(1)). Giá trị đo được với mạng 3 -> 4 -> 1: 1.739e-03.
GRADIENT_TOLERANCE = 1e-2
# Bước h cho finite difference trung tâm.
FINITE_DIFF_EPS = 1e-3
# Ngưỡng tối thiểu cho |pre-activation| ở hidden layer: đảm bảo điểm
# kiểm tra gradient nằm cách xa biên không khả vi của ReLU (z = 0).
MIN_ABS_PRE_ACTIVATION = 1e-3

# Tolerance for the float64 gradient check. Measured on the 3 -> 4 -> 4 -> 1
# network with eps = 1e-3: max relative error 1.475e-11 (max(|a|, |n|)
# definition), max absolute error 8.517e-13. The 1e-7 tolerance is about
# 7000x above the measured error, yet 5 orders tighter than the float32
# threshold (1e-2) and still tight enough to catch common derivative bugs
# (for example a missing 2/n factor in the MSE derivative shifts the
# error by O(1)).
GRADIENT_TOLERANCE_FLOAT64 = 1e-7
# Step for the float64 central difference. Same eps as the float32 check,
# so the two errors are directly comparable. The eps sweep on this setup
# (1e-2 down to 1e-9) shows the error is rounding-dominated and shrinks
# as eps grows, while eps stays 30x below the smallest |z| (3.0e-02), so
# no perturbation crosses a ReLU kink.
FINITE_DIFF_EPS_FLOAT64 = 1e-3


def _synthetic_data(seed=7, n_samples=6, n_features=3):
    """Dữ liệu synthetic tuyến tính kèm nhiễu nhỏ, seed cố định."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([2.0, -1.0, 0.5])
    y = (X @ true_weights + 1.0).reshape(-1, 1)
    y = y + rng.normal(scale=0.05, size=(n_samples, 1))
    return X, y


def _max_relative_error(analytic, numeric):
    """Sai số tương đối lớn nhất, mẫu số có sàn để tránh chia cho 0."""
    abs_error = np.abs(analytic - numeric)
    denominator = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
    return float(np.max(abs_error / denominator))


def _relative_error_max(analytic, numeric):
    """Relative error with a max denominator: |a - n| / max(|a|, |n|, 1e-12)."""
    abs_error = np.abs(analytic - numeric)
    denominator = np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-12)
    return float(np.max(abs_error / denominator))


def test_manual_gradient_matches_finite_difference():
    """Gradient giải tích của ``_backward`` khớp finite difference.

    Bao phủ toàn bộ weights VÀ biases của MỌI layer trong mạng
    3 -> 4 -> 1 (1 hidden layer + 1 output layer). Điểm kiểm tra tránh
    biên không khả vi của ReLU: dữ liệu ngẫu nhiên liên tục nên
    pre-activation gần như chắc chắn khác 0, và test kiểm tra trực
    tiếp ``min |z|`` ở hidden layer trước khi so sánh.
    """
    X, y = _synthetic_data()
    model = MLPTorchManualModel(
        hidden_layer_sizes=(4,),
        learning_rate=0.01,
        epochs=1,
        batch_size=6,
        random_state=123,
    )
    model.build_model(input_size=3)

    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))

    y_pred, cache = model._forward(X_tensor)
    _, pre_activations = cache
    # Chỉ hidden layer dùng ReLU; tầng output dùng Linear nên không có
    # điểm gãy. Nếu min |z| quá nhỏ, dữ liệu nằm gần điểm gãy ReLU và
    # finite difference không còn đáng tin: báo setup sai thay vì báo
    # gradient sai.
    min_abs_pre_activation = min(float(torch.min(torch.abs(z))) for z in pre_activations[:-1])
    assert min_abs_pre_activation > MIN_ABS_PRE_ACTIVATION, (
        f"min |z| ở hidden layer là {min_abs_pre_activation:.3e}, "
        "quá gần điểm gãy ReLU (z = 0); đổi seed dữ liệu rồi chạy lại."
    )

    grad_weights, grad_biases = model._backward(y_tensor, y_pred, cache)

    max_error = 0.0
    worst_location = None
    layer_count = len(model.weights_)
    for layer_index in range(layer_count):
        for kind, params, grads in (
            ("weights", model.weights_, grad_weights),
            ("biases", model.biases_, grad_biases),
        ):
            for index in np.ndindex(params[layer_index].shape):
                old_value = float(params[layer_index][index])
                params[layer_index][index] = old_value + FINITE_DIFF_EPS
                loss_plus = float(mse_loss(model._forward(X_tensor)[0], y_tensor))
                params[layer_index][index] = old_value - FINITE_DIFF_EPS
                loss_minus = float(mse_loss(model._forward(X_tensor)[0], y_tensor))
                params[layer_index][index] = old_value

                numeric_grad = (loss_plus - loss_minus) / (2.0 * FINITE_DIFF_EPS)
                analytic_grad = float(grads[layer_index][index])
                error = _max_relative_error(np.array(analytic_grad), np.array(numeric_grad))
                if error > max_error:
                    max_error = error
                    worst_location = (kind, layer_index, index)

    print(f"gradient check max relative error: {max_error:.3e} at {worst_location}")
    assert max_error < GRADIENT_TOLERANCE, (
        f"Sai số tương đối tối đa là {max_error:.3e} tại {worst_location}, "
        f"vượt ngưỡng {GRADIENT_TOLERANCE:.0e}; "
        "nghi đạo hàm trong _backward sai, xem lại ReLU/MSE theo từng layer."
    )


def test_manual_gradient_float64_central_difference():
    """Analytic gradient from ``_backward`` matches finite differences in float64.

    Same central-difference protocol as the float32 check, but every
    tensor is float64 and the network has two hidden layers (3 -> 4 -> 4
    -> 1, 41 parameters), so weights and biases of the first, hidden and
    output layers are all verified. The float64 run isolates errors in
    the hand-written math from float32 rounding: at the same eps = 1e-3
    the float32 check measures 1.739e-03 (sum-denominator definition)
    while float64 measures 7.374e-12 under that same definition.

    Setup notes: ``build_model`` always creates float32 parameters, so
    the test casts the He-initialized weights and biases to float64
    (an exact cast, same values). ``_forward``, ``_backward`` and
    ``mse_loss`` are dtype-agnostic; only ``_relu_derivative`` returns
    float32, which is exact here because its values are 0/1, and the
    test asserts the analytic gradients come back float64.

    The check point avoids the ReLU kink: seeds are chosen so that every
    hidden pre-activation has |z| above 3e-02 and every hidden unit sees
    both positive and negative pre-activations across the batch. (The
    float32 test's seeds give a dead sample row in this two-hidden-layer
    net: one sample's first-layer activations are all zero, so its
    second-layer pre-activations are exactly 0 and the central
    difference disagrees with the z = 0 subgradient convention by O(1).
    That disagreement is a property of the kink, not of ``_backward``.)

    Relative error uses |a - n| / max(|a|, |n|, 1e-12).
    """
    X, y = _synthetic_data(seed=8)
    model = MLPTorchManualModel(
        hidden_layer_sizes=(4, 4),
        learning_rate=0.01,
        epochs=1,
        batch_size=6,
        random_state=7,
    )
    model.build_model(input_size=3)
    # build_model creates float32 parameters; cast them to float64 so the
    # whole forward/backward pass runs in float64.
    model.weights_ = [weight.to(torch.float64) for weight in model.weights_]
    model.biases_ = [bias.to(torch.float64) for bias in model.biases_]
    for param in [*model.weights_, *model.biases_]:
        assert param.dtype == torch.float64

    X_tensor = torch.from_numpy(X.astype(np.float64))
    y_tensor = torch.from_numpy(y.astype(np.float64))

    y_pred, cache = model._forward(X_tensor)
    _, pre_activations = cache
    for layer_index, z in enumerate(pre_activations[:-1]):
        z_np = z.numpy()
        min_abs = float(np.min(np.abs(z_np)))
        assert min_abs > MIN_ABS_PRE_ACTIVATION, (
            f"min |z| in hidden layer {layer_index} is {min_abs:.3e}, "
            "too close to the ReLU kink (z = 0); change the seeds."
        )
        assert ((z_np > 0).sum(axis=0) > 0).all() and ((z_np < 0).sum(axis=0) > 0).all(), (
            f"every unit of hidden layer {layer_index} must see both "
            "positive and negative pre-activations; change the seeds."
        )

    grad_weights, grad_biases = model._backward(y_tensor, y_pred, cache)
    for grad in [*grad_weights, *grad_biases]:
        assert grad.dtype == torch.float64

    max_abs_error = 0.0
    max_rel_error = 0.0
    worst_location = None
    param_count = 0
    layer_count = len(model.weights_)
    for layer_index in range(layer_count):
        for kind, params, grads in (
            ("weights", model.weights_, grad_weights),
            ("biases", model.biases_, grad_biases),
        ):
            for index in np.ndindex(params[layer_index].shape):
                param_count += 1
                old_value = float(params[layer_index][index])
                params[layer_index][index] = old_value + FINITE_DIFF_EPS_FLOAT64
                loss_plus = float(mse_loss(model._forward(X_tensor)[0], y_tensor))
                params[layer_index][index] = old_value - FINITE_DIFF_EPS_FLOAT64
                loss_minus = float(mse_loss(model._forward(X_tensor)[0], y_tensor))
                params[layer_index][index] = old_value

                numeric_grad = (loss_plus - loss_minus) / (2.0 * FINITE_DIFF_EPS_FLOAT64)
                analytic_grad = float(grads[layer_index][index])
                abs_error = abs(analytic_grad - numeric_grad)
                rel_error = _relative_error_max(np.array(analytic_grad), np.array(numeric_grad))
                max_abs_error = max(max_abs_error, abs_error)
                if rel_error > max_rel_error:
                    max_rel_error = rel_error
                    worst_location = (kind, layer_index, index)

    print(
        f"float64 gradient check: max abs error {max_abs_error:.3e}, "
        f"max relative error {max_rel_error:.3e} "
        "(|a - n| / max(|a|, |n|, 1e-12)), "
        f"n_params={param_count}, worst at {worst_location}"
    )
    assert param_count == 41
    assert max_rel_error < GRADIENT_TOLERANCE_FLOAT64, (
        f"float64 max relative error is {max_rel_error:.3e} "
        f"(max abs error {max_abs_error:.3e}) at {worst_location}, "
        f"above the tolerance {GRADIENT_TOLERANCE_FLOAT64:.0e}; "
        "the hand-written derivative in _backward is likely wrong."
    )


def test_manual_learns_linear_mapping(tmp_path):
    """Trên dữ liệu tuyến tính đơn giản, R2 phải cao khi đủ số epoch."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10

    model = MLPTorchManualModel(
        epochs=500,
        batch_size=32,
        save_path=tmp_path / "mlp_torch_manual_parameters.pt",
    )
    model.build_model(input_size=X.shape[1])
    model.fit(X[:150], y[:150])

    metrics = regression_metrics(y[150:], model.predict(X[150:]))
    print(f"synthetic learning R2: {metrics['r2']:.4f}")

    assert model.history_["loss"][-1] < model.history_["loss"][0]
    assert metrics["r2"] > 0.8
    assert np.isfinite(metrics["rmse"])


def test_manual_training_path_uses_no_autograd(tmp_path):
    """Training path không dùng autograd hay optimizer có sẵn của torch.

    Quét source của ``mlp_torch_manual.py``: cấm gọi backward của
    autograd (``.backward(``), ``autograd.grad`` và ``torch.optim``.
    Mẫu regex cố tình không khớp tên phương thức viết tay ``_backward``
    của chính model (``._backward(`` không chứa ``.backward(``).
    """
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "house_price"
        / "models"
        / "mlp_torch_manual.py"
    )
    source = source_path.read_text(encoding="utf-8")

    forbidden_patterns = [
        r"\.backward\s*\(",
        r"autograd\s*\.\s*grad",
        r"torch\.optim",
        r"from torch import optim",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, source), (
            f"Source chứa mẫu bị cấm {pattern!r}: training path không được "
            "dùng autograd hay optimizer có sẵn của torch."
        )

    # Bằng chứng chạy: fit vẫn hoàn tất bên trong torch.no_grad(),
    # điều không thể xảy ra nếu training path cần đồ thị autograd.
    X, y = _synthetic_data(seed=0, n_samples=40, n_features=3)
    model = MLPTorchManualModel(
        epochs=5,
        batch_size=16,
        save_path=tmp_path / "mlp_torch_manual_parameters.pt",
    )
    model.build_model(input_size=3)
    with torch.no_grad():
        model.fit(X, y)

    assert model._is_fitted
    assert np.isfinite(model.history_["loss"]).all()


def test_manual_parameters_shape_dtype_device_and_no_grad():
    """Tham số phải đúng shape, float32, CPU và requires_grad=False."""
    model = MLPTorchManualModel()
    model.build_model(input_size=5)

    assert [tuple(w.shape) for w in model.weights_] == [(5, 32), (32, 16), (16, 1)]
    assert [tuple(b.shape) for b in model.biases_] == [(1, 32), (1, 16), (1, 1)]
    for param in [*model.weights_, *model.biases_]:
        assert param.dtype == torch.float32
        assert param.device.type == "cpu"
        assert param.requires_grad is False


def test_manual_rejects_invalid_parameters_and_inputs(tmp_path):
    """Tham số và dữ liệu không hợp lệ phải lỗi rõ ràng."""
    with pytest.raises(ValueError, match="hidden_layer_sizes"):
        MLPTorchManualModel(hidden_layer_sizes=())
    with pytest.raises(ValueError, match="learning_rate"):
        MLPTorchManualModel(learning_rate=0)
    with pytest.raises(ValueError, match="epochs"):
        MLPTorchManualModel(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        MLPTorchManualModel(batch_size=0)

    model = MLPTorchManualModel()
    with pytest.raises(ValueError, match="input_size"):
        model.build_model(input_size=0)

    with pytest.raises(RuntimeError, match="Chưa build_model"):
        MLPTorchManualModel().fit(np.zeros((2, 5)), np.zeros(2))

    model.build_model(input_size=5)
    with pytest.raises(RuntimeError, match="chưa fit"):
        model.predict(np.zeros((2, 5)))

    X = np.zeros((4, 5))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="đặc trưng"):
        model.fit(np.zeros((4, 3)), y)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(np.full((4, 5), np.nan), y)
    with pytest.raises(ValueError, match="rỗng"):
        model.fit(np.zeros((0, 5)), np.zeros(0))

    fitted = MLPTorchManualModel(epochs=1, save_path=tmp_path / "m.pt")
    fitted.build_model(input_size=5)
    fitted.fit(X, y)
    with pytest.raises(ValueError, match="rỗng"):
        fitted.predict(np.zeros((0, 5)))

    with pytest.raises(TypeError):
        MLPTorchManualModel().build_model()


def test_manual_adam_step_moves_against_gradient():
    """Một bước Adam viết tay đi ngược hướng gradient, độ lớn không quá lr."""
    learning_rate = 0.01
    model = MLPTorchManualModel(
        hidden_layer_sizes=(4,),
        learning_rate=learning_rate,
        epochs=1,
        random_state=123,
    )
    model.build_model(input_size=3)

    old_weights = [weight.clone() for weight in model.weights_]
    old_biases = [bias.clone() for bias in model.biases_]
    # Gradient đã biết, khác dấu giữa weights (+) và biases (-) để
    # kiểm tra cả hai chiều di chuyển.
    grad_weights = [torch.full_like(weight, 0.7) for weight in model.weights_]
    grad_biases = [torch.full_like(bias, -0.4) for bias in model.biases_]

    model._adam_update(grad_weights, grad_biases)

    for layer_index in range(len(model.weights_)):
        weight_step = model.weights_[layer_index] - old_weights[layer_index]
        bias_step = model.biases_[layer_index] - old_biases[layer_index]
        # Ngược dấu gradient trên mọi phần tử.
        assert torch.all(weight_step < 0), "Adam phải giảm weights khi gradient dương."
        assert torch.all(bias_step > 0), "Adam phải tăng biases khi gradient âm."
        # Với Adam, |m_hat| <= sqrt(v_hat) nên |bước| <= learning rate;
        # đồng thời bước không được triệt tiêu (bước đầu phải xấp xỉ lr).
        # Dung sai 1e-6 cho phép sai số làm tròn float32 (bản NumPy
        # float64 không cần dung sai này): tỉ số m_hat/sqrt(v_hat) có
        # thể vượt 1 một lượng cỡ 1e-7 do làm tròn.
        assert torch.all(torch.abs(weight_step) <= learning_rate + 1e-6)
        assert torch.all(torch.abs(bias_step) <= learning_rate + 1e-6)
        assert float(torch.max(torch.abs(weight_step))) > 0.5 * learning_rate
        assert float(torch.max(torch.abs(bias_step))) > 0.5 * learning_rate


def test_manual_save_and_load_round_trip(tmp_path):
    """Nạp lại mô hình viết tay đã lưu phải cho cùng dự đoán."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10
    save_path = tmp_path / "mlp_torch_manual_parameters.pt"

    model = MLPTorchManualModel(
        epochs=5,
        batch_size=16,
        save_path=save_path,
    )
    model.build_model(input_size=X.shape[1])
    model.fit(X, y)
    before = model.predict(X[:10])

    loaded = MLPTorchManualModel.load(save_path)
    after = loaded.predict(X[:10])

    max_diff = float(np.max(np.abs(before - after)))
    print(f"manual round-trip max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-5
    assert loaded.hidden_layer_sizes == (32, 16)
    assert loaded._input_size == X.shape[1]

    with pytest.raises(ValueError, match="model_type"):
        bad_checkpoint = {
            "format_version": 1,
            "model_type": "MLPTorchRegressorModel",
            "input_size": 5,
            "hidden_layer_sizes": [32, 16],
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 16,
            "random_state": 42,
            "weights": model.weights_,
            "biases": model.biases_,
        }
        bad_path = tmp_path / "wrong_type.pt"
        torch.save(bad_checkpoint, bad_path)
        MLPTorchManualModel.load(bad_path)


def test_manual_reproducible_with_same_seed(tmp_path):
    """Cùng seed cho cùng dự đoán; khác seed cho trọng số khởi tạo khác."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10

    def build_and_fit(save_path, random_state=42):
        model = MLPTorchManualModel(
            epochs=5,
            batch_size=16,
            random_state=random_state,
            save_path=save_path,
        )
        model.build_model(input_size=X.shape[1])
        model.fit(X, y)
        return model

    first = build_and_fit(tmp_path / "a.pt")
    second = build_and_fit(tmp_path / "b.pt")
    assert np.allclose(first.predict(X[:10]), second.predict(X[:10]))

    third = build_and_fit(tmp_path / "c.pt", random_state=7)
    assert any(not torch.equal(a, b) for a, b in zip(first.weights_, third.weights_))
