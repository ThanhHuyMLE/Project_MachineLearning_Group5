"""Kiểm tra toán học của MLP viết tay (``MLPRegressorModel``).

Các test dùng dữ liệu synthetic nhỏ, seed cố định, mỗi test chạy
dưới 5 giây và độc lập với nhau:

1. Gradient check: so sánh gradient giải tích từ ``_backward`` với
   gradient số bằng finite difference trung tâm, bao phủ toàn bộ
   trọng số (weights) VÀ bias của MỌI layer (không chỉ 1 layer),
   thực hiện tại điểm tránh biên không khả vi của ReLU: dữ liệu
   ngẫu nhiên liên tục và khởi tạo He làm pre-activation gần như
   chắc chắn khác 0, test còn kiểm tra trực tiếp ``min |z|`` ở các
   hidden layer trước khi so sánh gradient.
2. Loss giảm rõ rệt khi fit trên dữ liệu synthetic.
3. Một bước Adam di chuyển tham số ngược hướng gradient, độ lớn
   không vượt quá learning rate.
4. Cùng seed cho cùng trọng số khởi tạo và cùng loss history.

Ghi chú về dung sai gradient check: với số thực float64 và
``eps = 1e-5``, sai số cắt cụt của finite difference trung tâm là
``O(eps**2)`` và sai số làm tròn khoảng ``1e-11``, nên ngưỡng
``1e-5`` cho sai số tương đối tối đa là dung sai rộng rãi nhưng vẫn
đủ chặt để bắt các lỗi đạo hàm thường gặp (ví dụ thiếu hệ số ``2/n``
của MSE sẽ lệch cỡ ``O(1)``). Cấu hình test đã đo trước với mạng
3 -> 4 -> 3 -> 1: sai số tương đối tối đa ``~1.7e-8`` (lớn nhất ở
layer giữa), ``min |z|`` ở các hidden layer ``~0.019``, cách xa
điểm gãy ReLU tại ``z = 0``.
"""

from __future__ import annotations

import numpy as np

from house_price.models import LinearRegressorModel
from house_price.models.mlp import MLPRegressorModel

# Ngưỡng sai số tương đối tối đa cho gradient check (xem ghi chú module).
GRADIENT_TOLERANCE = 1e-5
# Bước h cho finite difference trung tâm.
FINITE_DIFF_EPS = 1e-5
# Ngưỡng tối thiểu cho |pre-activation| ở hidden layer: đảm bảo điểm
# kiểm tra gradient nằm cách xa biên không khả vi của ReLU (z = 0).
# Với dữ liệu ngẫu nhiên liên tục, P(z = 0) = 0; ngưỡng này chỉ là
# chốt chặn để test báo sai setup thay vì báo sai gradient nếu dữ
# liệu vô tình rơi gần điểm gãy. Giá trị đo được với cấu hình test
# là ~0.019, gấp ~19 lần ngưỡng.
MIN_ABS_PRE_ACTIVATION = 1e-3


def _synthetic_data(seed=7, n_samples=6, n_features=3):
    """Dữ liệu synthetic tuyến tính kèm nhiễu nhỏ, seed cố định."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([2.0, -1.0, 0.5])
    y = (X @ true_weights + 1.0).reshape(-1, 1)
    y = y + rng.normal(scale=0.05, size=(n_samples, 1))
    return X, y


def _mse_loss(model, X, y):
    """Loss MSE trung bình mà ``_backward`` lấy đạo hàm theo."""
    predictions, _ = model._forward(X)
    return float(np.mean(np.square(predictions - y)))


def _max_relative_error(analytic, numeric):
    """Sai số tương đối lớn nhất, mẫu số có sàn để tránh chia cho 0."""
    abs_error = np.abs(analytic - numeric)
    denominator = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
    return float(np.max(abs_error / denominator))


def test_gradient_matches_finite_difference():
    """Gradient giải tích của ``_backward`` khớp finite difference.

    Bao phủ toàn bộ weights VÀ biases của MỌI layer trong mạng
    3 -> 4 -> 3 -> 1 (2 hidden layer + 1 output layer). Điểm kiểm
    tra tránh biên không khả vi của ReLU: dữ liệu ngẫu nhiên liên
    tục nên pre-activation gần như chắc chắn khác 0, và test kiểm
    tra trực tiếp ``min |z|`` ở các hidden layer trước khi so sánh.
    """
    X, y = _synthetic_data()
    model = MLPRegressorModel(
        hidden_layer_sizes=(4, 3),
        learning_rate=0.01,
        epochs=1,
        batch_size=6,
        random_state=123,
    )
    model.build_model(input_size=3)

    y_pred, cache = model._forward(X)
    _, pre_activations = cache
    # Chỉ các hidden layer dùng ReLU; tầng output dùng Linear nên
    # không có điểm gãy. Nếu min |z| quá nhỏ, dữ liệu nằm gần điểm
    # gãy ReLU và finite difference không còn đáng tin: báo setup
    # sai thay vì báo gradient sai.
    min_abs_pre_activation = min(float(np.min(np.abs(z))) for z in pre_activations[:-1])
    assert min_abs_pre_activation > MIN_ABS_PRE_ACTIVATION, (
        f"min |z| ở hidden layer là {min_abs_pre_activation:.3e}, "
        "quá gần điểm gãy ReLU (z = 0); đổi seed dữ liệu rồi chạy lại."
    )

    grad_weights, grad_biases = model._backward(y, y_pred, cache)

    max_error_per_layer = {}
    worst_location = None
    for layer_index in range(len(model.weights_)):
        layer_max_error = 0.0
        for kind, params, grads in (
            ("weights", model.weights_, grad_weights),
            ("biases", model.biases_, grad_biases),
        ):
            for index in np.ndindex(params[layer_index].shape):
                old_value = params[layer_index][index]
                params[layer_index][index] = old_value + FINITE_DIFF_EPS
                loss_plus = _mse_loss(model, X, y)
                params[layer_index][index] = old_value - FINITE_DIFF_EPS
                loss_minus = _mse_loss(model, X, y)
                params[layer_index][index] = old_value

                numeric_grad = (loss_plus - loss_minus) / (2.0 * FINITE_DIFF_EPS)
                analytic_grad = float(grads[layer_index][index])
                error = _max_relative_error(np.array(analytic_grad), np.array(numeric_grad))
                if error > layer_max_error:
                    layer_max_error = error
                    worst_location = (kind, layer_index, index)
        max_error_per_layer[layer_index] = layer_max_error

    assert len(max_error_per_layer) == 3, (
        "Gradient check phải bao phủ cả 3 layer của mạng 3 -> 4 -> 3 -> 1, "
        f"chỉ kiểm tra được {len(max_error_per_layer)} layer."
    )
    for layer_index, layer_error in max_error_per_layer.items():
        assert layer_error < GRADIENT_TOLERANCE, (
            f"Sai số tương đối tối đa ở layer {layer_index} là "
            f"{layer_error:.3e} tại {worst_location}, "
            f"vượt ngưỡng {GRADIENT_TOLERANCE:.0e}; "
            "nghi đạo hàm trong _backward sai, xem lại ReLU/MSE theo từng layer."
        )


def test_loss_decreases_on_synthetic_data(tmp_path):
    """Loss train giảm rõ rệt sau khi fit dữ liệu synthetic."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    y = (3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10).reshape(-1, 1)

    model = MLPRegressorModel(
        hidden_layer_sizes=(8,),
        learning_rate=0.05,
        epochs=60,
        batch_size=16,
        random_state=42,
        save_path=tmp_path / "mlp_parameters.json",
    )
    model.build_model(input_size=3)
    model.fit(X, y)

    losses = model.history_["loss"]
    assert len(losses) == 60
    assert losses[-1] < 0.05 * losses[0], (
        f"Loss cuối {losses[-1]:.4f} không giảm đủ so với loss đầu {losses[0]:.4f}."
    )
    assert losses[-1] < 1.0


def test_adam_step_moves_against_gradient():
    """Một bước Adam đi ngược hướng gradient, độ lớn không quá learning rate."""
    learning_rate = 0.01
    model = MLPRegressorModel(
        hidden_layer_sizes=(4,),
        learning_rate=learning_rate,
        epochs=1,
        random_state=123,
    )
    model.build_model(input_size=3)

    old_weights = [weight.copy() for weight in model.weights_]
    old_biases = [bias.copy() for bias in model.biases_]
    # Gradient đã biết, khác dấu giữa weights (+) và biases (-) để
    # kiểm tra cả hai chiều di chuyển.
    grad_weights = [np.full_like(weight, 0.7) for weight in model.weights_]
    grad_biases = [np.full_like(bias, -0.4) for bias in model.biases_]

    model._adam_update(grad_weights, grad_biases)

    for layer_index in range(len(model.weights_)):
        weight_step = model.weights_[layer_index] - old_weights[layer_index]
        bias_step = model.biases_[layer_index] - old_biases[layer_index]
        # Ngược dấu gradient trên mọi phần tử.
        assert np.all(weight_step < 0), "Adam phải giảm weights khi gradient dương."
        assert np.all(bias_step > 0), "Adam phải tăng biases khi gradient âm."
        # Với Adam, |m_hat| <= sqrt(v_hat) nên |bước| <= learning rate;
        # đồng thời bước không được triệt tiêu (bước đầu phải xấp xỉ lr).
        assert np.all(np.abs(weight_step) <= learning_rate)
        assert np.all(np.abs(bias_step) <= learning_rate)
        assert np.max(np.abs(weight_step)) > 0.5 * learning_rate
        assert np.max(np.abs(bias_step)) > 0.5 * learning_rate


def test_same_seed_gives_same_weights_and_history(tmp_path):
    """Cùng seed cho cùng trọng số khởi tạo và cùng loss history."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = (2.0 * X[:, 0] + 1.0).reshape(-1, 1)

    def build_and_fit(save_path):
        model = MLPRegressorModel(
            hidden_layer_sizes=(4,),
            learning_rate=0.01,
            epochs=5,
            batch_size=16,
            random_state=123,
            save_path=save_path,
        )
        model.build_model(input_size=3)
        model.fit(X, y)
        return model

    first = build_and_fit(tmp_path / "first.json")
    second = build_and_fit(tmp_path / "second.json")

    for first_weight, second_weight in zip(first.weights_, second.weights_):
        assert np.array_equal(first_weight, second_weight)
    for first_bias, second_bias in zip(first.biases_, second.biases_):
        assert np.array_equal(first_bias, second_bias)
    assert first.history_ == second.history_


def test_build_model_signature_is_compatible():
    """MLP nhận input_size, model khác vẫn build được không tham số."""
    mlp = MLPRegressorModel(hidden_layer_sizes=(4,))
    mlp.build_model(input_size=3)
    assert mlp._input_size == 3
    assert len(mlp.weights_) == 2

    linear = LinearRegressorModel()
    linear.build_model()
    assert linear._is_built
