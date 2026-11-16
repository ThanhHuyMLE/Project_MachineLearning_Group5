"""MLP Regression viết tay bằng NumPy cho bài toán giá nhà King County.

Kiến trúc mặc định:
Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)

Forward Propagation, Backpropagation, MSE và Adam được cài đặt trực tiếp.
Không sử dụng thư viện Machine Learning.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from house_price.models.base import BaseRegressorModel


class MLPRegressorModel(BaseRegressorModel):
    """MLP hồi quy viết tay, lưu tham số thành JSON ngay sau khi fit."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        save_path: str | Path = "outputs/mlp_parameters.json",
    ) -> None:
        super().__init__()

        if not hidden_layer_sizes or any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes phải chứa số nguyên dương.")
        if learning_rate <= 0:
            raise ValueError("learning_rate phải lớn hơn 0.")
        if epochs <= 0:
            raise ValueError("epochs phải lớn hơn 0.")
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 và beta2 phải nằm trong [0, 1).")
        if epsilon <= 0:
            raise ValueError("epsilon phải lớn hơn 0.")

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.save_path = Path(save_path)

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self._input_size: int | None = None

        # Các biến trạng thái của Adam.
        self._adam_m_weights: list[np.ndarray] = []
        self._adam_v_weights: list[np.ndarray] = []
        self._adam_m_biases: list[np.ndarray] = []
        self._adam_v_biases: list[np.ndarray] = []
        self._adam_step = 0

    def build_model(self, input_size: int):
        """Khởi tạo trọng số và bias của mạng."""
        if input_size <= 0:
            raise ValueError("input_size phải lớn hơn 0.")

        layer_sizes = (input_size, *self.hidden_layer_sizes, 1)
        rng = np.random.default_rng(self.random_state)

        self.weights_ = []
        self.biases_ = []

        # He initialization giúp khởi tạo trọng số phù hợp với ReLU.
        for fan_in, fan_out in pairwise(layer_sizes):
            weight = rng.normal(
                loc=0.0,
                scale=np.sqrt(2.0 / fan_in),
                size=(fan_in, fan_out),
            )
            bias = np.zeros((1, fan_out), dtype=float)

            self.weights_.append(weight)
            self.biases_.append(bias)

        # Khởi tạo các moment của Adam bằng 0.
        self._adam_m_weights = [np.zeros_like(weight) for weight in self.weights_]
        self._adam_v_weights = [np.zeros_like(weight) for weight in self.weights_]
        self._adam_m_biases = [np.zeros_like(bias) for bias in self.biases_]
        self._adam_v_biases = [np.zeros_like(bias) for bias in self.biases_]
        self._adam_step = 0

        self._input_size = input_size
        self.history_ = {"loss": [], "val_loss": []}
        self._is_built = True
        self._is_fitted = False
        return self

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        """Hàm kích hoạt ReLU: max(0, z)."""
        return np.maximum(0.0, values)

    @staticmethod
    def _relu_derivative(values: np.ndarray) -> np.ndarray:
        """Đạo hàm ReLU dùng trong Backpropagation."""
        return (values > 0.0).astype(float)

    def _forward(self, X: np.ndarray):
        """Thực hiện Forward Propagation qua toàn bộ mạng."""
        activations = [X]
        pre_activations = []
        current = X

        for layer_index, (weight, bias) in enumerate(
            zip(self.weights_, self.biases_)
        ):
            z = current @ weight + bias
            pre_activations.append(z)

            if layer_index == len(self.weights_) - 1:
                # Tầng output dùng Linear activation: f(z) = z.
                current = z
            else:
                current = self._relu(z)

            activations.append(current)

        # current có shape (số mẫu, 1).
        return current, (activations, pre_activations)

    def _backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cache,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Tính gradient MSE theo từng trọng số và bias."""
        activations, pre_activations = cache
        sample_count = y_true.shape[0]
        layer_count = len(self.weights_)

        # Đạo hàm MSE theo đầu ra của mạng.
        delta = (2.0 / sample_count) * (y_pred - y_true)

        grad_weights: list[np.ndarray | None] = [None] * layer_count
        grad_biases: list[np.ndarray | None] = [None] * layer_count

        # Truyền gradient từ output ngược về các hidden layer.
        for layer_index in range(layer_count - 1, -1, -1):
            grad_weights[layer_index] = activations[layer_index].T @ delta
            grad_biases[layer_index] = np.sum(delta, axis=0, keepdims=True)

            if layer_index > 0:
                delta = delta @ self.weights_[layer_index].T
                delta *= self._relu_derivative(
                    pre_activations[layer_index - 1]
                )

        return grad_weights, grad_biases

    def _adam_update(
        self,
        grad_weights: list[np.ndarray],
        grad_biases: list[np.ndarray],
    ) -> None:
        """Cập nhật trọng số và bias bằng Adam."""
        self._adam_step += 1
        step = self._adam_step

        for index in range(len(self.weights_)):
            # Moment bậc nhất và bậc hai cho weights.
            self._adam_m_weights[index] = (
                self.beta1 * self._adam_m_weights[index]
                + (1.0 - self.beta1) * grad_weights[index]
            )
            self._adam_v_weights[index] = (
                self.beta2 * self._adam_v_weights[index]
                + (1.0 - self.beta2) * np.square(grad_weights[index])
            )

            # Moment bậc nhất và bậc hai cho biases.
            self._adam_m_biases[index] = (
                self.beta1 * self._adam_m_biases[index]
                + (1.0 - self.beta1) * grad_biases[index]
            )
            self._adam_v_biases[index] = (
                self.beta2 * self._adam_v_biases[index]
                + (1.0 - self.beta2) * np.square(grad_biases[index])
            )

            # Hiệu chỉnh độ lệch ban đầu của các moment.
            m_weight_hat = self._adam_m_weights[index] / (
                1.0 - self.beta1**step
            )
            v_weight_hat = self._adam_v_weights[index] / (
                1.0 - self.beta2**step
            )
            m_bias_hat = self._adam_m_biases[index] / (
                1.0 - self.beta1**step
            )
            v_bias_hat = self._adam_v_biases[index] / (
                1.0 - self.beta2**step
            )

            # Cập nhật tham số.
            self.weights_[index] -= (
                self.learning_rate
                * m_weight_hat
                / (np.sqrt(v_weight_hat) + self.epsilon)
            )
            self.biases_[index] -= (
                self.learning_rate
                * m_bias_hat
                / (np.sqrt(v_bias_hat) + self.epsilon)
            )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Huấn luyện MLP theo mini-batch, theo dõi val_loss và tự lưu tham số thành JSON."""
        self._require_built()

        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X_train phải là ma trận hai chiều.")
        if X.shape[0] == 0:
            raise ValueError("X_train không được rỗng.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train và y_train phải có cùng số mẫu.")
        if X.shape[1] != self._input_size:
            raise ValueError(
                f"Mạng được tạo với {self._input_size} đặc trưng, "
                f"nhưng X_train có {X.shape[1]} đặc trưng."
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Dữ liệu train không được chứa NaN hoặc vô cực.")

        if (X_val is None) != (y_val is None):
            raise ValueError("Phải truyền đồng thời cả X_val và y_val.")

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=float)
            y_val_arr = np.asarray(y_val, dtype=float).reshape(-1, 1)
            if X_val_arr.ndim != 2 or X_val_arr.shape[1] != self._input_size:
                raise ValueError("Kích thước X_val không khớp với input_size của mạng.")
            if X_val_arr.shape[0] == 0 or X_val_arr.shape[0] != y_val_arr.shape[0]:
                raise ValueError("X_val và y_val không được rỗng và phải cùng số mẫu.")
            if not np.isfinite(X_val_arr).all() or not np.isfinite(y_val_arr).all():
                raise ValueError("Dữ liệu validation không được chứa NaN hoặc vô cực.")

        rng = np.random.default_rng(self.random_state)
        self.history_ = {"loss": [], "val_loss": []} if has_val else {"loss": []}

        for _epoch in range(self.epochs):
            # Xáo trộn thứ tự mẫu mỗi epoch.
            shuffled_indices = rng.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            # Chia dữ liệu thành mini-batch.
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward Propagation -> MSE gradient -> Backpropagation.
                y_pred, cache = self._forward(X_batch)
                grad_weights, grad_biases = self._backward(
                    y_batch,
                    y_pred,
                    cache,
                )

                # Dùng Adam cập nhật weights và biases.
                self._adam_update(grad_weights, grad_biases)

            # Tính MSE trên toàn bộ tập train sau mỗi epoch để theo dõi.
            train_predictions, _ = self._forward(X)
            epoch_loss = float(np.mean(np.square(y - train_predictions)))
            self.history_["loss"].append(epoch_loss)

            # Tính MSE trên tập validation (chỉ forward, không cập nhật trọng số).
            if has_val:
                val_predictions, _ = self._forward(X_val_arr)
                val_epoch_loss = float(np.mean(np.square(y_val_arr - val_predictions)))
                self.history_["val_loss"].append(val_epoch_loss)

        self._is_fitted = True

        # Lưu trực tiếp tham số đã học sau khi fit hoàn tất.
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": 1,
            "model_type": "MLPRegressorModel",
            "input_size": self._input_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weights": [weight.tolist() for weight in self.weights_],
            "biases": [bias.tolist() for bias in self.biases_],
            "history": self.history_,
        }

        with self.save_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return self
    
    def predict(self, X):
        """Dự đoán một giá trị cho mỗi mẫu đầu vào."""
        self._require_fitted()

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim != 2:
            raise ValueError("X phải là ma trận hai chiều.")
        if X_array.shape[1] != self._input_size:
            raise ValueError(
                f"Mạng cần {self._input_size} đặc trưng, "
                f"nhưng X có {X_array.shape[1]} đặc trưng."
            )
        if not np.isfinite(X_array).all():
            raise ValueError("X không được chứa NaN hoặc vô cực.")

        predictions, _ = self._forward(X_array)
        return predictions.ravel()