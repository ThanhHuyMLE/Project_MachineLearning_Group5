"""MLP Regression bằng PyTorch viết tay (bản giáo dục) cho bài toán giá nhà.

Kiến trúc mặc định (giống hệt bản PyTorch chính thức):
Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)

Bản này dùng tensor torch cho mọi phép tính học, nhưng KHÔNG dùng autograd
và KHÔNG dùng optimizer có sẵn của torch: Forward Propagation,
Backpropagation theo chain rule, hàm mất mát MSE và Adam đều được cài đặt
trực tiếp, tham số đặt requires_grad=False để chứng minh training path
không đi qua autograd.

Lưu ý quan trọng:
- Đây là bản giáo dục, KHÔNG hứa parity bit-for-bit với bản PyTorch
  chính thức (house_price/models/mlp_torch.py): thứ tự phép tính số thực, cách xáo
  trộn mini-batch và cách khởi tạo có thể khác nhau nên kết quả số có
  thể lệch nhỏ dù cùng kiến trúc và siêu tham số.
- Autograd KHÔNG phải backprop viết tay: autograd tự xây đồ thị tính
  toán và tự suy ra đạo hàm, còn ở đây đạo hàm từng lớp được viết tay
  bằng chain rule trong _backward.
- Optimizer Adam có sẵn của torch KHÔNG phải Adam viết tay: ở đây các
  moment m, v và bước hiệu chỉnh độ lệch được cập nhật trực tiếp trong
  _adam_update.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from house_price.models.base import BaseRegressorModel


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Hàm mất mát MSE: trung bình bình phương sai số trên toàn bộ batch."""
    return torch.mean(torch.square(y_pred - y_true))


class MLPTorchManualModel(BaseRegressorModel):
    """MLP hồi quy bằng PyTorch viết tay, lưu tham số thành .pt sau khi fit."""

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
        save_path: str | Path = "outputs/mlp_torch_manual_parameters.pt",
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

        # Tham số là tensor torch float32 trên CPU, requires_grad=False:
        # mọi gradient đều tính tay, không qua autograd.
        self.weights_: list[torch.Tensor] = []
        self.biases_: list[torch.Tensor] = []
        self._input_size: int | None = None

        # Các biến trạng thái của Adam.
        self._adam_m_weights: list[torch.Tensor] = []
        self._adam_v_weights: list[torch.Tensor] = []
        self._adam_m_biases: list[torch.Tensor] = []
        self._adam_v_biases: list[torch.Tensor] = []
        self._adam_step = 0

    def build_model(self, input_size: int):
        """Khởi tạo trọng số và bias của mạng."""
        if input_size <= 0:
            raise ValueError("input_size phải lớn hơn 0.")

        layer_sizes = (input_size, *self.hidden_layer_sizes, 1)
        generator = torch.Generator().manual_seed(self.random_state)

        self.weights_ = []
        self.biases_ = []

        # He initialization: trọng số lấy từ phân phối chuẩn N(0, 2/fan_in).
        # ReLU chặn một nửa đầu ra về 0 nên phương sai đầu vào của mỗi
        # tầng cần hệ số 2 để giữ phương sai tín hiệu ổn định qua các
        # tầng; bias khởi tạo bằng 0.
        for fan_in, fan_out in pairwise(layer_sizes):
            weight = torch.randn(
                (fan_in, fan_out),
                generator=generator,
                dtype=torch.float32,
            ) * math.sqrt(2.0 / fan_in)
            weight.requires_grad_(False)
            bias = torch.zeros((1, fan_out), dtype=torch.float32)
            bias.requires_grad_(False)

            self.weights_.append(weight)
            self.biases_.append(bias)

        # Khởi tạo các moment của Adam bằng 0.
        self._adam_m_weights = [torch.zeros_like(weight) for weight in self.weights_]
        self._adam_v_weights = [torch.zeros_like(weight) for weight in self.weights_]
        self._adam_m_biases = [torch.zeros_like(bias) for bias in self.biases_]
        self._adam_v_biases = [torch.zeros_like(bias) for bias in self.biases_]
        self._adam_step = 0

        self._input_size = input_size
        self.history_ = {"loss": [], "val_loss": []}
        self._is_built = True
        self._is_fitted = False
        return self

    @staticmethod
    def _relu(values: torch.Tensor) -> torch.Tensor:
        """Hàm kích hoạt ReLU: max(0, z)."""
        return torch.clamp(values, min=0.0)

    @staticmethod
    def _relu_derivative(values: torch.Tensor) -> torch.Tensor:
        """Đạo hàm ReLU dùng trong Backpropagation.

        Quy ước tại điểm gãy z = 0: đạo hàm bằng 0. ReLU không khả vi
        tại 0, nên đây là một lựa chọn subgradient; với dữ liệu liên
        tục, xác suất pre-activation đúng bằng 0 là không đáng kể.
        """
        return (values > 0).to(torch.float32)

    def _forward(self, X: torch.Tensor):
        """Thực hiện Forward Propagation qua toàn bộ mạng."""
        activations = [X]
        pre_activations = []
        current = X

        for layer_index, (weight, bias) in enumerate(zip(self.weights_, self.biases_)):
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
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        cache,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tính gradient MSE theo từng trọng số và bias bằng chain rule.

        Đạo hàm MSE theo đầu ra của mạng là (2/n) * (y_pred - y_true),
        rồi truyền ngược qua từng lớp: gradient của tầng output đi qua
        Linear, gradient của tầng ẩn đi qua đạo hàm ReLU.
        """
        activations, pre_activations = cache
        sample_count = y_true.shape[0]
        layer_count = len(self.weights_)

        # Đạo hàm MSE theo đầu ra của mạng.
        delta = (2.0 / sample_count) * (y_pred - y_true)

        grad_weights: list[torch.Tensor | None] = [None] * layer_count
        grad_biases: list[torch.Tensor | None] = [None] * layer_count

        # Truyền gradient từ output ngược về các hidden layer.
        for layer_index in range(layer_count - 1, -1, -1):
            grad_weights[layer_index] = activations[layer_index].T @ delta
            grad_biases[layer_index] = torch.sum(delta, dim=0, keepdim=True)

            if layer_index > 0:
                delta = delta @ self.weights_[layer_index].T
                delta = delta * self._relu_derivative(pre_activations[layer_index - 1])

        return grad_weights, grad_biases

    @torch.no_grad()
    def _adam_update(
        self,
        grad_weights: list[torch.Tensor],
        grad_biases: list[torch.Tensor],
    ) -> None:
        """Cập nhật trọng số và bias bằng Adam viết tay.

        Cập nhật chạy dưới torch.no_grad vì tham số có
        requires_grad=False và không có đồ thị autograd nào liên quan:
        đây là phép gán số học trực tiếp lên tensor tham số.
        """
        self._adam_step += 1
        step = self._adam_step

        for index in range(len(self.weights_)):
            # Moment bậc nhất và bậc hai cho weights.
            self._adam_m_weights[index] = (
                self.beta1 * self._adam_m_weights[index] + (1.0 - self.beta1) * grad_weights[index]
            )
            self._adam_v_weights[index] = self.beta2 * self._adam_v_weights[index] + (
                1.0 - self.beta2
            ) * torch.square(grad_weights[index])

            # Moment bậc nhất và bậc hai cho biases.
            self._adam_m_biases[index] = (
                self.beta1 * self._adam_m_biases[index] + (1.0 - self.beta1) * grad_biases[index]
            )
            self._adam_v_biases[index] = self.beta2 * self._adam_v_biases[index] + (
                1.0 - self.beta2
            ) * torch.square(grad_biases[index])

            # Hiệu chỉnh độ lệch ban đầu của các moment.
            m_weight_hat = self._adam_m_weights[index] / (1.0 - self.beta1**step)
            v_weight_hat = self._adam_v_weights[index] / (1.0 - self.beta2**step)
            m_bias_hat = self._adam_m_biases[index] / (1.0 - self.beta1**step)
            v_bias_hat = self._adam_v_biases[index] / (1.0 - self.beta2**step)

            # Cập nhật tham số.
            self.weights_[index] -= (
                self.learning_rate * m_weight_hat / (torch.sqrt(v_weight_hat) + self.epsilon)
            )
            self.biases_[index] -= (
                self.learning_rate * m_bias_hat / (torch.sqrt(v_bias_hat) + self.epsilon)
            )

    @torch.no_grad()
    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính MSE toàn bộ tập bằng forward, không cập nhật trọng số."""
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        predictions, _ = self._forward(X_tensor)
        return float(mse_loss(predictions, y_tensor))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Huấn luyện MLP viết tay theo mini-batch, theo dõi val_loss."""
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

        # Chuẩn bị dữ liệu (Dataset và DataLoader): chỉ dùng DataLoader
        # để nạp mini-batch, mọi phép tính học đều viết tay. DataLoader
        # xáo trộn thứ tự mẫu mỗi epoch bằng generator có seed cố định,
        # nên kết quả lặp lại được với cùng random_state.
        generator = torch.Generator().manual_seed(self.random_state)
        train_dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.history_ = {"loss": [], "val_loss": []} if has_val else {"loss": []}

        # Training loop: forward tay -> gradient tay -> Adam tay.
        for _epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                y_pred, cache = self._forward(X_batch)
                grad_weights, grad_biases = self._backward(y_batch, y_pred, cache)
                self._adam_update(grad_weights, grad_biases)

            # Tính MSE trên toàn bộ tập train sau mỗi epoch để theo dõi.
            epoch_loss = self._mse(X, y)
            if not math.isfinite(epoch_loss):
                raise RuntimeError("Loss huấn luyện không hữu hạn (NaN hoặc vô cực).")
            self.history_["loss"].append(epoch_loss)

            # Tính MSE trên tập validation (chỉ forward, không cập nhật trọng số).
            if has_val:
                val_epoch_loss = self._mse(X_val_arr, y_val_arr)
                if not math.isfinite(val_epoch_loss):
                    raise RuntimeError("Loss validation không hữu hạn (NaN hoặc vô cực).")
                self.history_["val_loss"].append(val_epoch_loss)

        self._is_fitted = True

        # Lưu trực tiếp tham số đã học sau khi fit hoàn tất.
        self.save()
        return self

    def predict(self, X):
        """Dự đoán một giá trị cho mỗi mẫu đầu vào."""
        self._require_fitted()

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim != 2:
            raise ValueError("X phải là ma trận hai chiều.")
        if X_array.shape[0] == 0:
            raise ValueError("X không được rỗng.")
        if X_array.shape[1] != self._input_size:
            raise ValueError(
                f"Mạng cần {self._input_size} đặc trưng, nhưng X có {X_array.shape[1]} đặc trưng."
            )
        if not np.isfinite(X_array).all():
            raise ValueError("X không được chứa NaN hoặc vô cực.")

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_array.astype(np.float32))
            predictions, _ = self._forward(X_tensor)
        return predictions.numpy().ravel().astype(float)

    def save(self, path: str | Path | None = None) -> Path:
        """Lưu tham số và siêu tham số thành checkpoint .pt."""
        if not self._is_fitted:
            raise RuntimeError("Mô hình chưa fit(). Gọi fit() trước save().")
        save_path = Path(path) if path is not None else self.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "model_type": "MLPTorchManualModel",
            "input_size": self._input_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weights": self.weights_,
            "biases": self.biases_,
            "history": self.history_,
        }
        torch.save(checkpoint, save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> MLPTorchManualModel:
        """Nạp mô hình đã lưu."""
        # Checkpoint chỉ chứa tensor và kiểu dữ liệu nguyên thủy nên nạp
        # an toàn với weights_only=True.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint không phải dict, không nạp được mô hình.")
        if checkpoint.get("model_type") != "MLPTorchManualModel":
            raise ValueError(
                "Checkpoint không phải của MLPTorchManualModel "
                f"(model_type={checkpoint.get('model_type')!r})."
            )
        required_keys = {
            "input_size",
            "hidden_layer_sizes",
            "learning_rate",
            "epochs",
            "batch_size",
            "random_state",
            "weights",
            "biases",
        }
        missing_keys = required_keys - set(checkpoint)
        if missing_keys:
            raise ValueError(f"Checkpoint thiếu khóa: {sorted(missing_keys)}.")

        model = cls(
            hidden_layer_sizes=tuple(checkpoint["hidden_layer_sizes"]),
            learning_rate=checkpoint["learning_rate"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            random_state=checkpoint["random_state"],
            beta1=checkpoint.get("beta1", 0.9),
            beta2=checkpoint.get("beta2", 0.999),
            epsilon=checkpoint.get("epsilon", 1e-8),
            save_path=path,
        )
        model.build_model(input_size=checkpoint["input_size"])

        # Kiểm tra hình dạng từng tham số khớp với kiến trúc đã khai báo.
        layer_sizes = (
            checkpoint["input_size"],
            *checkpoint["hidden_layer_sizes"],
            1,
        )
        expected_layer_count = len(layer_sizes) - 1
        if (
            len(checkpoint["weights"]) != expected_layer_count
            or len(checkpoint["biases"]) != expected_layer_count
        ):
            raise ValueError(
                "Checkpoint có số layer không khớp kiến trúc đã khai báo "
                f"(cần {expected_layer_count} layer)."
            )
        for layer_index, (fan_in, fan_out) in enumerate(pairwise(layer_sizes)):
            weight = checkpoint["weights"][layer_index]
            bias = checkpoint["biases"][layer_index]
            if tuple(weight.shape) != (fan_in, fan_out):
                raise ValueError(
                    f"Trọng số layer {layer_index} có shape {tuple(weight.shape)}, "
                    f"không khớp kiến trúc ({fan_in}, {fan_out})."
                )
            if tuple(bias.shape) != (1, fan_out):
                raise ValueError(
                    f"Bias layer {layer_index} có shape {tuple(bias.shape)}, "
                    f"không khớp kiến trúc (1, {fan_out})."
                )
            model.weights_[layer_index] = weight.to(torch.float32)
            model.weights_[layer_index].requires_grad_(False)
            model.biases_[layer_index] = bias.to(torch.float32)
            model.biases_[layer_index].requires_grad_(False)

        model.history_ = checkpoint.get("history")
        model._is_fitted = True
        return model
