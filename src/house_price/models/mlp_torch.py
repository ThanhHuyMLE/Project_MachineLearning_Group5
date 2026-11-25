"""MLP Regression bằng PyTorch cho bài toán giá nhà King County.

Kiến trúc mặc định:
Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)

Thực hiện theo 6 bước huấn luyện DNN bằng PyTorch trong slide của cô giáo:
1. Chuẩn bị dữ liệu (Dataset và DataLoader).
2. Định nghĩa mô hình (kiến trúc mạng nơ-ron).
3. Chọn hàm loss (MSE) và optimizer (Adam).
4. Viết training loop.
5. Validation (theo dõi val_loss mỗi epoch).
6. Lưu và nạp lại mô hình (state_dict).
"""

from __future__ import annotations

import json
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from house_price.models.base import BaseRegressorModel


class _MLPNetwork(nn.Module):
    """Mạng MLP 32-16-1: hai tầng ẩn ReLU, tầng cuối Linear."""

    def __init__(self, input_size: int, hidden_layer_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layer_sizes = (input_size, *hidden_layer_sizes, 1)
        layers: list[nn.Module] = []
        for fan_in, fan_out in pairwise(layer_sizes):
            linear = nn.Linear(fan_in, fan_out)
            # He initialization dạng chuẩn (normal) cho mọi tầng; bias 0.
            nn.init.kaiming_normal_(linear.weight, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if fan_out != 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTorchRegressorModel(BaseRegressorModel):
    """MLP hồi quy bằng PyTorch, lưu state_dict ngay sau khi fit."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        save_path: str | Path = "outputs/mlp_torch_parameters.pt",
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

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.save_path = Path(save_path)

        self.network: _MLPNetwork | None = None
        self._input_size: int | None = None

    def build_model(self, input_size: int):
        """Tạo mạng nơ-ron (bước 2) và reset trạng thái đã học."""
        if input_size <= 0:
            raise ValueError("input_size phải lớn hơn 0.")

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.network = _MLPNetwork(input_size, self.hidden_layer_sizes)
        self._input_size = input_size
        self.history_ = {"loss": [], "val_loss": []}
        self._is_built = True
        self._is_fitted = False
        return self

    @torch.no_grad()
    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính MSE toàn bộ tập bằng forward, không cập nhật trọng số."""
        self.network.eval()
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        predictions = self.network(X_tensor)
        return float(nn.functional.mse_loss(predictions, y_tensor))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Huấn luyện MLP bằng PyTorch theo mini-batch, theo dõi val_loss."""
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

        # Bước 1: chuẩn bị dữ liệu (Dataset và DataLoader).
        # DataLoader xáo trộn thứ tự mẫu mỗi epoch bằng generator có seed
        # cố định, nên kết quả lặp lại được với cùng random_state.
        torch.manual_seed(self.random_state)
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

        # Bước 3: hàm loss MSE và optimizer Adam.
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.history_ = {"loss": [], "val_loss": []} if has_val else {"loss": []}

        # Bước 4: training loop.
        for _epoch in range(self.epochs):
            self.network.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.network(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            # Bước 5: validation (chỉ forward, không cập nhật trọng số).
            self.history_["loss"].append(self._mse(X, y))
            if has_val:
                self.history_["val_loss"].append(self._mse(X_val_arr, y_val_arr))

        self._is_fitted = True

        # Bước 6: lưu mô hình (state_dict) ngay sau khi fit hoàn tất.
        self.save()
        return self

    def predict(self, X):
        """Dự đoán một giá trị cho mỗi mẫu đầu vào."""
        self._require_fitted()

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim != 2:
            raise ValueError("X phải là ma trận hai chiều.")
        if X_array.shape[1] != self._input_size:
            raise ValueError(
                f"Mạng cần {self._input_size} đặc trưng, nhưng X có {X_array.shape[1]} đặc trưng."
            )
        if not np.isfinite(X_array).all():
            raise ValueError("X không được chứa NaN hoặc vô cực.")

        with torch.no_grad():
            self.network.eval()
            X_tensor = torch.from_numpy(X_array.astype(np.float32))
            predictions = self.network(X_tensor).numpy()
        return predictions.ravel().astype(float)

    def save(self, path: str | Path | None = None) -> Path:
        """Lưu state_dict và siêu tham số (bước 6: Save)."""
        save_path = Path(path) if path is not None else self.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "model_type": "MLPTorchRegressorModel",
            "input_size": self._input_size,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "state_dict": self.network.state_dict(),
            "history": self.history_,
        }
        torch.save(checkpoint, save_path)
        return save_path

    def export_json(self, path: str | Path, feature_names=None) -> Path:
        """Xuất kiến trúc, trọng số và bias sang JSON có thể nạp lại."""
        self._require_fitted()
        if feature_names is not None and len(feature_names) != self._input_size:
            raise ValueError(
                f"Có {len(feature_names)} tên đặc trưng nhưng MLP có {self._input_size} đầu vào."
            )
        payload = {
            "format_version": 1,
            "model_type": "MLPTorchRegressorModel",
            "input_size": self._input_size,
            "feature_names": None if feature_names is None else list(feature_names),
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "state_dict": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in self.network.state_dict().items()
            },
            "history": self.history_,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> MLPTorchRegressorModel:
        """Nạp mô hình đã lưu (bước 6: Load)."""
        # Checkpoint do chính project lưu, không phải file ngoài không rõ nguồn.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            hidden_layer_sizes=tuple(checkpoint["hidden_layer_sizes"]),
            learning_rate=checkpoint["learning_rate"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            random_state=checkpoint["random_state"],
            save_path=path,
        )
        model.build_model(input_size=checkpoint["input_size"])
        model.network.load_state_dict(checkpoint["state_dict"])
        model.history_ = checkpoint.get("history")
        model._is_fitted = True
        return model

    @classmethod
    def load_json(cls, path: str | Path) -> MLPTorchRegressorModel:
        """Nạp MLP từ JSON do :meth:`export_json` tạo."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "MLPTorchRegressorModel":
            raise ValueError("JSON không chứa MLPTorchRegressorModel.")
        model = cls(
            hidden_layer_sizes=tuple(payload["hidden_layer_sizes"]),
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            random_state=payload["random_state"],
            save_path=path,
        )
        model.build_model(input_size=payload["input_size"])
        template = model.network.state_dict()
        state_dict = {
            name: torch.tensor(values, dtype=template[name].dtype)
            for name, values in payload["state_dict"].items()
        }
        model.network.load_state_dict(state_dict)
        model.history_ = payload.get("history")
        model._is_fitted = True
        return model
