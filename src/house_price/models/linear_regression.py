"""Mô hình Linear Regression (Stage 04) - cài đặt tay bằng NumPy.

Không dùng bất kỳ estimator có sẵn nào của thư viện học máy: hệ số
được học bằng batch gradient descent viết tay (xem
docs/adr/007-mo-hinh-numpy.md). NumPy chỉ làm phép tính mảng.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from house_price.models.base import BaseRegressorModel


class LinearRegressorModel(BaseRegressorModel):
    """Mô hình Hồi quy tuyến tính: y_hat = X beta + b.

    Công thức huấn luyện (batch gradient descent trên toàn bộ tập train):

    - Ma trận thiết kế mở rộng hệ số chặn: ``X_aug = [1 | X]`` và
      vector tham số gộp ``beta_aug = [b, beta]``.
    - Hàm mất mát MSE: ``L = (1/n) ||X_aug beta_aug - y||^2``.
    - Gradient: ``dL/dbeta_aug = (2/n) X_aug^T (X_aug beta_aug - y)``.
    - Cập nhật mỗi epoch: ``beta_aug <- beta_aug - lr * dL/dbeta_aug``,
      với ``lr`` là ``learning_rate``.

    Quy tắc dừng: dừng sớm khi thay đổi TƯƠNG ĐỐI của loss giữa hai
    epoch liên tiếp nhỏ hơn ``tol``::

        |L_t - L_{t-1}| / max(L_{t-1}, 1e-12) < tol

    Dùng thay đổi tương đối (không phải tuyệt đối) vì y trong pipeline
    là giá nhà thô (đơn vị USD, MSE cỡ 1e10) nên một ngưỡng tuyệt đối
    không có ý nghĩa chung cho mọi thang đo của y. Nếu không dừng sớm
    thì chạy đủ ``max_epochs`` epoch và ``converged_`` là False.

    Giả định về đầu vào: X đã được chuẩn hóa (StandardScaler trong
    pipeline tiền xử lý; đặc trưng one-hot là 0/1), nên các cột của
    ``X_aug`` có thang đo cỡ 1 và learning rate mặc định 0.1 là ổn
    định. Tốc độ hội tụ của gradient descent chỉ phụ thuộc vào X
    (qua trị riêng của ``X_aug^T X_aug``), không phụ thuộc thang đo
    của y. Nếu đưa X chưa chuẩn hóa với thang đo lớn thì thuật toán
    có thể phân kỳ; khi đó ``fit`` báo lỗi rõ ràng thay vì trả về hệ
    số vô hạn hoặc NaN.

    Sau khi fit, ``history_`` là ``{"loss": [...]}`` với một giá trị
    MSE sau mỗi epoch, ``n_iter_`` là số epoch đã chạy và
    ``converged_`` cho biết có dừng sớm hay không.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError(f"learning_rate phải > 0, nhận được {learning_rate}.")
        if max_epochs < 1:
            raise ValueError(f"max_epochs phải >= 1, nhận được {max_epochs}.")
        if tol <= 0:
            raise ValueError(f"tol phải > 0, nhận được {tol}.")
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.tol = float(tol)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self.n_iter_ = 0
        self.converged_ = False

    def build_model(self):
        """Khởi tạo trạng thái, chưa có hệ số; gọi lại sẽ reset trạng thái."""

        self._coef = None
        self._intercept = None
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        self.n_iter_ = 0
        self.converged_ = False
        return self

    def fit(self, X_train, y_train):
        """Học hệ số bằng batch gradient descent viết tay trên tập train.

        Khởi tạo ``beta_aug`` bằng 0 rồi lặp: tính gradient MSE trên
        toàn bộ tập train và cập nhật ``beta_aug``. Dừng sớm theo quy
        tắc thay đổi tương đối của loss trong docstring của class.
        """
        self._require_built()
        X = self._as_array(X_train, "X_train")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X_train phải là ma trận 2 chiều, nhận được {X.ndim} chiều.")
        y = self._as_array(y_train, "y_train").ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_train có {X.shape[0]} dòng nhưng y_train có {y.shape[0]} giá trị.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X_train/y_train chứa giá trị không hữu hạn (NaN/inf).")
        n_samples = X.shape[0]
        design = np.column_stack([np.ones(n_samples), X])
        beta = np.zeros(design.shape[1])
        losses: list[float] = []
        converged = False
        for _ in range(self.max_epochs):
            residual = design @ beta - y
            gradient = (2.0 / n_samples) * (design.T @ residual)
            beta = beta - self.learning_rate * gradient
            loss = float(np.mean((design @ beta - y) ** 2))
            if not np.isfinite(loss):
                raise RuntimeError(
                    "Gradient descent phân kỳ (loss không hữu hạn). "
                    "Giảm learning_rate hoặc chuẩn hóa X trước khi fit."
                )
            if losses:
                previous_loss = losses[-1]
                relative_change = abs(previous_loss - loss) / max(previous_loss, 1e-12)
                if relative_change < self.tol:
                    losses.append(loss)
                    converged = True
                    break
            losses.append(loss)
        self._intercept = float(beta[0])
        self._coef = np.asarray(beta[1:], dtype=float)
        self.history_ = {"loss": losses}
        self.n_iter_ = len(losses)
        self.converged_ = converged
        self._is_fitted = True
        return self

    def predict(self, X):
        """Trả giá dự đoán y_hat = X beta + b; báo lỗi nếu chưa fit."""
        self._require_fitted()
        X = self._as_array(X, "X")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X phải là ma trận 2 chiều, nhận được {X.ndim} chiều.")
        if X.shape[1] != len(self._coef):
            raise ValueError(f"X có {X.shape[1]} đặc trưng, mô hình cần {len(self._coef)} hệ số.")
        return X @ self._coef + self._intercept

    @property
    def coef_(self):
        """Vector hệ số beta (trọng số), chỉ có sau fit."""
        self._require_fitted()
        return self._coef

    @property
    def intercept_(self):
        """Hệ số chặn b (bias), chỉ có sau fit."""
        self._require_fitted()
        return self._intercept

    def export_params(self, path, feature_names=None):
        """Xuất tham số đã học ra file JSON để đọc và đối chiếu.

        Ghi ``{"model": ..., "method": ..., "n_features": ...,
        "intercept": b, "coefficients": [{"feature": ..., "weight": ...}],
        "learning_rate": ..., "n_epochs": ..., "final_loss": ...,
        "converged": ...}``. Không có tên đặc trưng thì dùng ``x0``,
        ``x1``, ...
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(self._coef))]
        if len(feature_names) != len(self._coef):
            raise ValueError(
                f"Có {len(feature_names)} tên đặc trưng nhưng mô hình có {len(self._coef)} hệ số."
            )
        payload = {
            "format_version": 1,
            "model_type": "LinearRegressorModel",
            "model": "LinearRegression",
            "method": (
                "Batch gradient descent viết tay bằng NumPy: "
                "beta <- beta - learning_rate * (2/n) X_aug^T (X_aug beta - y), "
                "X_aug = [1 | X]"
            ),
            "n_features": len(self._coef),
            "intercept": self._intercept,
            "coefficients": [
                {"feature": str(name), "weight": float(weight)}
                for name, weight in zip(feature_names, self._coef)
            ],
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tol": self.tol,
            "n_epochs": self.n_iter_,
            "history": self.history_,
            "final_loss": self.history_["loss"][-1],
            "converged": self.converged_,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_json(self, path, feature_names=None):
        """Tên thống nhất với các mô hình khác; giữ `export_params` để tương thích."""
        return self.export_params(path, feature_names=feature_names)

    @classmethod
    def load_json(cls, path):
        """Nạp hệ số đã học từ JSON do :meth:`export_json` tạo."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "LinearRegressorModel":
            raise ValueError("JSON không chứa LinearRegressorModel.")
        model = cls(
            learning_rate=payload["learning_rate"],
            max_epochs=payload["max_epochs"],
            tol=payload["tol"],
        ).build_model()
        model._coef = np.asarray([item["weight"] for item in payload["coefficients"]], dtype=float)
        model._intercept = float(payload["intercept"])
        model.history_ = payload["history"]
        model.n_iter_ = int(payload["n_epochs"])
        model.converged_ = bool(payload["converged"])
        model._is_fitted = True
        return model
