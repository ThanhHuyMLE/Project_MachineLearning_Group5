"""Lớp cơ sở cho các mô hình Hồi quy.

Mỗi mô hình tuân theo cùng mẫu 4 phương thức của giảng viên:
``__init__`` -> ``build_model`` -> ``fit`` -> ``predict``.
Không gọi ``build_model`` trong ``__init__``. ``predict`` trước ``fit``
phải báo lỗi rõ ràng. ``build_model`` lại sẽ reset trạng thái đã học.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRegressorModel(ABC):
    """Lớp cơ sở nhẹ: chỉ quản lý trạng thái, không chứa logic thuật toán."""

    def __init__(self) -> None:
        self._is_built = False
        self._is_fitted = False
        self.history_ = None

    @abstractmethod
    def build_model(self, input_size=None):
        """Tạo estimator/kiến trúc và reset trạng thái đã học.

        ``input_size`` là số chiều của đầu vào. Tham số này bắt buộc
        với mô hình mà kiến trúc phụ thuộc vào số chiều input (ví dụ
        MLP viết tay: số cột của ma trận trọng số tầng đầu tiên bằng
        ``input_size``), và không bắt buộc với mô hình khác (ví dụ
        hồi quy tuyến tính suy ra số chiều từ dữ liệu lúc fit).
        """

    @abstractmethod
    def fit(self, X_train, y_train):
        """Huấn luyện mô hình trên tập train."""

    @abstractmethod
    def predict(self, X):
        """Trả giá trị dự đoán cho X."""

    def _require_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Chưa build_model(). Gọi build_model() trước fit().")

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Mô hình chưa fit(). Gọi fit() trước predict().")

    @staticmethod
    def _as_array(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{name} rỗng.")
        return arr
