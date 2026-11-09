"""Mô hình Linear Regression (Sinh viên 4).

Đây là mô hình DUY NHẤT được cài đặt đầy đủ trong Phase 1, làm mẫu
tham chiếu cho 3 class còn lại: cùng mẫu __init__/build_model/fit/predict,
cùng cách báo lỗi và lưu trạng thái. Xem docs/algorithms/linear_regression.md.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from house_price.models.base import BaseRegressorModel


class LinearRegressorModel(BaseRegressorModel):
    """Mô hình Hồi quy tuyến tính: y_hat = X beta + b."""

    def __init__(self) -> None:
        super().__init__()
        self.model: LinearRegression | None = None

    def build_model(self):
        """Tạo estimator LinearRegression và reset trạng thái."""
        self.model = LinearRegression()
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        return self

    def fit(self, X_train, y_train):
        """Giải bài toán bình phương tối thiểu (OLS) trên tập train."""
        self._require_built()
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Trả giá dự đoán; báo lỗi nếu chưa fit."""
        self._require_fitted()
        return self.model.predict(X)

    @property
    def coef_(self):
        """Vector hệ số beta (chỉ có sau fit)."""
        self._require_fitted()
        return self.model.coef_

    @property
    def intercept_(self):
        """Hệ số chặn b (chỉ có sau fit)."""
        self._require_fitted()
        return self.model.intercept_
