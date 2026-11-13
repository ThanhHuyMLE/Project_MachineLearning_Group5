"""Mô hình Random Forest Regression (Sinh viên 4).

TODO (Sinh viên 4): điền phần thân của ``build_model``, ``fit`` và
``predict`` theo hướng dẫn trong docstring từng phương thức, theo mẫu
của ``LinearRegressorModel`` trong ``linear_regression.py``.
Xem docs/algorithms/random_forest.md.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from house_price.models.base import BaseRegressorModel


class RandomForestRegressorModel(BaseRegressorModel):
    """Mô hình rừng ngẫu nhiên Hồi quy theo mẫu 4 phương thức."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def build_model(self):
        """Tạo estimator rừng ngẫu nhiên và reset trạng thái.

        TODO: tạo ``sklearn.ensemble.RandomForestRegressor`` với
        ``n_estimators=self.n_estimators``,
        ``random_state=self.random_state``; gán vào ``self.model``,
        đặt ``self._is_built = True``, ``self._is_fitted = False``.
        """
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        return self

    def fit(self, X_train, y_train):
        """Huấn luyện n_estimators cây trên tập train.

        TODO: gọi ``self._require_built()``, fit ``self.model``,
        đặt ``self._is_fitted = True`` và trả về ``self``.
        """
        self._require_built()
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Trả trung bình dự đoán của các cây.

        TODO: gọi ``self._require_fitted()`` rồi trả ``self.model.predict(X)``.
        """
        self._require_fitted()
        return self.model.predict(X)
