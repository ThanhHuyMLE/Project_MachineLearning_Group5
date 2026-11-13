"""Mô hình Decision Tree Regression (Sinh viên 4).

TODO (Sinh viên 4): điền phần thân của ``build_model``, ``fit`` và
``predict`` theo hướng dẫn trong docstring từng phương thức, theo mẫu
của ``LinearRegressorModel`` trong ``linear_regression.py``.
Xem docs/algorithms/decision_tree.md.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from house_price.models.base import BaseRegressorModel


class DecisionTreeRegressorModel(BaseRegressorModel):
    """Mô hình cây quyết định Hồi quy theo mẫu 4 phương thức."""

    def __init__(self, max_depth: int | None = None, random_state: int = 42) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def build_model(self):
        """Tạo estimator cây quyết định và reset trạng thái.

        TODO: tạo ``sklearn.tree.DecisionTreeRegressor`` với
        ``max_depth=self.max_depth``, ``random_state=self.random_state``;
        gán vào ``self.model``, đặt ``self._is_built = True``,
        ``self._is_fitted = False``.
        """
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        return self

    def fit(self, X_train, y_train):
        """Xây cây bằng cách chia đệ quy trên tập train.

        TODO: gọi ``self._require_built()``, fit ``self.model``,
        đặt ``self._is_fitted = True`` và trả về ``self``.
        """
        self._require_built()
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Đi từ gốc xuống lá và trả giá trị của lá.

        TODO: gọi ``self._require_fitted()`` rồi trả ``self.model.predict(X)``.
        """
        self._require_fitted()
        return self.model.predict(X)
