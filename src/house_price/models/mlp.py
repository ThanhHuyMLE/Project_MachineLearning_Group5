"""Mô hình MLP Regression (Sinh viên 2).

TODO (Sinh viên 2): điền phần thân của ``build_model``, ``fit`` và
``predict`` theo hướng dẫn trong docstring từng phương thức.
Kiến trúc mặc định theo đề bài: Dense(32, relu) -> Dense(16, relu)
-> Dense(1, linear), optimizer Adam, loss MSE, epochs 100, batch 32.
Backend mặc định: sklearn ``MLPRegressor`` (xem docs/adr/004).
"""

from __future__ import annotations

from house_price.models.base import BaseRegressorModel


class MLPRegressorModel(BaseRegressorModel):
    """Mô hình MLP Hồi quy theo mẫu __init__/build_model/fit/predict."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        max_iter: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def build_model(self):
        """Tạo estimator MLP và reset trạng thái.

        TODO: tạo ``sklearn.neural_network.MLPRegressor`` với
        ``hidden_layer_sizes=self.hidden_layer_sizes``,
        ``activation=self.activation``, ``solver=self.solver``,
        ``learning_rate_init=self.learning_rate_init``,
        ``max_iter=self.max_iter``, ``batch_size=self.batch_size``,
        ``random_state=self.random_state``; gán vào ``self.model``,
        đặt ``self._is_built = True`` và ``self._is_fitted = False``.
        """
        raise NotImplementedError("TODO (Sinh viên 2): cài đặt build_model() cho MLP.")

    def fit(self, X_train, y_train):
        """Huấn luyện MLP trên tập train.

        TODO: gọi ``self._require_built()``, fit ``self.model`` trên
        (X_train, y_train), lưu lịch sử loss vào ``self.history_``
        (dạng ``{"loss": [...]}``, thêm ``val_loss`` nếu có tập validation),
        đặt ``self._is_fitted = True`` và trả về ``self``.
        """
        raise NotImplementedError("TODO (Sinh viên 2): cài đặt fit() cho MLP.")

    def predict(self, X):
        """Dự đoán giá nhà cho X.

        TODO: gọi ``self._require_fitted()`` rồi trả ``self.model.predict(X)``.
        """
        raise NotImplementedError("TODO (Sinh viên 2): cài đặt predict() cho MLP.")
