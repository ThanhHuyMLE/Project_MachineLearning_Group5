"""Mô hình Random Forest Regression viết tay (Stage 04).

Thuật toán (bagging + lấy mẫu đặc trưng, xem docs/algorithms/random_forest.md):

- Mỗi cây trong rừng là một ``DecisionTreeRegressorModel`` viết tay
  (cây CART, không dùng sklearn), học trên một mẫu bootstrap riêng:
  lấy mẫu có hoàn lại ``n`` dòng từ tập train, nên mỗi cây thấy một
  tập dữ liệu hơi khác nhau và có khoảng 1/3 số dòng không xuất hiện
  trong mẫu của nó (out-of-bag).
- Tại mỗi lần chia, cây chỉ xét một tập con ngẫu nhiên các đặc trưng
  (do ``max_features`` quy định, mặc định ``"sqrt"``), làm các cây
  càng khác nhau hơn.
- Dự đoán cuối là trung bình cộng dự đoán của tất cả các cây::

      y_hat(x) = (1 / T) * sum_t y_hat_t(x)

Vì sao trung bình nhiều cây giảm phương sai: mỗi cây đơn lẻ có
phương sai cao (nhạy với dữ liệu train), nhưng trung bình của ``T``
cây có phương sai xấp xỉ ``rho * sigma^2 + (1 - rho) * sigma^2 / T``,
trong đó ``sigma^2`` là phương sai một cây và ``rho`` là tương quan
trung bình giữa các cây. Bootstrap và lấy mẫu đặc trưng làm giảm
``rho``, nên phương sai chung giảm mà độ chệch (bias) hầu như không
tăng, vì mỗi cây vẫn học đủ sâu trên dữ liệu của nó.

Tham số:

- ``n_estimators``: số cây trong rừng (mặc định 100). Nhiều cây hơn
  thì lỗi giảm dần rồi bão hòa, đổi lại thời gian huấn luyện tăng
  tuyến tính theo số cây.
- ``max_depth``, ``min_samples_split``, ``min_samples_leaf``: giới hạn
  của từng cây, truyền thẳng sang ``DecisionTreeRegressorModel``.
- ``max_features``: số đặc trưng xét tại mỗi lần chia của từng cây.
  ``None`` nghĩa là xét mọi đặc trưng; ``"sqrt"`` nghĩa là mỗi lần
  chia lấy mẫu ngẫu nhiên ``sqrt(n_features)`` đặc trưng; ``int``
  hoặc ``float`` tương tự (theo quy ước của cây quyết định).
- ``random_state``: hạt giống điều khiển toàn bộ tính ngẫu nhiên của
  rừng (bootstrap và hạt giống riêng của từng cây). Cùng seed cho
  cùng một rừng; khác seed cho rừng khác nhau.

Cách sinh số ngẫu nhiên (tái lập được và khác nhau giữa các cây):
``np.random.SeedSequence(random_state)`` được ``spawn`` thành
``n_estimators`` dãy con, mỗi cây nhận một dãy con riêng. Từ dãy con
đó tạo một ``np.random.Generator``: một phần rút chỉ số bootstrap,
phần còn lại rút hạt giống ``random_state`` truyền cho cây. Nhờ vậy
hai lần ``fit`` với cùng seed cho cùng rừng, và thứ tự cây không
phụ thuộc vào trạng thái RNG toàn cục.

Giới hạn:

- Chậm và tốn bộ nhớ hơn cây đơn lẻ khoảng ``n_estimators`` lần;
  fit 100 cây trên 15k dòng x 88 đặc trưng có thể mất nhiều phút
  với cây CART viết tay bằng Python thuần.
- Dự đoán bị chặn trong khoảng giá trị lá của các cây nên không
  ngoại suy tốt ngoài miền giá trị train.
- Khó giải thích từng dự đoán riêng lẻ (mô hình hộp đen tập hợp).
"""

from __future__ import annotations

import numpy as np

from house_price.models.base import BaseRegressorModel
from house_price.models.decision_tree import DecisionTreeRegressorModel


class RandomForestRegressorModel(BaseRegressorModel):
    """Mô hình rừng ngẫu nhiên Hồi quy: trung bình nhiều cây CART viết tay."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int):
            raise TypeError("n_estimators phải là số nguyên.")
        if n_estimators < 1:
            raise ValueError("n_estimators phải >= 1.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state phải là số nguyên hoặc None.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeRegressorModel] = []
        self.n_features_in_: int | None = None

    def build_model(self):
        """Tạo rừng rỗng và reset trạng thái đã học."""
        self.trees_ = []
        self.n_features_in_ = None
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        return self

    def fit(self, X_train, y_train):
        """Huấn luyện ``n_estimators`` cây, mỗi cây trên một mẫu bootstrap.

        Trả về ``self`` để gọi nối tiếp được theo mẫu chung.
        """
        self._require_built()
        X = self._as_array(X_train, "X_train")
        y = self._as_array(y_train, "y_train")
        if X.ndim != 2:
            raise ValueError("X_train phải là mảng 2 chiều (n_samples, n_features).")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y_train phải là mảng 1 chiều (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train và y_train phải có cùng số dòng.")

        n_samples, n_features = X.shape
        # Một dãy con SeedSequence riêng cho mỗi cây: bootstrap và hạt
        # giống của cây đều rút từ dãy con đó, vừa tái lập được vừa
        # khác nhau giữa các cây, không đụng RNG toàn cục.
        seed_sequence = np.random.SeedSequence(self.random_state)
        child_sequences = seed_sequence.spawn(self.n_estimators)

        trees: list[DecisionTreeRegressorModel] = []
        for child_sequence in child_sequences:
            rng = np.random.default_rng(child_sequence)
            # Mẫu bootstrap: n chỉ số lấy có hoàn lại từ [0, n_samples).
            bootstrap_indices = rng.integers(0, n_samples, size=n_samples)
            tree_seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressorModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.build_model()
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            trees.append(tree)

        self.trees_ = trees
        self.n_features_in_ = n_features
        self._is_fitted = True
        return self

    def predict(self, X):
        """Trả trung bình cộng dự đoán của các cây; báo lỗi nếu chưa fit."""
        self._require_fitted()
        X = self._as_array(X, "X")
        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều (n_samples, n_features).")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X có {X.shape[1]} đặc trưng, mô hình fit trên {self.n_features_in_} đặc trưng."
            )
        predictions = np.column_stack(
            [np.asarray(tree.predict(X), dtype=float) for tree in self.trees_]
        )
        return predictions.mean(axis=1)
