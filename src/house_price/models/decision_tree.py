"""Mô hình Decision Tree Regression viết tay (CART) (Stage 04).

Thuật toán: cây hồi quy CART, chỉ dùng NumPy, không import sklearn.

Tiêu chí chia: tại mỗi nút, xét từng đặc trưng ứng viên và từng ngưỡng
chia, chọn cách chia làm giảm phương sai có trọng số nhiều nhất, tức là
tối thiểu hoá MSE có trọng số của hai nút con::

    MSE_có_trọng_số = (n_trái * MSE_trái + n_phải * MSE_phải) / n_nút

tương đương với tối đa hoá mức giảm phương sai (variance reduction).
Ngưỡng chia là trung điểm giữa hai giá trị riêng biệt liên tiếp của
đặc trưng trong nút. Quy ước ngưỡng: mẫu có giá trị đặc trưng đúng
bằng ngưỡng thì đi nhánh trái (``x <= threshold`` đi trái,
``x > threshold`` đi phải); quy ước này áp dụng giống nhau ở cả
``fit`` và ``predict``. Giá trị dự đoán tại lá là trung bình mục tiêu
của các mẫu trong lá.

Chiến lược tìm kiếm và độ phức tạp: với mỗi đặc trưng, sắp xếp các
giá trị (O(n log n)) rồi dùng tổng tích lũy (cumulative sum) của ``y``
và ``y**2`` để tính SSE của mọi vị trí chia ứng viên trong O(n), nên
mỗi nút tốn O(p * n log n) với p là số đặc trưng ứng viên và n là số
mẫu trong nút. Không dùng tìm kiếm exhaustive naive O(p * n^2) và
không giới hạn ngưỡng theo phân vị: mọi ngưỡng giữa hai giá trị riêng
biệt liên tiếp đều được xét, vì tổng tích lũy đã làm việc này đủ rẻ.
Trên dữ liệu thật (~15.000 dòng x 88 đặc trưng) một cây đầy đủ fit
trong vài giây (xem thời gian đo trong báo cáo nghiệm thu, không đo
ở đây vì docstring không chạy code).

Quy tắc dừng: dừng chia khi đạt ``max_depth``, số mẫu trong nút nhỏ
hơn ``min_samples_split``, mọi cách chia đều vi phạm ``min_samples_leaf``
(ít nhất một phía có ít mẫu hơn giới hạn), hoặc không còn cách chia
nào giảm MSE quá ``min_impurity_decrease`` (mặc định 0.0, tức là dừng
khi mục tiêu hằng số hoặc mọi đặc trưng đều hằng số trong nút).

Tính xác định: thứ tự xét đặc trưng và ngưỡng là cố định (đặc trưng
theo chỉ số tăng dần, ngưỡng theo giá trị tăng dần); khi hoà MSE thì
cách chia đầu tiên tốt nhất được giữ (so sánh nghiêm ngặt ``<``).
Khi ``max_features`` yêu cầu lấy mẫu đặc trưng, việc lấy mẫu dùng
``numpy.random.default_rng(self.random_state)`` được tạo lại ở đầu
mỗi lần ``fit``, nên cùng seed cho cùng cây.

Cấu trúc cây lưu trong ``self.tree_`` (dict lồng nhau) để có thể
introspect trong test và để Random Forest dùng lại cây này: nút lá
có ``is_leaf=True`` và ``value``; nút trong có ``feature``,
``threshold``, ``left`` và ``right``. Mọi nút đều có ``n_samples``,
``mse`` (phương sai của ``y`` trong nút) và ``depth``.
Xem docs/algorithms/decision_tree.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from house_price.models.base import BaseRegressorModel


class DecisionTreeRegressorModel(BaseRegressorModel):
    """Mô hình cây quyết định hồi quy theo mẫu 4 phương thức."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        min_impurity_decrease: float = 0.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth phải là None hoặc số nguyên >= 1.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split phải là số nguyên >= 2.")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf phải là số nguyên >= 1.")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease phải >= 0.")
        if isinstance(max_features, str) and max_features != "sqrt":
            raise ValueError("max_features dạng chuỗi chỉ nhận 'sqrt'.")
        if isinstance(max_features, int) and max_features < 1:
            raise ValueError("max_features dạng int phải >= 1.")
        if isinstance(max_features, float) and not 0.0 < max_features <= 1.0:
            raise ValueError("max_features dạng float phải nằm trong (0, 1].")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.model = None
        self.tree_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def build_model(self):
        """Tạo trạng thái cây rỗng và reset trạng thái đã học."""
        self.model = None
        self.tree_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self._is_built = True
        self._is_fitted = False
        self.history_ = None
        return self

    def fit(self, X_train, y_train):
        """Xây cây bằng cách chia đệ quy trên tập train (cài đặt lặp)."""
        self._require_built()
        X = self._as_array(X_train, "X_train")
        if X.ndim != 2:
            raise ValueError("X_train phải là ma trận 2 chiều (n_samples, n_features).")
        y = self._as_array(y_train, "y_train")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y_train phải là vector 1 chiều.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train và y_train phải có cùng số dòng.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X_train và y_train không được chứa NaN hay inf.")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        importances = np.zeros(n_features, dtype=float)

        root: dict = {}
        # Ngăn xếp thay cho đệ quy: cây không giới hạn độ sâu có thể
        # sâu hơn giới hạn đệ quy mặc định của Python.
        stack = [(root, np.arange(n_samples), 0)]
        while stack:
            node, indices, depth = stack.pop()
            X_node = X[indices]
            y_node = y[indices]
            n_node = indices.shape[0]
            mean = float(y_node.mean())
            mse = float(np.mean((y_node - mean) ** 2))
            node.update(
                {
                    "is_leaf": True,
                    "value": mean,
                    "n_samples": int(n_node),
                    "mse": mse,
                    "depth": int(depth),
                    "feature": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                }
            )
            if n_node < self.min_samples_split:
                continue
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if mse <= 0.0:
                continue
            best = self._best_split(X_node, y_node, rng)
            if best is None:
                continue
            feature, threshold, weighted_mse = best
            improvement = mse - weighted_mse
            if improvement <= self.min_impurity_decrease:
                continue
            left_mask = X_node[:, feature] <= threshold
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
            if left_indices.shape[0] == 0 or right_indices.shape[0] == 0:
                continue
            importances[feature] += n_node * improvement
            node["is_leaf"] = False
            node["feature"] = int(feature)
            node["threshold"] = float(threshold)
            node["left"] = {}
            node["right"] = {}
            stack.append((node["right"], right_indices, depth + 1))
            stack.append((node["left"], left_indices, depth + 1))

        self.tree_ = root
        self.model = root
        self.n_features_in_ = n_features
        total_importance = importances.sum()
        if total_importance > 0:
            importances = importances / total_importance
        self.feature_importances_ = importances
        self._is_fitted = True
        return self

    def predict(self, X):
        """Đi từ gốc xuống lá và trả giá trị của lá.

        Tại mỗi nút trong, mẫu có ``x <= threshold`` đi nhánh trái,
        ``x > threshold`` đi nhánh phải (cùng quy ước ngưỡng như ở
        ``fit``).
        """
        self._require_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X phải là ma trận 2 chiều (n_samples, n_features).")
        if X_arr.shape[0] == 0:
            raise ValueError("X rỗng.")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X có {X_arr.shape[1]} đặc trưng, mô hình cần {self.n_features_in_}.")
        predictions = np.empty(X_arr.shape[0], dtype=float)
        for i, row in enumerate(X_arr):
            node = self.tree_
            while not node["is_leaf"]:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions[i] = node["value"]
        return predictions

    def to_state(self, feature_names=None) -> dict[str, object]:
        """Trả trạng thái đã học ở dạng chỉ gồm kiểu dữ liệu JSON."""
        self._require_fitted()
        nodes = [self.tree_]
        rows = []
        cursor = 0
        while cursor < len(nodes):
            node = nodes[cursor]
            left_index = -1
            right_index = -1
            if not node["is_leaf"]:
                left_index = len(nodes)
                nodes.append(node["left"])
                right_index = len(nodes)
                nodes.append(node["right"])
            rows.append(
                [
                    bool(node["is_leaf"]),
                    float(node["value"]),
                    int(node["n_samples"]),
                    float(node["mse"]),
                    int(node["depth"]),
                    -1 if node["feature"] is None else int(node["feature"]),
                    None if node["threshold"] is None else float(node["threshold"]),
                    left_index,
                    right_index,
                ]
            )
            cursor += 1

        if feature_names is not None and len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"Có {len(feature_names)} tên đặc trưng nhưng cây có {self.n_features_in_} đặc trưng."
            )
        return {
            "format_version": 1,
            "model_type": "DecisionTreeRegressorModel",
            "config": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "min_impurity_decrease": self.min_impurity_decrease,
                "random_state": self.random_state,
            },
            "n_features_in": self.n_features_in_,
            "feature_names": None if feature_names is None else list(feature_names),
            "feature_importances": self.feature_importances_.tolist(),
            "node_fields": [
                "is_leaf",
                "value",
                "n_samples",
                "mse",
                "depth",
                "feature",
                "threshold",
                "left",
                "right",
            ],
            "nodes": rows,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> DecisionTreeRegressorModel:
        """Khôi phục cây từ trạng thái JSON do :meth:`to_state` tạo."""
        if state.get("model_type") != "DecisionTreeRegressorModel":
            raise ValueError("JSON không chứa DecisionTreeRegressorModel.")
        config = state["config"]
        model = cls(**config).build_model()
        rows = state["nodes"]
        if not rows:
            raise ValueError("JSON cây quyết định không có node.")

        nodes = []
        for row in rows:
            nodes.append(
                {
                    "is_leaf": bool(row[0]),
                    "value": float(row[1]),
                    "n_samples": int(row[2]),
                    "mse": float(row[3]),
                    "depth": int(row[4]),
                    "feature": None if int(row[5]) < 0 else int(row[5]),
                    "threshold": None if row[6] is None else float(row[6]),
                    "left": None,
                    "right": None,
                }
            )
        for node, row in zip(nodes, rows):
            if not node["is_leaf"]:
                node["left"] = nodes[int(row[7])]
                node["right"] = nodes[int(row[8])]

        model.tree_ = nodes[0]
        model.model = model.tree_
        model.n_features_in_ = int(state["n_features_in"])
        model.feature_importances_ = np.asarray(state["feature_importances"], dtype=float)
        model._is_fitted = True
        return model

    def export_json(self, path: str | Path, feature_names=None) -> Path:
        """Xuất toàn bộ cấu trúc cây để có thể nạp lại và dự đoán."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_state(feature_names), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> DecisionTreeRegressorModel:
        """Nạp cây đã xuất bằng :meth:`export_json`."""
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))

    def _candidate_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """Chỉ số đặc trưng ứng viên tại một nút (có thể lấy mẫu)."""
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, str):  # "sqrt"
            count = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            count = min(self.max_features, n_features)
        else:  # float trong (0, 1]
            count = max(1, int(self.max_features * n_features))
        if count >= n_features:
            return np.arange(n_features)
        # Sắp xếp lại theo chỉ số để thứ tự xét luôn ổn định.
        return np.sort(rng.choice(n_features, size=count, replace=False))

    def _best_split(
        self, X_node: np.ndarray, y_node: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, float, float] | None:
        """Cách chia tốt nhất trong nút, hoặc None nếu không chia được.

        Với mỗi đặc trưng: sắp xếp giá trị rồi dùng tổng tích lũy để
        tính SSE của mọi vị trí chia ứng viên trong O(n) sau bước sort.
        """
        n_node, n_features = X_node.shape
        best_feature = None
        best_threshold = None
        best_weighted_mse = float("inf")
        for feature in self._candidate_features(n_features, rng):
            values = X_node[:, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_y = y_node[order]
            # Vị trí chia ứng viên: ngay sau mỗi giá trị riêng biệt.
            change = sorted_values[1:] != sorted_values[:-1]
            positions = np.flatnonzero(change)
            if positions.shape[0] == 0:
                continue
            left_counts = positions + 1
            valid = (left_counts >= self.min_samples_leaf) & (
                (n_node - left_counts) >= self.min_samples_leaf
            )
            if not valid.any():
                continue
            cum_y = np.cumsum(sorted_y)
            cum_y2 = np.cumsum(sorted_y * sorted_y)
            total_y = cum_y[-1]
            total_y2 = cum_y2[-1]
            left_sums = cum_y[positions]
            left_sums2 = cum_y2[positions]
            right_sums = total_y - left_sums
            right_sums2 = total_y2 - left_sums2
            right_counts = n_node - left_counts
            left_sse = np.maximum(left_sums2 - left_sums * left_sums / left_counts, 0.0)
            right_sse = np.maximum(right_sums2 - right_sums * right_sums / right_counts, 0.0)
            weighted_mse = (left_sse + right_sse) / n_node
            weighted_mse = np.where(valid, weighted_mse, np.inf)
            # argmin trả về vị trí đầu tiên nhỏ nhất: tie-breaking
            # ổn định theo ngưỡng tăng dần.
            local_best = int(np.argmin(weighted_mse))
            # So sánh nghiêm ngặt: hoà MSE thì giữ đặc trưng xét trước.
            if weighted_mse[local_best] < best_weighted_mse:
                best_weighted_mse = float(weighted_mse[local_best])
                best_feature = int(feature)
                split_at = int(positions[local_best])
                best_threshold = float(
                    (sorted_values[split_at] + sorted_values[split_at + 1]) / 2.0
                )
        if best_feature is None:
            return None
        return best_feature, best_threshold, best_weighted_mse
