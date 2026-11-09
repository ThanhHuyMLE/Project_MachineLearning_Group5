"""Nạp dữ liệu, làm sạch, chia tập và chuẩn hóa (Sinh viên 1).

Quy tắc chống rò rỉ dữ liệu (Data Leakage) áp dụng trong toàn bộ module:
chia train/validation/test TRƯỚC, chỉ fit imputer và scaler trên tập train,
rồi dùng cùng đối tượng đã fit để transform validation, test và dữ liệu mới.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "price"

# Các cột không mang thông tin dự báo, loại khỏi tập huấn luyện.
DROP_COLUMNS = ["Unnamed: 0", "id", "date"]

# 18 đặc trưng đầu vào, theo thứ tự cố định trong hợp đồng dữ liệu.
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Đọc file CSV gốc và trả về DataFrame thô, chưa xử lý gì.

    Kiểm tra file tồn tại và có cột mục tiêu ``price``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dữ liệu thiếu cột mục tiêu '{TARGET_COLUMN}'.")
    return frame


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Loại cột thừa và trả về bản sao sạch, CHƯA bù khuyết hay chuẩn hóa.

    Bù khuyết và chuẩn hóa nằm trong preprocessor (xem ``build_preprocessor``)
    để đảm bảo chỉ fit trên tập train. Hàm này chỉ loại cột và kiểm tra
    đủ 18 cột đặc trưng.
    """
    cleaned = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    missing = [c for c in FEATURE_COLUMNS if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột đặc trưng: {missing}")
    return cleaned


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách DataFrame thành ma trận đặc trưng X và mục tiêu y."""
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].copy()


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Chia dữ liệu thành train/validation/test theo tỷ lệ 70/15/15.

    Hồi quy không có nhãn lớp nên không stratify; chỉ xáo trộn ngẫu nhiên
    với ``random_state`` cố định để kết quả tái lập được.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """Fit imputer (median) + scaler trên tập train, trả về pipeline đã fit.

    Đây là điểm duy nhất được fit tiền xử lý. Mọi tập khác chỉ được transform
    bằng pipeline này, không được fit lại.
    """
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor.fit(X_train)
    return preprocessor


def prepare_data(
    path: str | Path,
) -> dict[str, object]:
    """Chạy trọn tiền xử lý và trả về dict chứa 3 tập đã chuẩn hóa.

    Thứ tự: nạp -> làm sạch -> tách X/y -> chia 3 tập -> fit preprocessor
    trên train -> transform cả 3 tập. Trả về các khóa: ``X_train``,
    ``X_val``, ``X_test``, ``y_train``, ``y_val``, ``y_test``,
    ``preprocessor``, ``feature_columns``.
    """
    frame = clean_dataset(load_raw_dataset(path))
    X, y = split_features_target(frame)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    preprocessor = build_preprocessor(X_train)
    return {
        "X_train": preprocessor.transform(X_train),
        "X_val": preprocessor.transform(X_val),
        "X_test": preprocessor.transform(X_test),
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
        "feature_columns": list(FEATURE_COLUMNS),
    }
