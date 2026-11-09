"""Nạp dữ liệu, kỹ thuật đặc trưng, chia tập và chuẩn hóa (Sinh viên 1).

Quy tắc chống rò rỉ dữ liệu (Data Leakage) áp dụng trong toàn bộ module:
chia train/validation/test TRƯỚC, chỉ fit imputer, scaler và one-hot
encoder trên tập train, rồi dùng cùng đối tượng đã fit để transform
validation, test và dữ liệu mới.

Chia tập mặc định là chia theo nhóm (group-aware) trên ``id`` căn nhà:
mọi giao dịch của cùng một căn nhà nằm trong cùng một partition, để
đánh giá đo khả năng dự đoán căn nhà chưa thấy, không phải giao dịch
chưa thấy của căn nhà đã thấy. Xem ``docs/adr/005-chia-theo-nhom-id.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price"

# Các cột không mang thông tin dự báo, loại khỏi tập huấn luyện.
DROP_COLUMNS = ["Unnamed: 0", "id", "date"]

# 18 cột đặc trưng gốc trong file CSV (trước kỹ thuật đặc trưng).
RAW_COLUMNS = [
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

# 18 đặc trưng số sau kỹ thuật đặc trưng (xem engineer_features):
# yr_built -> house_age, yr_renovated -> was_renovated (nhị phân),
# thêm sqft_per_bedroom. Thứ tự cố định trong hợp đồng dữ liệu.
NUMERIC_COLUMNS = [
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
    "house_age",
    "was_renovated",
    "sqft_per_bedroom",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

# zipcode là mã vùng, không phải số liên tục: mã hóa one-hot.
# Xem docs/adr/006-feature-engineering-va-zipcode.md.
CATEGORICAL_COLUMNS = ["zipcode"]

# 19 cột đầu vào trước mã hóa (18 số + 1 phân loại).
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

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


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng mới từ cột gốc, trả về bản sao (không fit gì cả).

    - ``house_age = năm bán - yr_built``, kẹp tối thiểu 0. 12 dòng có
      ``yr_built`` lớn hơn năm bán (tuổi âm, lỗi ghi dữ liệu) được kẹp về 0
      thay vì xóa dòng. Cột ``yr_built`` gốc được thay bằng ``house_age``
      vì hai cột gần như trùng thông tin trong cùng một năm bán.
    - ``was_renovated = 1`` nếu ``0 < yr_renovated <= năm bán``, ngược lại 0.
      Quy ước của dữ liệu: 0 nghĩa là chưa từng cải tạo, nên giá trị số của
      ``yr_renovated`` trộn cờ hiệu với năm, không dùng trực tiếp làm
      đặc trưng số. 6 dòng có ``yr_renovated`` lớn hơn năm bán (cải tạo sau
      khi bán, không thể xảy ra) được tính là chưa cải tạo, giống cách
      ``house_age`` kẹp tuổi âm về 0. Cột ``yr_renovated`` gốc được thay
      bằng cờ này.
    - ``sqft_per_bedroom = sqft_living / bedrooms``. Khi ``bedrooms``
      khuyết hoặc bằng 0 thì kết quả là NaN và sẽ được bù bằng median
      của tập train trong preprocessor, không bao giờ chia cho 0.

    Nếu frame đã có sẵn 3 đặc trưng mới (không còn cột gốc để tính)
    thì trả về bản sao nguyên vẹn, để hàm dùng được cho dữ liệu mới.
    """
    engineered = frame.copy()
    raw_needed = {"date", "yr_built", "yr_renovated"}
    if raw_needed.issubset(engineered.columns):
        sale_year = pd.to_datetime(engineered["date"], format="%Y%m%dT%H%M%S").dt.year
        engineered["house_age"] = (sale_year - engineered["yr_built"]).clip(lower=0)
        engineered["was_renovated"] = (
            (engineered["yr_renovated"] > 0) & (engineered["yr_renovated"] <= sale_year)
        ).astype(int)
        bedrooms = engineered["bedrooms"].replace(0, np.nan)
        engineered["sqft_per_bedroom"] = engineered["sqft_living"] / bedrooms
        engineered = engineered.drop(columns=["yr_built", "yr_renovated"])
    elif not set(NUMERIC_COLUMNS).issubset(engineered.columns):
        missing = sorted(set(NUMERIC_COLUMNS) - set(engineered.columns))
        raise ValueError(f"Không tính được đặc trưng mới, thiếu cột: {missing}")
    return engineered


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Kỹ thuật đặc trưng rồi loại cột thừa, CHƯA bù khuyết hay chuẩn hóa.

    Bù khuyết và chuẩn hóa nằm trong preprocessor (xem ``build_preprocessor``)
    để đảm bảo chỉ fit trên tập train. Hàm này chỉ tạo đặc trưng mới,
    loại cột ``id``/``date``/index và kiểm tra đủ 19 cột đặc trưng đầu vào.
    """
    engineered = engineer_features(frame)
    cleaned = engineered.drop(columns=[c for c in DROP_COLUMNS if c in engineered.columns])
    missing = [c for c in FEATURE_COLUMNS if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột đặc trưng: {missing}")
    return cleaned[FEATURE_COLUMNS + [TARGET_COLUMN]]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách DataFrame thành ma trận đặc trưng X và mục tiêu y."""
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].copy()


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Chia dữ liệu thành train/validation/test theo tỷ lệ 70/15/15.

    Hồi quy không có nhãn lớp nên không stratify; chỉ xáo trộn ngẫu nhiên
    với ``random_state`` cố định để kết quả tái lập được.

    Khi truyền ``groups`` (mã ``id`` căn nhà), dùng GroupShuffleSplit để
    mọi quan sát của cùng một căn nhà nằm trong cùng một partition. Khi
    ``groups`` là None thì chia ngẫu nhiên theo dòng như Phase 1.
    """
    if groups is None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=random_state
        )
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size + val_size, random_state=random_state
    )
    train_idx, temp_idx = next(splitter.split(X, y, groups))
    val_ratio = val_size / (test_size + val_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=1 - val_ratio, random_state=random_state)
    val_rel, test_rel = next(splitter2.split(X.iloc[temp_idx], y.iloc[temp_idx], groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return (
        X.iloc[train_idx],
        X.iloc[val_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[val_idx],
        y.iloc[test_idx],
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor trên tập train, trả về ColumnTransformer đã fit.

    Đây là điểm duy nhất được fit tiền xử lý. Mọi tập khác chỉ được
    transform bằng preprocessor này, không được fit lại.

    - Đặc trưng số: bù khuyết median rồi StandardScaler.
    - ``zipcode`` (mã vùng): bù giá trị thường gặp nhất rồi one-hot
      (mã chưa thấy ở train được bỏ qua, cho toàn 0).
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    preprocessor.fit(X_train)
    return preprocessor


def output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Tên cột sau mã hóa, bỏ tiền tố ``num__``/``cat__`` của sklearn."""
    names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in names]


def prepare_data(
    path: str | Path,
) -> dict[str, object]:
    """Chạy trọn tiền xử lý và trả về dict chứa 3 tập đã chuẩn hóa.

    Thứ tự: nạp -> kỹ thuật đặc trưng -> tách X/y -> chia 3 tập theo
    nhóm ``id`` -> fit preprocessor trên train -> transform cả 3 tập.
    Trả về các khóa: ``X_train``, ``X_val``, ``X_test``, ``y_train``,
    ``y_val``, ``y_test``, ``preprocessor``, ``feature_columns`` (tên cột
    sau mã hóa, 18 số + one-hot zipcode), ``input_columns`` (19 cột đầu
    vào trước mã hóa), ``id_train``, ``id_val``, ``id_test`` (mã ``id``
    căn nhà theo từng partition, chỉ để kiểm chứng chia theo nhóm,
    không phải đặc trưng mô hình).
    """
    raw = load_raw_dataset(path)
    groups = raw["id"] if "id" in raw.columns else None
    frame = clean_dataset(raw)
    X, y = split_features_target(frame)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, groups=groups)
    preprocessor = build_preprocessor(X_train)
    result = {
        "X_train": preprocessor.transform(X_train),
        "X_val": preprocessor.transform(X_val),
        "X_test": preprocessor.transform(X_test),
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
        "feature_columns": output_feature_names(preprocessor),
        "input_columns": list(FEATURE_COLUMNS),
    }
    if groups is not None:
        groups = np.asarray(groups)
        result["id_train"] = groups[y_train.index]
        result["id_val"] = groups[y_val.index]
        result["id_test"] = groups[y_test.index]
    return result
