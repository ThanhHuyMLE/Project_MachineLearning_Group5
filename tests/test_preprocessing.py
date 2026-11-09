"""Kiểm tra hợp đồng tiền xử lý: kỹ thuật đặc trưng, chia theo nhóm id,
chia tập, chống rò rỉ dữ liệu và mã hóa zipcode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price.preprocessing import (
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_preprocessor,
    clean_dataset,
    engineer_features,
    output_feature_names,
    prepare_data,
    split_features_target,
    split_train_val_test,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"
ZIPCODE_CHOICES = [98001, 98002, 98101, 98102, 98103]


def _sample_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "bedrooms": rng.integers(1, 6, size=n).astype(float),
        "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5], size=n),
        "sqft_living": rng.integers(500, 4000, size=n),
        "sqft_lot": rng.integers(1000, 20000, size=n),
        "floors": rng.choice([1.0, 1.5, 2.0], size=n),
        "waterfront": rng.integers(0, 2, size=n),
        "view": rng.integers(0, 5, size=n),
        "condition": rng.integers(1, 6, size=n),
        "grade": rng.integers(4, 12, size=n),
        "sqft_above": rng.integers(500, 3000, size=n),
        "sqft_basement": rng.integers(0, 1000, size=n),
        "yr_built": rng.integers(1950, 2015, size=n),
        "yr_renovated": np.where(rng.random(n) < 0.1, rng.integers(1980, 2015, size=n), 0),
        "zipcode": rng.choice(ZIPCODE_CHOICES, size=n),
        "lat": rng.uniform(47.1, 47.8, size=n),
        "long": rng.uniform(-122.5, -121.3, size=n),
        "sqft_living15": rng.integers(500, 4000, size=n),
        "sqft_lot15": rng.integers(1000, 20000, size=n),
    }
    data["price"] = rng.normal(500_000, 50_000, size=n)
    data["id"] = range(n)
    data["date"] = ["20150101T000000"] * n
    data["Unnamed: 0"] = range(n)
    frame = pd.DataFrame(data)
    frame.loc[0, "bedrooms"] = np.nan
    frame.loc[1, "bathrooms"] = np.nan
    return frame


def _frame_with_repeated_ids(n_ids: int = 50, seed: int = 0) -> pd.DataFrame:
    """Mỗi id xuất hiện đúng 2 lần, mô phỏng nhà bán lặp."""
    frame = _sample_frame(n=n_ids * 2, seed=seed)
    frame["id"] = np.repeat(np.arange(n_ids), 2)
    frame["date"] = np.where(np.arange(len(frame)) % 2 == 0, "20140501T000000", "20150101T000000")
    return frame


def test_clean_dataset_drops_id_date_and_index_columns():
    cleaned = clean_dataset(_sample_frame())
    assert "id" not in cleaned.columns
    assert "date" not in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns
    assert "yr_built" not in cleaned.columns
    assert "yr_renovated" not in cleaned.columns
    assert list(cleaned.columns)[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert "house_age" in cleaned.columns
    assert "was_renovated" in cleaned.columns
    assert "sqft_per_bedroom" in cleaned.columns


def test_engineer_features_house_age_clips_negative():
    frame = _sample_frame(n=5)
    frame.loc[0, "yr_built"] = 2016  # Xây sau năm bán 2015: lỗi ghi dữ liệu.
    engineered = engineer_features(frame)
    assert engineered.loc[0, "house_age"] == 0
    assert (engineered["house_age"] >= 0).all()


def test_engineer_features_was_renovated_flag():
    frame = _sample_frame(n=5)
    frame["yr_renovated"] = [0, 1990, 0, 2010, 0]
    engineered = engineer_features(frame)
    assert list(engineered["was_renovated"]) == [0, 1, 0, 1, 0]


def test_engineer_features_sqft_per_bedroom_handles_zero_and_missing():
    frame = _sample_frame(n=5)
    frame.loc[0, "bedrooms"] = 0
    frame.loc[1, "bedrooms"] = np.nan
    frame.loc[2, "bedrooms"] = 2
    frame.loc[2, "sqft_living"] = 1000
    engineered = engineer_features(frame)
    assert np.isnan(engineered.loc[0, "sqft_per_bedroom"])
    assert np.isnan(engineered.loc[1, "sqft_per_bedroom"])
    assert engineered.loc[2, "sqft_per_bedroom"] == pytest.approx(500.0)


def test_split_ratios_are_70_15_15():
    X, y = split_features_target(clean_dataset(_sample_frame(n=200)))
    X_train, X_val, X_test, _y_train, _y_val, _y_test = split_train_val_test(X, y)
    assert len(X_train) == 140
    assert len(X_val) == 30
    assert len(X_test) == 30


def test_splits_do_not_overlap():
    X, y = split_features_target(clean_dataset(_sample_frame(n=200)))
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


def test_group_split_keeps_same_id_in_one_partition():
    frame = _frame_with_repeated_ids(n_ids=50)
    groups = frame["id"]
    X, y = split_features_target(clean_dataset(frame))
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y, groups=groups)
    ids_train = set(groups.loc[X_train.index])
    ids_val = set(groups.loc[X_val.index])
    ids_test = set(groups.loc[X_test.index])
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
    assert len(X_train) + len(X_val) + len(X_test) == len(frame)
    assert 60 <= len(X_train) <= 80
    assert 10 <= len(X_val) <= 20
    assert 10 <= len(X_test) <= 20


def test_splits_are_reproducible():
    X, y = split_features_target(clean_dataset(_sample_frame(n=200)))
    first = split_train_val_test(X, y)
    second = split_train_val_test(X, y)
    for a, b in zip(first, second, strict=True):
        assert (
            a.index.equals(b.index) if isinstance(a, pd.DataFrame) else (a.values == b.values).all()
        )


def test_preprocessor_fits_only_on_train(tmp_path):
    """Scaler fit trên train phải có mean đúng bằng mean của train đã bù khuyết."""
    frame = clean_dataset(_sample_frame(n=200))
    X, y = split_features_target(frame)
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    preprocessor = build_preprocessor(X_train)
    numeric = preprocessor.named_transformers_["num"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    imputed_train = imputer.transform(X_train[NUMERIC_COLUMNS])
    expected_mean = imputed_train[:, NUMERIC_COLUMNS.index("bedrooms")].mean()
    assert scaler.mean_[NUMERIC_COLUMNS.index("bedrooms")] == pytest.approx(expected_mean)
    n_out = len(output_feature_names(preprocessor))
    assert preprocessor.transform(X_val).shape == (30, n_out)
    assert preprocessor.transform(X_test).shape == (30, n_out)
    assert np.isfinite(preprocessor.transform(X_val)).all()


def test_preprocessor_ignores_unseen_zipcode():
    frame = clean_dataset(_sample_frame(n=200))
    X, y = split_features_target(frame)
    X_train, X_val, _, _, _, _ = split_train_val_test(X, y)
    preprocessor = build_preprocessor(X_train)
    X_unseen = X_val.copy()
    X_unseen["zipcode"] = 99999  # Mã chưa từng thấy ở train.
    transformed = preprocessor.transform(X_unseen)
    assert transformed.shape[1] == len(output_feature_names(preprocessor))
    assert np.isfinite(transformed).all()


def test_split_features_target_excludes_target_and_ids():
    X, y = split_features_target(clean_dataset(_sample_frame()))
    assert "price" not in X.columns
    assert "id" not in X.columns
    assert "date" not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert (y.name) == "price"


def test_prepare_data_returns_three_scaled_partitions(tmp_path):
    path = tmp_path / "houses.csv"
    _frame_with_repeated_ids(n_ids=100).to_csv(path, index=False)
    data = prepare_data(path)
    n_out = 18 + len(ZIPCODE_CHOICES)
    assert data["X_train"].shape[1] == n_out
    assert data["X_val"].shape[1] == n_out
    assert data["X_test"].shape[1] == n_out
    assert data["X_train"].shape[0] + data["X_val"].shape[0] + data["X_test"].shape[0] == 200
    assert 130 <= data["X_train"].shape[0] <= 150
    assert 25 <= data["X_val"].shape[0] <= 35
    assert 25 <= data["X_test"].shape[0] <= 35
    assert np.isfinite(data["X_train"]).all()
    assert np.isfinite(data["X_val"]).all()
    assert np.isfinite(data["X_test"]).all()
    assert len(data["feature_columns"]) == n_out
    assert data["input_columns"] == FEATURE_COLUMNS
    assert any(name.startswith("zipcode_") for name in data["feature_columns"])
    ids_train = set(data["id_train"])
    assert ids_train.isdisjoint(data["id_val"])
    assert ids_train.isdisjoint(data["id_test"])
    assert set(data["id_val"]).isdisjoint(data["id_test"])


def test_feature_columns_order_is_stable(tmp_path):
    path = tmp_path / "houses.csv"
    _frame_with_repeated_ids(n_ids=100).to_csv(path, index=False)
    first = prepare_data(path)["feature_columns"]
    second = prepare_data(path)["feature_columns"]
    assert first == second


@pytest.mark.skipif(not REAL_CSV.is_file(), reason="Thiếu file CSV King County thật")
def test_prepare_data_on_real_csv():
    """Kiểm tra pipeline trên CSV thật: kích thước, one-hot 70 zipcode, không rò id."""
    data = prepare_data(REAL_CSV)
    assert data["X_train"].shape == (15129, 88)
    assert data["X_val"].shape == (3243, 88)
    assert data["X_test"].shape == (3241, 88)
    assert np.isfinite(data["X_train"]).all()
    assert np.isfinite(data["X_val"]).all()
    assert np.isfinite(data["X_test"]).all()
    assert len(data["feature_columns"]) == 88
    assert set(data["id_train"]).isdisjoint(data["id_val"])
    assert set(data["id_train"]).isdisjoint(data["id_test"])
    assert set(data["id_val"]).isdisjoint(data["id_test"])
