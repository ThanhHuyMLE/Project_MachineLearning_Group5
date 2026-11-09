"""Kiểm tra hợp đồng tiền xử lý: chia tập và chống rò rỉ dữ liệu."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    clean_dataset,
    prepare_data,
    split_features_target,
    split_train_val_test,
)


def _sample_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(100, 10, size=n) for col in FEATURE_COLUMNS}
    data["price"] = rng.normal(500_000, 50_000, size=n)
    data["id"] = range(n)
    data["date"] = ["20150101T000000"] * n
    data["Unnamed: 0"] = range(n)
    frame = pd.DataFrame(data)
    frame.loc[0, "bedrooms"] = np.nan
    frame.loc[1, "bathrooms"] = np.nan
    return frame


def test_clean_dataset_drops_id_date_and_index_columns():
    cleaned = clean_dataset(_sample_frame())
    assert "id" not in cleaned.columns
    assert "date" not in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns
    assert list(cleaned.columns)[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS


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


def test_preprocessor_fits_only_on_train(tmp_path):
    """Scaler fit trên train phải có mean đúng bằng mean của train."""
    frame = clean_dataset(_sample_frame(n=200))
    X, y = split_features_target(frame)
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    preprocessor = build_preprocessor(X_train)
    scaler = preprocessor.named_steps["scaler"]
    expected_mean = X_train["bedrooms"].mean()
    assert scaler.mean_[FEATURE_COLUMNS.index("bedrooms")] == pytest.approx(expected_mean)
    assert preprocessor.transform(X_val).shape == (30, len(FEATURE_COLUMNS))
    assert preprocessor.transform(X_test).shape == (30, len(FEATURE_COLUMNS))


def test_prepare_data_returns_three_scaled_partitions(tmp_path):
    path = tmp_path / "houses.csv"
    _sample_frame(n=200).to_csv(path, index=False)
    data = prepare_data(path)
    assert data["X_train"].shape == (140, len(FEATURE_COLUMNS))
    assert data["X_val"].shape == (30, len(FEATURE_COLUMNS))
    assert data["X_test"].shape == (30, len(FEATURE_COLUMNS))
    assert np.isfinite(data["X_train"]).all()
