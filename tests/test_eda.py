"""Kiểm tra các hàm vẽ biểu đồ EDA, tóm tắt số liệu EDA và ghi file JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from house_price import eda
from house_price.preprocessing import load_raw_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"

matplotlib.use("Agg")


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [200000, 350000, 500000, 750000, 1200000, 300000, 450000],
            "sqft_living": [800, 1200, 1800, 2400, 3500, 1500, 2000],
            "bedrooms": [2, 3, 3, 4, 5, 3, 4],
            "bathrooms": [1.0, 1.5, 2.0, 2.5, 3.0, 2.0, 2.5],
            "grade": [6, 7, 7, 8, 10, 7, 8],
            "waterfront": [0, 0, 0, 1, 0, 0, 0],
            "sqft_lot": [5000, 6000, 7000, 9000, 12000, 6500, 8000],
            "lat": [47.5, 47.6, 47.5, 47.6, 47.7, 47.5, 47.6],
            "long": [-122.2, -122.3, -122.2, -122.3, -122.1, -122.2, -122.3],
        }
    )


@pytest.mark.parametrize(
    "plot_fn",
    [
        eda.plot_price_distribution,
        eda.plot_missing_values,
        eda.plot_correlation_heatmap,
        eda.plot_sqft_living_vs_price,
        eda.plot_outlier_boxplots,
        eda.plot_mean_price_by_grade,
        eda.plot_price_by_waterfront,
    ],
)
def test_eda_plot_returns_figure(frame, plot_fn, tmp_path):
    path = tmp_path / "figure.png"
    fig = plot_fn(frame, save_path=path)
    assert fig is not None
    assert path.is_file()
    assert path.stat().st_size > 0
    matplotlib.pyplot.close(fig)


@pytest.fixture
def summary_frame() -> pd.DataFrame:
    """Frame nhỏ có id/date trùng lặp và cột gốc để tính đặc trưng mới."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4, 5],
            "id": [1, 1, 2, 2, 2, 3],
            "date": [
                "20140502T000000",
                "20140602T000000",
                "20150110T000000",
                "20150110T000000",
                "20150110T000000",
                "20140615T000000",
            ],
            "price": [200000, 210000, 300000, 310000, 320000, 500000],
            "bedrooms": [3.0, 3.0, 3.0, 4.0, 4.0, np.nan],
            "bathrooms": [2.0, 2.0, 2.5, 2.5, 2.5, 3.0],
            "sqft_living": [1500, 1500, 2000, 2000, 2000, 3000],
            "sqft_lot": [5000, 5000, 7000, 7000, 7000, 12000],
            "floors": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "waterfront": [0, 0, 0, 0, 0, 1],
            "view": [0, 0, 1, 1, 1, 2],
            "condition": [3, 3, 3, 4, 4, 4],
            "grade": [7, 7, 8, 8, 8, 9],
            "sqft_above": [1500, 1500, 2000, 2000, 2000, 3000],
            "sqft_basement": [0, 0, 0, 0, 0, 0],
            "yr_built": [1990, 1990, 1980, 1980, 1980, 2005],
            "yr_renovated": [0, 0, 2000, 2000, 2000, 0],
            "zipcode": [98101, 98101, 98102, 98102, 98102, 98103],
            "lat": [47.5, 47.5, 47.6, 47.6, 47.6, 47.7],
            "long": [-122.2, -122.2, -122.3, -122.3, -122.3, -122.1],
            "sqft_living15": [1600, 1600, 2100, 2100, 2100, 3100],
            "sqft_lot15": [5500, 5500, 7500, 7500, 7500, 12500],
        }
    )


def test_summarize_eda_structure(summary_frame):
    summary = eda.summarize_eda(summary_frame)
    assert set(summary) == {
        "dataset",
        "missing",
        "duplicates",
        "price",
        "correlations_with_price",
        "mean_price_by_grade",
        "mean_price_by_view",
        "waterfront",
        "engineered_features",
    }
    assert summary["dataset"]["rows"] == 6
    assert summary["dataset"]["columns"] == 22
    assert len(summary["dataset"]["column_names"]) == 22
    assert summary["dataset"]["column_names"][0] == "Unnamed: 0"


def test_summarize_eda_missing_and_duplicates(summary_frame):
    summary = eda.summarize_eda(summary_frame)
    assert summary["missing"] == {"by_column": {"bedrooms": 1}, "total": 1}
    assert summary["duplicates"] == {
        "fully_duplicate_rows": 0,
        "unique_ids": 3,
        "repeated_ids": 2,
        "ids_twice": 1,
        "ids_thrice": 1,
        "rows_involved": 5,
        "excess_rows": 3,
        "same_date_groups": 1,
        "different_price_groups": 2,
        "identical_groups": 0,
        "max_date_spread_days": 31,
    }


def test_summarize_eda_price_stats(summary_frame):
    price = eda.summarize_eda(summary_frame)["price"]
    assert price["min"] == 200000
    assert price["max"] == 500000
    assert price["median"] == 305000
    assert price["mean"] == pytest.approx(306666.6667, abs=0.01)
    assert price["q1"] == pytest.approx(232500, abs=0.01)
    assert price["q3"] == pytest.approx(317500, abs=0.01)
    assert price["iqr_upper"] == pytest.approx(445000, abs=0.01)
    assert price["rows_over_iqr_upper"] == 1
    assert price["rows_over_1m"] == 0
    assert price["rows_over_2m"] == 0


def test_summarize_eda_group_means(summary_frame):
    summary = eda.summarize_eda(summary_frame)
    assert summary["mean_price_by_grade"]["7"] == {"mean": 205000, "count": 2}
    assert summary["mean_price_by_grade"]["8"] == {"mean": 310000, "count": 3}
    assert summary["mean_price_by_grade"]["9"] == {"mean": 500000, "count": 1}
    assert summary["mean_price_by_view"]["0"] == {"mean": 205000, "count": 2}
    assert summary["waterfront"]["count_with_waterfront"] == 1
    assert summary["waterfront"]["mean_with_waterfront"] == 500000
    assert summary["waterfront"]["mean_without_waterfront"] == 268000


def test_summarize_eda_engineered(summary_frame):
    engineered = eda.summarize_eda(summary_frame)["engineered_features"]
    assert engineered["house_age"]["min"] == 9
    assert engineered["house_age"]["max"] == 35
    assert engineered["house_age"]["median"] == 29.5
    assert engineered["house_age"]["mean"] == 27.0
    assert engineered["was_renovated"]["count_renovated"] == 3
    assert engineered["was_renovated"]["pct_renovated"] == 50.0
    assert engineered["sqft_per_bedroom"]["min"] == 500
    assert engineered["sqft_per_bedroom"]["median"] == 500
    assert engineered["sqft_per_bedroom"]["mean"] == pytest.approx(533.3333, abs=0.01)
    assert engineered["sqft_per_bedroom"]["nan_count"] == 1


def test_summarize_eda_correlations(summary_frame):
    corr = eda.summarize_eda(summary_frame)["correlations_with_price"]
    assert "price" not in corr
    assert "Unnamed: 0" not in corr
    assert "id" not in corr
    assert corr["sqft_living"] > 0
    assert all(-1.0 <= value <= 1.0 for value in corr.values())


def test_summarize_eda_requires_price():
    frame = pd.DataFrame({"bedrooms": [3.0, 4.0]})
    with pytest.raises(ValueError, match="price"):
        eda.summarize_eda(frame)


def test_summarize_eda_skips_missing_sections():
    frame = pd.DataFrame(
        {
            "price": [200000, 300000, 400000],
            "bedrooms": [3.0, 4.0, 5.0],
        }
    )
    summary = eda.summarize_eda(frame)
    assert set(summary) == {"dataset", "missing", "price", "correlations_with_price"}
    assert summary["dataset"]["rows"] == 3


def test_summarize_eda_real_csv():
    frame = load_raw_dataset(REAL_CSV)
    summary = eda.summarize_eda(frame)

    assert summary["dataset"] == {
        "rows": 21613,
        "columns": 22,
        "column_names": list(frame.columns),
    }
    assert summary["missing"] == {
        "by_column": {"bedrooms": 13, "bathrooms": 10},
        "total": 23,
    }
    assert summary["duplicates"] == {
        "fully_duplicate_rows": 0,
        "unique_ids": 21436,
        "repeated_ids": 176,
        "ids_twice": 175,
        "ids_thrice": 1,
        "rows_involved": 353,
        "excess_rows": 177,
        "same_date_groups": 0,
        "different_price_groups": 173,
        "identical_groups": 3,
        "max_date_spread_days": 315,
    }

    price = summary["price"]
    assert price["min"] == 75000
    assert price["q1"] == pytest.approx(321950, abs=1.0)
    assert price["median"] == 450000
    assert price["mean"] == pytest.approx(540088, abs=1.0)
    assert price["q3"] == pytest.approx(645000, abs=1.0)
    assert price["max"] == 7700000
    assert price["skew"] == pytest.approx(4.02, abs=0.005)
    assert price["kurtosis"] == pytest.approx(34.59, abs=0.005)
    assert price["p90"] == pytest.approx(887000, abs=1.0)
    assert price["p95"] == pytest.approx(1156480, abs=1.0)
    assert price["p99"] == pytest.approx(1964400, abs=1.0)
    assert price["iqr_upper"] == pytest.approx(1129575, abs=1.0)
    assert price["rows_over_iqr_upper"] == 1146
    assert price["pct_over_iqr_upper"] == pytest.approx(5.3, abs=0.05)
    assert price["rows_over_1m"] == 1465
    assert price["pct_over_1m"] == pytest.approx(6.8, abs=0.05)
    assert price["rows_over_2m"] == 198
    assert price["pct_over_2m"] == pytest.approx(0.9, abs=0.05)

    corr = summary["correlations_with_price"]
    assert corr["sqft_living"] == pytest.approx(0.702, abs=0.001)
    assert corr["grade"] == pytest.approx(0.667, abs=0.001)
    assert corr["sqft_above"] == pytest.approx(0.606, abs=0.001)
    assert corr["sqft_living15"] == pytest.approx(0.585, abs=0.001)
    assert corr["bathrooms"] == pytest.approx(0.526, abs=0.001)

    by_grade = summary["mean_price_by_grade"]
    assert by_grade["1"] == {"mean": pytest.approx(142000, abs=1.0), "count": 1}
    assert by_grade["4"]["count"] == 29
    assert by_grade["6"] == {"mean": pytest.approx(301920, abs=1.0), "count": 2038}
    assert by_grade["7"] == {"mean": pytest.approx(402590, abs=1.0), "count": 8981}
    assert by_grade["8"] == {"mean": pytest.approx(542853, abs=1.0), "count": 6068}
    assert by_grade["13"] == {"mean": pytest.approx(3709615, abs=1.0), "count": 13}
    by_view = summary["mean_price_by_view"]
    assert by_view["0"]["mean"] == pytest.approx(496564, abs=1.0)
    assert by_view["4"]["mean"] == pytest.approx(1463711, abs=1.0)

    assert summary["waterfront"]["count_with_waterfront"] == 163
    assert summary["waterfront"]["mean_with_waterfront"] == pytest.approx(1661876, abs=1.0)
    assert summary["waterfront"]["mean_without_waterfront"] == pytest.approx(531564, abs=1.0)

    engineered = summary["engineered_features"]
    assert engineered["house_age"]["min"] == 0
    assert engineered["house_age"]["max"] == 115
    assert engineered["house_age"]["median"] == 40.0
    assert engineered["house_age"]["mean"] == pytest.approx(43.3, abs=0.05)
    assert engineered["house_age"]["correlation_with_price"] == pytest.approx(-0.054, abs=0.001)
    assert engineered["was_renovated"]["count_renovated"] == 908
    assert engineered["was_renovated"]["pct_renovated"] == pytest.approx(4.20, abs=0.01)
    assert engineered["was_renovated"]["correlation_with_price"] == pytest.approx(0.126, abs=0.001)
    assert engineered["sqft_per_bedroom"]["min"] == pytest.approx(49.1, abs=0.1)
    assert engineered["sqft_per_bedroom"]["max"] == pytest.approx(3420, abs=1.0)
    assert engineered["sqft_per_bedroom"]["median"] == pytest.approx(576.7, abs=0.1)
    assert engineered["sqft_per_bedroom"]["mean"] == pytest.approx(618.2, abs=0.1)
    assert engineered["sqft_per_bedroom"]["nan_count"] == 13
    assert engineered["sqft_per_bedroom"]["correlation_with_price"] == pytest.approx(
        0.580, abs=0.001
    )


def test_export_eda_summary_writes_json(summary_frame, tmp_path):
    path = tmp_path / "eda_summary.json"
    written = eda.export_eda_summary(summary_frame, path)
    assert written == path
    assert path.is_file()
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["dataset"]["rows"] == 6
    assert summary["missing"]["total"] == 1
    assert summary["duplicates"]["repeated_ids"] == 2
