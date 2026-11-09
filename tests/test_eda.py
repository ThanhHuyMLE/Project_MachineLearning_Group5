"""Kiểm tra các hàm vẽ biểu đồ EDA trả về figure và lưu được file."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

from house_price import eda

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
