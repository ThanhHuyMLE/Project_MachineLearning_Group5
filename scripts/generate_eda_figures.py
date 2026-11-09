"""Sinh toàn bộ biểu đồ EDA vào outputs/eda/ từ dữ liệu thật.

Chạy từ thư mục gốc repo:
    .venv/bin/python scripts/generate_eda_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from house_price import eda
from house_price.preprocessing import load_raw_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "eda"

FIGURES = [
    ("price_distribution.png", eda.plot_price_distribution),
    ("missing_values.png", eda.plot_missing_values),
    ("correlation_heatmap.png", eda.plot_correlation_heatmap),
    ("sqft_living_vs_price.png", eda.plot_sqft_living_vs_price),
    ("outlier_boxplots.png", eda.plot_outlier_boxplots),
    ("mean_price_by_grade.png", eda.plot_mean_price_by_grade),
    ("price_by_waterfront.png", eda.plot_price_by_waterfront),
]


def main() -> int:
    """Nạp CSV gốc và lưu 7 biểu đồ EDA vào outputs/eda/."""
    # Console Windows mặc định dùng cp1252, không in được tiếng Việt có dấu
    # nên print() sẽ crash (UnicodeEncodeError). Chuyển sang UTF-8;
    # trên Linux/macOS không đổi gì. Lỗi được bỏ qua nếu stdout/stderr
    # không hỗ trợ reconfigure (ví dụ khi chạy trong Jupyter).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    frame = load_raw_dataset(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, plot_fn in FIGURES:
        plot_fn(frame, save_path=OUTPUT_DIR / filename)
        print(f"Đã lưu {OUTPUT_DIR / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
