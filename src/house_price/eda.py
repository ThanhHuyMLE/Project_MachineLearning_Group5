"""Biểu đồ phân tích khám phá dữ liệu EDA (Sinh viên 1).

Mọi hàm chỉ vẽ từ dữ liệu thật được truyền vào, không sinh số giả.
Nếu truyền ``save_path`` thì lưu file ảnh, nếu không thì trả về figure
để notebook hiển thị. Biểu đồ EDA tách riêng khỏi ``visualization.py``
(vốn vẽ biểu đồ đánh giá mô hình của Sinh viên 3).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EDA_COLUMNS = ["price", "sqft_living", "bedrooms", "bathrooms"]


def _save(fig, save_path: str | Path | None):
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_price_distribution(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Histogram phân bố giá bán, kèm đường trung vị và trung bình."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["price"], bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(frame["price"].median(), color="#DD8452", linewidth=2, label="Trung vị")
    ax.axvline(frame["price"].mean(), color="#55A868", linewidth=2, label="Trung bình")
    ax.set_xlabel("Giá bán (USD)")
    ax.set_ylabel("Số căn nhà")
    ax.set_title("Phân bố giá bán nhà King County (n = 21.613)")
    ax.legend()
    return _save(fig, save_path)


def plot_missing_values(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Biểu đồ cột số giá trị khuyết theo từng cột (chỉ cột có khuyết)."""
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "Không có giá trị khuyết", ha="center", va="center")
        ax.set_title("Giá trị khuyết theo cột (tổng 0 giá trị)")
        ax.set_xticks([])
        ax.set_yticks([])
        return _save(fig, save_path)
    ax.bar(missing.index, missing.values, color="#C44E52")
    for i, value in enumerate(missing.values):
        pct = 100 * value / len(frame)
        ax.text(i, value, f"{value} ({pct:.2f}%)", ha="center", va="bottom")
    ax.set_ylabel("Số giá trị khuyết")
    ax.set_title("Giá trị khuyết theo cột (tổng 23 giá trị)")
    ax.set_ylim(0, max(missing.values) * 1.25)
    return _save(fig, save_path)


def plot_correlation_heatmap(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Heatmap tương quan Pearson giữa các đặc trưng số và giá bán."""
    numeric = frame.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=False, ax=ax)
    ax.set_title("Ma trận tương quan Pearson (đặc trưng số và price)")
    return _save(fig, save_path)


def plot_sqft_living_vs_price(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Scatter diện tích sống so với giá bán."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frame["sqft_living"], frame["price"], s=5, alpha=0.25, color="#4C72B0")
    ax.set_xlabel("Diện tích sống sqft_living (sqft)")
    ax.set_ylabel("Giá bán (USD)")
    ax.set_title("Diện tích sống so với giá bán")
    return _save(fig, save_path)


def plot_outlier_boxplots(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Boxplot 4 cột price, sqft_living, bedrooms, bathrooms (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    titles = {
        "price": "Giá bán (USD)",
        "sqft_living": "Diện tích sống (sqft)",
        "bedrooms": "Số phòng ngủ",
        "bathrooms": "Số phòng tắm",
    }
    for ax, column in zip(axes.ravel(), EDA_COLUMNS, strict=True):
        ax.boxplot(frame[column].dropna())
        ax.set_title(titles[column])
        ax.set_xticks([])
    fig.suptitle("Boxplot kiểm tra giá trị ngoại lai")
    return _save(fig, save_path)


def plot_mean_price_by_grade(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Giá trung bình theo grade, ghi số căn nhà trên mỗi cột."""
    grouped = frame.groupby("grade")["price"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index.astype(str), grouped["mean"], color="#4C72B0")
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Grade (điểm chất lượng xây dựng 1-13)")
    ax.set_ylabel("Giá trung bình (USD)")
    ax.set_title("Giá trung bình theo grade")
    ax.set_ylim(0, grouped["mean"].max() * 1.2)
    return _save(fig, save_path)


def plot_price_by_waterfront(frame: pd.DataFrame, save_path: str | Path | None = None):
    """Boxplot giá bán theo waterfront (0 = không giáp nước, 1 = giáp nước)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    no_water = frame.loc[frame["waterfront"] == 0, "price"]
    has_water = frame.loc[frame["waterfront"] == 1, "price"]
    ax.boxplot(
        [no_water, has_water],
        tick_labels=[f"Không giáp nước (n={len(no_water):,})", f"Giáp nước (n={len(has_water):,})"],
    )
    ax.set_ylabel("Giá bán (USD)")
    ax.set_title("Giá bán theo waterfront")
    return _save(fig, save_path)
