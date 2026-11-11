"""Biểu đồ phân tích khám phá dữ liệu EDA (Stage 01: Data Understanding & Preprocessing).

Mọi hàm chỉ vẽ từ dữ liệu thật được truyền vào, không sinh số giả.
Nếu truyền ``save_path`` thì lưu file ảnh, nếu không thì trả về figure
để notebook hiển thị. Biểu đồ EDA tách riêng khỏi ``visualization.py``
(vốn vẽ biểu đồ đánh giá mô hình của Stage 03: Model Training & Evaluation).
(vốn vẽ biểu đồ đánh giá mô hình của Sinh viên 3).

``summarize_eda`` tính toàn bộ số liệu EDA mô tả (thiếu, trùng, phân bố
giá, tương quan, giá theo grade/view/waterfront, đặc trưng mới) thành dict
ghi được ra JSON; ``export_eda_summary`` ghi dict đó ra file để các giai
đoạn sau dùng lại mà không cần tính lại hay copy số từ notebook.
"""

from __future__ import annotations

import json
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


def _describe_numeric(series: pd.Series) -> dict[str, float]:
    """Min/max/trung vị/trung bình của một cột số (bỏ NaN)."""
    values = series.dropna()
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "mean": float(values.mean()),
    }


def _summarize_duplicates(frame: pd.DataFrame) -> dict[str, int]:
    """Thống kê id lặp; cần cột ``id`` và ``date`` (định dạng %Y%m%dT%H%M%S)."""
    duplicated_id = frame[frame.duplicated("id", keep=False)].copy()
    duplicated_id["sale_date"] = frame["date"].astype(str).str.slice(0, 8)
    id_counts = frame["id"].value_counts()
    grouped = duplicated_id.groupby("id")
    sale_dates = duplicated_id.copy()
    sale_dates["sale_date_dt"] = pd.to_datetime(sale_dates["sale_date"], format="%Y%m%d")
    spread = (
        sale_dates.groupby("id")["sale_date_dt"].max()
        - sale_dates.groupby("id")["sale_date_dt"].min()
    ).dt.days
    model_columns = [c for c in frame.columns if c not in ("id", "date", "Unnamed: 0")]
    identical_rows = duplicated_id.duplicated(subset=model_columns, keep=False)
    return {
        "fully_duplicate_rows": int(frame.duplicated().sum()),
        "unique_ids": int(frame["id"].nunique()),
        "repeated_ids": int((id_counts > 1).sum()),
        "ids_twice": int((id_counts == 2).sum()),
        "ids_thrice": int((id_counts == 3).sum()),
        "rows_involved": len(duplicated_id),
        "excess_rows": int((id_counts - 1).clip(lower=0).sum()),
        "same_date_groups": int((grouped["sale_date"].nunique() == 1).sum()),
        "different_price_groups": int((grouped["price"].nunique() > 1).sum()),
        "identical_groups": int(duplicated_id[identical_rows]["id"].nunique()),
        "max_date_spread_days": int(spread.max()) if len(spread) else 0,
    }


def summarize_eda(frame: pd.DataFrame) -> dict[str, object]:
    """Tính toàn bộ số liệu EDA mô tả từ frame thô, trả về dict ghi được ra JSON.

    Mọi số liệu đều tính từ ``frame`` truyền vào, không ghi cứng số nào.
    Các mục chỉ tính khi đủ cột: ``duplicates`` cần ``id`` + ``date``,
    ``mean_price_by_grade``/``mean_price_by_view`` cần ``grade``/``view``,
    ``waterfront`` cần ``waterfront``, ``engineered_features`` cần các cột
    gốc để ``engineer_features`` chạy được (``date``, ``yr_built``,
    ``yr_renovated``). Thiếu cột thì bỏ qua mục đó, không báo lỗi.
    Ngoại lệ duy nhất: thiếu cột mục tiêu ``price`` thì báo ValueError.

    Tương quan tính trên các cột số (trừ ``price`` và các cột không phải
    đặc trưng như ``Unnamed: 0``, ``id``). Số thực giữ nguyên độ chính
    xác đầy đủ, không làm tròn.
    """
    if "price" not in frame.columns:
        raise ValueError("Dữ liệu thiếu cột mục tiêu 'price'.")
    price = frame["price"]
    summary: dict[str, object] = {}

    summary["dataset"] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": [str(column) for column in frame.columns],
    }

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    summary["missing"] = {
        "by_column": {str(column): int(value) for column, value in missing.items()},
        "total": int(missing.sum()),
    }

    if "id" in frame.columns and "date" in frame.columns:
        summary["duplicates"] = _summarize_duplicates(frame)

    q1 = float(price.quantile(0.25))
    q3 = float(price.quantile(0.75))
    iqr_upper = q3 + 1.5 * (q3 - q1)
    summary["price"] = {
        "min": float(price.min()),
        "q1": q1,
        "median": float(price.median()),
        "mean": float(price.mean()),
        "q3": q3,
        "max": float(price.max()),
        "std": float(price.std()),
        "skew": float(price.skew()),
        "kurtosis": float(price.kurtosis()),
        "p90": float(price.quantile(0.90)),
        "p95": float(price.quantile(0.95)),
        "p99": float(price.quantile(0.99)),
        "iqr_upper": float(iqr_upper),
        "rows_over_iqr_upper": int((price > iqr_upper).sum()),
        "pct_over_iqr_upper": float((price > iqr_upper).mean() * 100),
        "rows_over_1m": int((price > 1_000_000).sum()),
        "pct_over_1m": float((price > 1_000_000).mean() * 100),
        "rows_over_2m": int((price > 2_000_000).sum()),
        "pct_over_2m": float((price > 2_000_000).mean() * 100),
    }

    numeric = frame.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)["price"].drop("price", errors="ignore")
    corr = corr.drop(
        [column for column in ("Unnamed: 0", "id") if column in corr.index],
        errors="ignore",
    )
    corr = corr.dropna().sort_values(ascending=False)
    summary["correlations_with_price"] = {
        str(column): float(value) for column, value in corr.items()
    }

    for group_column, key in (("grade", "mean_price_by_grade"), ("view", "mean_price_by_view")):
        if group_column in frame.columns:
            grouped = frame.groupby(group_column)["price"].agg(["mean", "count"])
            summary[key] = {
                str(level): {"mean": float(row["mean"]), "count": int(row["count"])}
                for level, row in grouped.iterrows()
            }

    if "waterfront" in frame.columns:
        with_water = frame.loc[frame["waterfront"] == 1, "price"]
        without_water = frame.loc[frame["waterfront"] == 0, "price"]
        summary["waterfront"] = {
            "count_with_waterfront": len(with_water),
            "mean_with_waterfront": float(with_water.mean()) if len(with_water) else None,
            "mean_without_waterfront": float(without_water.mean()) if len(without_water) else None,
        }

    try:
        from house_price.preprocessing import engineer_features

        engineered = engineer_features(frame)
    except ValueError:
        engineered = None
    if engineered is not None and {
        "house_age",
        "was_renovated",
        "sqft_per_bedroom",
    }.issubset(engineered.columns):
        engineered_corr = engineered[
            ["house_age", "was_renovated", "sqft_per_bedroom", "price"]
        ].corr(numeric_only=True)["price"]
        summary["engineered_features"] = {
            "house_age": {
                **_describe_numeric(engineered["house_age"]),
                "correlation_with_price": float(engineered_corr["house_age"]),
            },
            "was_renovated": {
                "count_renovated": int(engineered["was_renovated"].sum()),
                "pct_renovated": float(engineered["was_renovated"].mean() * 100),
                "correlation_with_price": float(engineered_corr["was_renovated"]),
            },
            "sqft_per_bedroom": {
                **_describe_numeric(engineered["sqft_per_bedroom"]),
                "nan_count": int(engineered["sqft_per_bedroom"].isna().sum()),
                "correlation_with_price": float(engineered_corr["sqft_per_bedroom"]),
            },
        }

    return summary


def export_eda_summary(frame: pd.DataFrame, path: str | Path) -> Path:
    """Tính ``summarize_eda(frame)`` rồi ghi JSON (UTF-8, indent 2).

    Trả về Path đã ghi. Tạo thư mục cha nếu chưa có.
    """
    summary = summarize_eda(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
