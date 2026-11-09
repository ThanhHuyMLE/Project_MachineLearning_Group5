"""Bảng so sánh các mô hình Hồi quy (Sinh viên 4).

Mỗi mô hình phải được đánh giá trên cùng tập test và cùng quy trình
đánh giá thì bảng so sánh mới công bằng.
"""

from __future__ import annotations

import pandas as pd

METRIC_COLUMNS = ["model", "mse", "rmse", "mae", "r2"]


def compare_models(results: dict[str, dict[str, float | None]]) -> pd.DataFrame:
    """Gộp metric của các mô hình thành DataFrame, sắp theo MSE tăng dần.

    ``results`` có dạng ``{"ten_mo_hinh": {"mse": ..., "rmse": ..., ...}}``.
    """
    rows = [
        {"model": name, **{k: metrics.get(k) for k in METRIC_COLUMNS[1:]}}
        for name, metrics in results.items()
    ]
    table = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    return table.sort_values("mse").reset_index(drop=True)
