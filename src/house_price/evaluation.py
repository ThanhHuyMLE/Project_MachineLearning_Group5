"""Chỉ số đánh giá Hồi quy (Sinh viên 3).

MSE, RMSE, MAE càng nhỏ càng tốt. R2 càng lớn càng tốt, có thể âm.
MAPE chỉ có ý nghĩa khi giá trị thật khác 0; trả None nếu có giá trị 0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    """Tính MSE, RMSE, MAE, R2 và MAPE cho một bộ dự đoán.

    ``y_true`` và ``y_pred`` là mảng 1 chiều cùng độ dài.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải có cùng kích thước.")
    mse = float(mean_squared_error(y_true, y_pred))
    if np.any(y_true == 0):
        mape = None
    else:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }
