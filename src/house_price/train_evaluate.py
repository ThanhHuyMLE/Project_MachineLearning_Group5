"""Điều phối trọn pipeline (Sinh viên 3) - entry point của dự án.

Luồng chạy: nạp dữ liệu -> chia train/validation/test -> build 4 mô hình
-> train -> predict -> đánh giá -> so sánh -> lưu outputs/metrics.csv.
File này chỉ điều phối, không chứa logic mô hình hay metric.
Chạy: ``python -m house_price.train_evaluate`` từ thư mục gốc repo.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from house_price.evaluation import regression_metrics
from house_price.model_comparison import compare_models
from house_price.models import (
    DecisionTreeRegressorModel,
    LinearRegressorModel,
    MLPRegressorModel,
    RandomForestRegressorModel,
)
from house_price.preprocessing import prepare_data

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


def build_models() -> dict[str, object]:
    """Tạo 4 mô hình theo mẫu __init__/build_model/fit/predict."""
    return {
        "MLP Regression": MLPRegressorModel(),
        "Linear Regression": LinearRegressorModel(),
        "Decision Tree Regression": DecisionTreeRegressorModel(),
        "Random Forest Regression": RandomForestRegressorModel(),
    }


def train_and_evaluate(data: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Huấn luyện từng mô hình, đánh giá trên tập test, trả bảng so sánh.

    Mô hình nào chưa cài đặt xong (NotImplementedError) sẽ được bỏ qua
    kèm cảnh báo, để pipeline vẫn chạy được trong Phase 1.
    """
    results: dict[str, dict[str, float | None]] = {}
    predictions: dict[str, object] = {}
    for name, model in build_models().items():
        try:
            model.build_model()
            model.fit(data["X_train"], data["y_train"])
        except NotImplementedError as exc:
            warnings.warn(f"Bỏ qua {name}: {exc}")
            continue
        y_pred = model.predict(data["X_test"])
        results[name] = regression_metrics(data["y_test"], y_pred)
        predictions[name] = y_pred
    table = compare_models(results)
    return table, predictions


def main(argv: list[str] | None = None) -> int:
    """Entry point: chạy pipeline và lưu kết quả vào outputs/."""
    # Console Windows mặc định dùng cp1252, không in được tiếng Việt có dấu
    # nên print()/warnings sẽ crash hoặc in lỗi font (UnicodeEncodeError).
    # Chuyển stdout/stderr sang UTF-8; trên Linux/macOS không đổi gì.
    # Lỗi được bỏ qua nếu stream không hỗ trợ reconfigure.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá 4 mô hình Hồi quy.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Đường dẫn file CSV King County")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu kết quả")
    args = parser.parse_args(argv)

    data = prepare_data(args.data)
    table, predictions = train_and_evaluate(data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "metrics.csv", index=False)
    if predictions:
        pred_frame = pd.DataFrame({"y_true": data["y_test"], **predictions})
        pred_frame.to_csv(output_dir / "predictions.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nĐã lưu kết quả vào {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
