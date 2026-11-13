"""Điều phối pipeline huấn luyện và đánh giá các mô hình hồi quy."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from house_price.evaluation import regression_metrics
from house_price.model_comparison import compare_models
from house_price.models import (
    DecisionTreeRegressorModel,
    LinearRegressorModel,
    MLPRegressorModel,
    RandomForestRegressorModel,
)
from house_price.preprocessing import prepare_data
from house_price.visualization import (
    plot_actual_vs_predicted,
    plot_metric_comparison,
    plot_training_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "data"
    / "kc_house_data_NaN"
    / "data_raw"
    / "kc_house_data_NaN.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


def build_models(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Tạo các mô hình; MLP lưu tham số vào output_dir."""
    output_dir = Path(output_dir)

    return {
        "MLP Regression": MLPRegressorModel(
            save_path=output_dir / "mlp_parameters.json"
        ),
        "Linear Regression": LinearRegressorModel(),
        "Decision Tree Regression": DecisionTreeRegressorModel(),
        "Random Forest Regression": RandomForestRegressorModel(),
    }


def train_and_evaluate(
    data: dict[str, object],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit trên train và chỉ dùng test để đánh giá cuối."""
    output_dir = Path(output_dir)
    results: dict[str, dict[str, float | None]] = {}
    predictions: dict[str, object] = {}

    models = build_models(output_dir)

    for name, model in models.items():
        try:
            if isinstance(model, MLPRegressorModel):
                model.build_model(input_size=data["X_train"].shape[1])

                if "X_val" in data and "y_val" in data:
                    X_tr, y_tr = data["X_train"], data["y_train"]
                    X_val, y_val = data["X_val"], data["y_val"]
                else:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        data["X_train"],
                        data["y_train"],
                        test_size=0.15,
                        random_state=42,
                    )

                model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            else:
                model.build_model()
                model.fit(data["X_train"], data["y_train"])

        except NotImplementedError as exc:
            warnings.warn(f"Bỏ qua {name}: {exc}", stacklevel=2)
            continue

        y_pred = model.predict(data["X_test"])
        results[name] = regression_metrics(data["y_test"], y_pred)
        predictions[name] = y_pred

        # Vẽ biểu đồ Actual vs Predicted cho từng mô hình
        slug = name.lower().replace(" ", "_")
        plot_actual_vs_predicted(
            data["y_test"],
            y_pred,
            title=f"Actual vs Predicted - {name}",
            save_path=output_dir / f"actual_vs_predicted_{slug}.png",
        )

        # Vẽ đường cong loss nếu mô hình có lưu lịch sử huấn luyện (MLP)
        history = getattr(model, "history_", None) or getattr(model, "history", None)
        if isinstance(history, dict) and "loss" in history:
            plot_training_history(
                history,
                save_path=output_dir / f"training_history_{slug}.png",
            )

    table = compare_models(results)

    # Vẽ biểu đồ cột so sánh các chỉ số giữa các mô hình
    if not table.empty:
        col_map = {col.lower(): col for col in table.columns}
        for metric in ("rmse", "mae", "r2"):
            if metric in col_map:
                plot_metric_comparison(
                    table,
                    metric=col_map[metric],
                    save_path=output_dir / f"comparison_{metric}.png",
                )
    return table, predictions


def main(argv: list[str] | None = None) -> int:
    """Chạy pipeline và ghi metric, dự đoán vào thư mục outputs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá các mô hình hồi quy."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Đường dẫn file CSV King County",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục lưu metric, dự đoán và tham số mô hình",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(args.data)
    table, predictions = train_and_evaluate(data, output_dir)

    table.to_csv(output_dir / "metrics.csv", index=False)

    if predictions:
        prediction_frame = pd.DataFrame(
            {
                "y_true": data["y_test"],
                **predictions,
            }
        )
        prediction_frame.to_csv(
            output_dir / "predictions.csv",
            index=False,
        )

    print(table.to_string(index=False))
    print(f"\nĐã lưu kết quả và biểu đồ vào {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())