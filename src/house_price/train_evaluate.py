"""Điều phối pipeline huấn luyện và đánh giá các mô hình hồi quy."""

from __future__ import annotations

import argparse
import sys
import time
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
DEFAULT_DATA = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

# Cấu hình benchmark chính thức (xem báo cáo nghiệm thu để biết số đo
# validation làm căn cứ chọn từng giá trị; không dùng test để chọn):
# - MLP: kiến trúc Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear),
#   Adam, 100 epoch, batch 32.
# - Linear: batch gradient descent viết tay, lr=0.1, tối đa 1000 epoch,
#   dừng sớm khi thay đổi tương đối của loss < tol=1e-6.
# - Decision Tree: CART viết tay, max_depth=8. Đo trên validation:
#   depth 8 vừa nhanh hơn hẳn (1.81s so với 48.31s) vừa tốt hơn
#   cây không giới hạn độ sâu (val R2 0.7198 so với 0.6926).
# - Random Forest: CART viết tay, n_estimators=100, max_depth=None,
#   max_features="sqrt". Đo trên validation (10 cây): sqrt cho
#   29.56s và val R2 0.8363, ngoại suy 100 cây khoảng 5 phút; còn
#   max_features mặc định (None) hết 276.71s cho 10 cây, ngoại suy
#   100 cây khoảng 46 phút nên không dùng, dù val R2 cao hơn
#   (0.8720). Giới hạn depth=8 cho rừng bị loại vì val R2 rơi
#   xuống 0.7839 (10 cây, sqrt).
BENCHMARK_CONFIG = {
    "MLP Regression": {"epochs": 100, "batch_size": 32},
    "Linear Regression": {
        "learning_rate": 0.1,
        "max_epochs": 1000,
        "tol": 1e-6,
    },
    "Decision Tree Regression": {"max_depth": 8},
    "Random Forest Regression": {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": "sqrt",
    },
}

MODEL_NAMES = [
    "MLP Regression",
    "Linear Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
]


def build_models(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Tạo đủ 4 mô hình với cấu hình benchmark trong BENCHMARK_CONFIG."""
    output_dir = Path(output_dir)
    mlp_config = BENCHMARK_CONFIG["MLP Regression"]
    linear_config = BENCHMARK_CONFIG["Linear Regression"]
    tree_config = BENCHMARK_CONFIG["Decision Tree Regression"]
    forest_config = BENCHMARK_CONFIG["Random Forest Regression"]

    return {
        "MLP Regression": MLPRegressorModel(
            epochs=mlp_config["epochs"],
            batch_size=mlp_config["batch_size"],
            save_path=output_dir / "mlp_parameters.json",
        ),
        "Linear Regression": LinearRegressorModel(
            learning_rate=linear_config["learning_rate"],
            max_epochs=linear_config["max_epochs"],
            tol=linear_config["tol"],
        ),
        "Decision Tree Regression": DecisionTreeRegressorModel(
            max_depth=tree_config["max_depth"],
        ),
        "Random Forest Regression": RandomForestRegressorModel(
            n_estimators=forest_config["n_estimators"],
            max_depth=forest_config["max_depth"],
            max_features=forest_config["max_features"],
        ),
    }


def train_and_evaluate(
    data: dict[str, object],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit cả 4 mô hình trên train và chỉ dùng test để đánh giá cuối.

    Mọi mô hình trong BENCHMARK_CONFIG đều bắt buộc: nếu mô hình nào
    chưa được cài đặt (NotImplementedError) thì run FAIL ngay với lỗi
    rõ ràng, không bỏ qua lặng lẽ.
    """
    output_dir = Path(output_dir)
    results: dict[str, dict[str, float | None]] = {}
    predictions: dict[str, object] = {}
    fit_seconds: dict[str, float] = {}

    models = build_models(output_dir)

    for name, model in models.items():
        fit_start = time.perf_counter()
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
            raise RuntimeError(
                f"Mô hình {name} chưa được cài đặt, benchmark dừng lại: {exc}"
            ) from exc
        fit_seconds[name] = time.perf_counter() - fit_start
        print(f"Fit {name}: {fit_seconds[name]:.2f} giây", flush=True)

        # Xuất tham số đã học cho mô hình hỗ trợ export_params (Linear).
        # MLP tự lưu tham số thành JSON trong fit qua save_path.
        export_params = getattr(model, "export_params", None)
        if callable(export_params):
            slug = name.lower().replace(" ", "_")
            feature_names = data.get("feature_columns")
            export_params(
                output_dir / f"{slug}_parameters.json",
                feature_names=feature_names,
            )

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

        # Vẽ đường cong loss nếu mô hình có lưu lịch sử huấn luyện
        history = getattr(model, "history_", None) or getattr(model, "history", None)
        if isinstance(history, dict) and "loss" in history:
            plot_training_history(
                history,
                save_path=output_dir / f"training_history_{slug}.png",
            )

    missing = [name for name in MODEL_NAMES if name not in results]
    if missing:
        raise RuntimeError("Thiếu kết quả cho mô hình bắt buộc: " + ", ".join(missing))

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
    print("Thời gian fit (giây):", flush=True)
    for name in MODEL_NAMES:
        print(f"  {name}: {fit_seconds[name]:.2f}", flush=True)
    return table, predictions


def main(argv: list[str] | None = None) -> int:
    """Chạy pipeline và ghi metric, dự đoán vào thư mục outputs."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá các mô hình hồi quy.")
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

    started = time.perf_counter()
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
    print(f"\nTổng thời gian chạy: {time.perf_counter() - started:.2f} giây")
    print(f"Đã lưu kết quả và biểu đồ vào {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
