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
    MLPTorchRegressorModel,
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
# validation làm căn cứ chọn từng giá trị; không dùng test để chọn).
# Số validation đã xác nhận bằng scripts/compare_validation.py
# (một lần prepare_data, seed 42, 88 cột;
# outputs/validation_comparison.csv). Thời gian fit tuyệt đối phụ
# thuộc phần cứng và tải máy nên chỉ thứ tự tương đối mới là bằng chứng.
# - MLP: kiến trúc Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear),
#   Adam, 100 epoch, batch 32. Bản NumPy cài đặt bằng NumPy; bản PyTorch
#   (yêu cầu của cô giáo, xem ADR-008) dùng cùng kiến trúc và cấu hình,
#   huấn luyện bằng torch.nn, DataLoader, MSELoss và Adam.
# - Linear: batch gradient descent viết tay, lr=0.1, tối đa 1000 epoch,
#   dừng sớm khi thay đổi tương đối của loss < tol=1e-6.
#   Đo trên validation: val R2 0.8091 (fit 1.96s).
# - Decision Tree: CART viết tay, max_depth=8. Đo trên validation:
#   depth 8 vừa nhanh hơn hẳn (0.63s so với 16.33s, khoảng 26 lần)
#   vừa tốt hơn cây không giới hạn độ sâu (val R2 0.7198 so với 0.6926).
# - Random Forest: CART viết tay, n_estimators=100, max_depth=None,
#   max_features="sqrt". Đo trên validation (10 cây): sqrt cho
#   12.80s và val R2 0.8363; còn max_features mặc định (None) hết
#   108.62s cho 10 cây, ngoại suy tuyến tính 100 cây khoảng 18 phút
#   nên không dùng, dù val R2 cao hơn (0.8720). Giới hạn depth=8 cho
#   rừng bị loại vì val R2 rơi xuống 0.7839 (10 cây, sqrt, fit 1.38s).
BENCHMARK_CONFIG = {
    "MLP Regression": {"epochs": 100, "batch_size": 32},
    "PyTorch MLP Regression": {"epochs": 100, "batch_size": 32},
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
    "PyTorch MLP Regression",
    "Linear Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
]

# Tập đánh giá -> (nhãn hiển thị, hậu tố tên file output). Split "test"
# giữ tên file cũ để tương thích ngược; split "val" thêm hậu tố "_val"
# để không trộn lẫn với kết quả test chính thức.
_EVALUATION_SPLITS = {
    "val": ("validation", "_val"),
    "test": ("test (final)", ""),
}


def _evaluation_split_info(evaluation_split: str) -> tuple[str, str]:
    """Trả (nhãn hiển thị, hậu tố tên file) cho tập đánh giá đã chọn."""
    try:
        return _EVALUATION_SPLITS[evaluation_split]
    except KeyError:
        raise ValueError(
            f"evaluation_split phải là 'val' hoặc 'test', nhận được {evaluation_split!r}."
        ) from None


def build_models(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Tạo đủ 5 mô hình với cấu hình benchmark trong BENCHMARK_CONFIG."""
    output_dir = Path(output_dir)
    mlp_config = BENCHMARK_CONFIG["MLP Regression"]
    mlp_torch_config = BENCHMARK_CONFIG["PyTorch MLP Regression"]
    linear_config = BENCHMARK_CONFIG["Linear Regression"]
    tree_config = BENCHMARK_CONFIG["Decision Tree Regression"]
    forest_config = BENCHMARK_CONFIG["Random Forest Regression"]

    return {
        "MLP Regression": MLPRegressorModel(
            epochs=mlp_config["epochs"],
            batch_size=mlp_config["batch_size"],
            save_path=output_dir / "mlp_parameters.json",
        ),
        "PyTorch MLP Regression": MLPTorchRegressorModel(
            epochs=mlp_torch_config["epochs"],
            batch_size=mlp_torch_config["batch_size"],
            save_path=output_dir / "mlp_torch_parameters.pt",
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
    evaluation_split: str = "val",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit cả 5 mô hình trên train và đánh giá trên tập đã chọn.

    Mặc định đánh giá trên validation (``evaluation_split="val"``), dùng
    khi chọn/tinh chỉnh cấu hình; truyền ``"test"`` để đánh giá cuối trên
    test. Mọi mô hình chỉ fit trên train; X_val vẫn dùng để theo dõi
    val_loss khi fit MLP.

    Mọi mô hình trong BENCHMARK_CONFIG đều bắt buộc: nếu mô hình nào
    chưa được cài đặt (NotImplementedError) thì run FAIL ngay với lỗi
    rõ ràng, không bỏ qua lặng lẽ.
    """
    split_label, suffix = _evaluation_split_info(evaluation_split)
    if evaluation_split == "val":
        if "X_val" not in data or "y_val" not in data:
            raise ValueError("evaluation_split='val' yêu cầu data có X_val/y_val.")
        X_eval, y_eval = data["X_val"], data["y_val"]
    else:
        X_eval, y_eval = data["X_test"], data["y_test"]
    print(f"Evaluation split: {split_label}", flush=True)

    output_dir = Path(output_dir)
    results: dict[str, dict[str, float | None]] = {}
    predictions: dict[str, object] = {}
    fit_seconds: dict[str, float] = {}

    models = build_models(output_dir)

    for name, model in models.items():
        fit_start = time.perf_counter()
        try:
            if isinstance(model, (MLPRegressorModel, MLPTorchRegressorModel)):
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
        # MLP NumPy tự lưu tham số thành JSON trong fit qua save_path;
        # MLP PyTorch tự lưu state_dict thành .pt trong fit qua save_path.
        export_params = getattr(model, "export_params", None)
        if callable(export_params):
            slug = name.lower().replace(" ", "_")
            feature_names = data.get("feature_columns")
            export_params(
                output_dir / f"{slug}_parameters.json",
                feature_names=feature_names,
            )

        y_pred = model.predict(X_eval)
        results[name] = regression_metrics(y_eval, y_pred)
        predictions[name] = y_pred

        # Vẽ biểu đồ Actual vs Predicted cho từng mô hình
        slug = name.lower().replace(" ", "_")
        plot_actual_vs_predicted(
            y_eval,
            y_pred,
            title=f"Actual vs Predicted - {name}",
            save_path=output_dir / f"actual_vs_predicted_{slug}{suffix}.png",
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
                    save_path=output_dir / f"comparison_{metric}{suffix}.png",
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
    parser.add_argument(
        "--evaluation-split",
        choices=["val", "test"],
        default="val",
        help="Tập dùng để đánh giá sau khi fit: 'val' (validation, mặc định, "
        "dùng khi chọn/tinh chỉnh cấu hình) hoặc 'test' (đánh giá cuối, chỉ "
        "dùng một lần sau khi đã chốt cấu hình)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_label, suffix = _evaluation_split_info(args.evaluation_split)

    started = time.perf_counter()
    data = prepare_data(args.data)
    table, predictions = train_and_evaluate(
        data, output_dir, evaluation_split=args.evaluation_split
    )

    metrics_file = output_dir / f"metrics{suffix}.csv"
    table.to_csv(metrics_file, index=False)

    if predictions:
        y_eval = data["y_val"] if args.evaluation_split == "val" else data["y_test"]
        prediction_frame = pd.DataFrame(
            {
                "y_true": y_eval,
                **predictions,
            }
        )
        prediction_frame.to_csv(
            output_dir / f"predictions{suffix}.csv",
            index=False,
        )

    print(table.to_string(index=False))
    print(f"\nTổng thời gian chạy: {time.perf_counter() - started:.2f} giây")
    print(f"Evaluation split: {split_label} (đã lưu {metrics_file.name})")
    print(f"Đã lưu kết quả và biểu đồ vào {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
