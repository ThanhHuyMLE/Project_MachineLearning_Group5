"""Controlled validation comparison for the numbers in docs/progress_results.md.

Purpose: verify (or refute) the validation numbers recorded in
docs/progress_results.md lines 56-58 and in the comments of
src/house_price/train_evaluate.py, which currently have no script, log,
or output file backing them. This script re-runs the comparison under
controlled conditions on one fixed validation split.

Method: a single call to prepare_data() produces one fixed train /
validation / test split (group-aware by house id, seed 42). Every
configuration below is fitted on the same X_train and scored with
regression_metrics() on the same X_val. Nothing here is evaluated on
the test set and nothing is selected using the test set; the existing
outputs/metrics.csv is only read back from disk and printed for
reference. Fit time is measured with time.perf_counter() around fit().

Configurations (matching the documented claims):
- Decision Tree: max_depth=8 vs max_depth=None.
- Random Forest, 10 trees: max_features="sqrt" vs max_features=None
  vs max_features="sqrt" with max_depth=8.
- Linear Regression: learning_rate=0.1, max_epochs=1000, tol=1e-6
  (the constructor parameter is named learning_rate, not lr).

Results are printed to stdout and written incrementally to
outputs/validation_comparison.csv (one row per configuration, as soon
as that configuration finishes, so a partial run still leaves usable
rows behind).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from house_price.evaluation import regression_metrics
from house_price.models.decision_tree import DecisionTreeRegressorModel
from house_price.models.linear_regression import LinearRegressorModel
from house_price.models.random_forest import RandomForestRegressorModel
from house_price.preprocessing import prepare_data

DATA_PATH = REPO_ROOT / "data" / "kc_house_data_NaN" / "data_raw" / "kc_house_data_NaN.csv"
OUTPUT_CSV = REPO_ROOT / "outputs" / "validation_comparison.csv"
METRICS_CSV = REPO_ROOT / "outputs" / "metrics.csv"

FIELDNAMES = ["model", "config", "fit_seconds", "val_mse", "val_rmse", "val_mae", "val_r2"]


def run_config(model, model_name: str, config: str, X_train, y_train, X_val, y_val, writer) -> dict:
    """Fit one model, score it on validation, print and record the row."""
    print(f"--- {model_name} [{config}] fitting ...", flush=True)
    model.build_model()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    metrics = regression_metrics(y_val, model.predict(X_val))
    row = {
        "model": model_name,
        "config": config,
        "fit_seconds": round(fit_seconds, 2),
        "val_mse": metrics["mse"],
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_r2": metrics["r2"],
    }
    writer.writerow(row)
    print(
        f"{model_name} [{config}]: fit={fit_seconds:.2f}s "
        f"val_mse={metrics['mse']:.2f} val_rmse={metrics['rmse']:.2f} "
        f"val_mae={metrics['mae']:.2f} val_r2={metrics['r2']:.4f}",
        flush=True,
    )
    return row


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    started = time.perf_counter()
    data = prepare_data(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(
        f"Split (single prepare_data, seed 42): "
        f"train={X_train.shape} val={X_val.shape} test={data['X_test'].shape}",
        flush=True,
    )
    print(f"Features: {X_train.shape[1]} columns", flush=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        handle.flush()

        # Decision Tree: max_depth=8 vs unlimited.
        rows.append(
            run_config(
                DecisionTreeRegressorModel(max_depth=8),
                "Decision Tree",
                "max_depth=8",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )
        rows.append(
            run_config(
                DecisionTreeRegressorModel(max_depth=None),
                "Decision Tree",
                "max_depth=None",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )

        # Linear Regression (fast, run before the slow forests).
        rows.append(
            run_config(
                LinearRegressorModel(learning_rate=0.1, max_epochs=1000, tol=1e-6),
                "Linear Regression",
                "lr=0.1, max_epochs=1000, tol=1e-6",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )

        # Random Forest, 10 trees: sqrt vs None vs sqrt + max_depth=8.
        # max_features=None is the slowest configuration, run it last.
        rows.append(
            run_config(
                RandomForestRegressorModel(n_estimators=10, max_features="sqrt"),
                "Random Forest",
                "n_estimators=10, max_features=sqrt",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )
        rows.append(
            run_config(
                RandomForestRegressorModel(n_estimators=10, max_features="sqrt", max_depth=8),
                "Random Forest",
                "n_estimators=10, max_features=sqrt, max_depth=8",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )
        rows.append(
            run_config(
                RandomForestRegressorModel(n_estimators=10, max_features=None),
                "Random Forest",
                "n_estimators=10, max_features=None",
                X_train,
                y_train,
                X_val,
                y_val,
                writer,
            )
        )

    print("\nValidation comparison (all on the same fixed validation split):", flush=True)
    print(
        f"{'model':<18} {'config':<48} {'fit_s':>8} {'val_r2':>8} {'val_rmse':>12} {'val_mae':>12}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['model']:<18} {row['config']:<48} {row['fit_seconds']:>8.2f} "
            f"{row['val_r2']:>8.4f} {row['val_rmse']:>12.2f} {row['val_mae']:>12.2f}",
            flush=True,
        )

    # Reference only: previously recorded TEST metrics, read from disk.
    # No test evaluation is performed by this script.
    if METRICS_CSV.is_file():
        print(
            "\nPreviously recorded test metrics (outputs/metrics.csv, read-only reference):",
            flush=True,
        )
        print(METRICS_CSV.read_text(encoding="utf-8"), flush=True)

    print(f"\nWrote {OUTPUT_CSV}", flush=True)
    print(f"Total elapsed: {time.perf_counter() - started:.2f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
