"""Dự đoán giá cho file CSV mới bằng MLP và preprocessor đã lưu."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from house_price.artifacts import load_manifest, load_preprocessor
from house_price.models import MLPTorchRegressorModel
from house_price.preprocessing import prepare_feature_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "predictions_new.csv"


def predict_frame(frame: pd.DataFrame, artifact_dir: str | Path) -> pd.DataFrame:
    """Trả bản sao dữ liệu đầu vào có thêm cột ``predicted_price``."""
    if frame.empty:
        raise ValueError("Dữ liệu dự đoán không được rỗng.")

    artifact_dir = Path(artifact_dir)
    manifest = load_manifest(artifact_dir)
    preprocessor = load_preprocessor(artifact_dir, manifest)

    model_filename = manifest.get("model_file")
    if not isinstance(model_filename, str) or not model_filename:
        raise ValueError("Manifest thiếu model_file.")
    model = MLPTorchRegressorModel.load(artifact_dir / model_filename)

    features = prepare_feature_frame(frame)
    transformed = preprocessor.transform(features)
    predictions = model.predict(transformed)
    if not np.isfinite(predictions).all():
        raise ValueError("Mô hình tạo ra dự đoán không hữu hạn.")

    result = frame.copy()
    result["predicted_price"] = predictions
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI dự đoán một hoặc nhiều căn nhà trong file CSV."""
    parser = argparse.ArgumentParser(description="Dự đoán giá nhà từ file CSV mới.")
    parser.add_argument("--input", required=True, help="File CSV chứa các đặc trưng căn nhà")
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Thư mục chứa model, preprocessor và inference_manifest.json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="File CSV nhận kết quả dự đoán",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = pd.read_csv(input_path)
    result = predict_frame(frame, args.artifact_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(result[["predicted_price"]].to_string(index=False))
    print(f"Đã lưu {len(result)} dự đoán vào {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
