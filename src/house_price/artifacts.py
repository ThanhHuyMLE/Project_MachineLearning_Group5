"""Lưu và nạp bộ artifact cần thiết cho suy luận giá nhà."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ARTIFACT_FORMAT_VERSION = 1
MANIFEST_FILENAME = "inference_manifest.json"
MODEL_FILENAME = "mlp_torch_parameters.pt"
PREPROCESSOR_FILENAME = "preprocessor.joblib"


def save_inference_artifacts(data: dict[str, object], output_dir: str | Path) -> Path:
    """Lưu preprocessor và manifest sau khi pipeline huấn luyện thành công."""
    if "preprocessor" not in data:
        raise ValueError("Thiếu preprocessor trong dữ liệu đã chuẩn bị.")

    target_dir = Path(output_dir)
    model_path = target_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"Chưa có checkpoint MLP để tạo artifact: {model_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(data["preprocessor"], target_dir / PREPROCESSOR_FILENAME)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": "MLPTorchRegressorModel",
        "model_file": MODEL_FILENAME,
        "preprocessor_file": PREPROCESSOR_FILENAME,
        "target_column": "price",
        "input_columns": list(data.get("input_columns", [])),
        "feature_columns": list(data.get("feature_columns", [])),
    }
    manifest_path = target_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_manifest(artifact_dir: str | Path) -> dict[str, object]:
    """Đọc và kiểm tra manifest của artifact do chính project tạo."""
    path = Path(artifact_dir) / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Phiên bản artifact không được hỗ trợ: {manifest.get('format_version')!r}."
        )
    if manifest.get("model_type") != "MLPTorchRegressorModel":
        raise ValueError(f"Loại mô hình không được hỗ trợ: {manifest.get('model_type')!r}.")
    return manifest


def load_preprocessor(artifact_dir: str | Path, manifest: dict[str, object]):
    """Nạp preprocessor tin cậy đã được pipeline của project lưu."""
    filename = manifest.get("preprocessor_file")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Manifest thiếu preprocessor_file.")
    return joblib.load(Path(artifact_dir) / filename)
