"""Kiểm tra lưu/nạp (persistence) cho MLP PyTorch chính thức.

Bao phủ round-trip save/load của ``MLPTorchRegressorModel``: nạp lại
mô hình đã lưu phải cho cùng dự đoán, load sai định dạng hoặc sai
kiến trúc phải báo lỗi rõ ràng, save phải tự tạo thư mục đích, và
``load()`` phải dùng ``weights_only=True``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import numpy as np
import pytest
import torch

from house_price.models.mlp_torch import MLPTorchRegressorModel


def _synthetic_data(n=80, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + 10
    return X, y


def _trained_model(tmp_path, epochs=5, save_name="mlp_torch_parameters.pt"):
    """Huấn luyện ngắn một mô hình chính thức trên dữ liệu synthetic."""
    X, y = _synthetic_data()
    save_path = tmp_path / save_name
    model = MLPTorchRegressorModel(epochs=epochs, batch_size=16, save_path=save_path)
    model.build_model(input_size=X.shape[1])
    model.fit(X, y)
    return model, X, save_path


def test_torch_mlp_round_trip_predictions_match(tmp_path):
    """Nạp lại mô hình đã lưu phải cho gần như đúng dự đoán ban đầu."""
    model, X, save_path = _trained_model(tmp_path)
    before = model.predict(X[:10])

    loaded = MLPTorchRegressorModel.load(save_path)
    after = loaded.predict(X[:10])

    max_diff = float(np.max(np.abs(before - after)))
    print(f"round-trip max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-5
    assert loaded.hidden_layer_sizes == (32, 16)
    assert loaded._input_size == X.shape[1]
    assert loaded._is_fitted
    assert len(loaded.history_["loss"]) == 5


def test_torch_mlp_json_round_trip_predictions_match(tmp_path):
    model, X, _ = _trained_model(tmp_path)
    json_path = model.export_json(
        tmp_path / "mlp_torch_parameters.json",
        feature_names=[f"x{i}" for i in range(X.shape[1])],
    )

    loaded = MLPTorchRegressorModel.load_json(json_path)

    np.testing.assert_allclose(loaded.predict(X[:10]), model.predict(X[:10]), atol=1e-6)
    assert loaded._input_size == X.shape[1]


def test_torch_mlp_load_uses_weights_only(tmp_path):
    """``load()`` phải nạp checkpoint với ``weights_only=True``."""
    source_path = (
        Path(__file__).resolve().parents[1] / "src" / "house_price" / "models" / "mlp_torch.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "weights_only=True" in source
    assert "weights_only=False" not in source

    # Checkpoint lưu từ trước (cùng định dạng: tensor + kiểu nguyên
    # thủy + dict/list) vẫn nạp được với weights_only=True: kiểm chứng
    # bằng checkpoint do chính save() hiện tại tạo ra, vì nội dung
    # checkpoint không đổi khi chỉ đổi cờ lúc load.
    _, X, save_path = _trained_model(tmp_path)
    loaded = MLPTorchRegressorModel.load(save_path)
    assert np.isfinite(loaded.predict(X[:5])).all()


def test_torch_mlp_save_creates_missing_directories(tmp_path):
    """Save phải tự tạo thư mục đích lồng nhau nếu chưa có."""
    model, X, _ = _trained_model(tmp_path)
    nested_path = tmp_path / "a" / "b" / "c" / "mlp_torch_parameters.pt"

    saved_path = model.save(nested_path)

    assert saved_path == nested_path
    assert nested_path.is_file()
    loaded = MLPTorchRegressorModel.load(nested_path)
    assert np.allclose(loaded.predict(X[:5]), model.predict(X[:5]))


def test_torch_mlp_load_rejects_wrong_format(tmp_path):
    """Load file sai định dạng phải báo lỗi rõ ràng, không lặng lẽ sai."""
    wrong_dict_path = tmp_path / "wrong_format.pt"
    torch.save({"khong_phai": "checkpoint"}, wrong_dict_path)
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        MLPTorchRegressorModel.load(wrong_dict_path)

    garbage_path = tmp_path / "garbage.pt"
    garbage_path.write_bytes(b"day khong phai la checkpoint torch")
    with pytest.raises(pickle.UnpicklingError):
        MLPTorchRegressorModel.load(garbage_path)

    with pytest.raises(FileNotFoundError):
        MLPTorchRegressorModel.load(tmp_path / "khong_ton_tai.pt")


def test_torch_mlp_load_rejects_wrong_architecture(tmp_path):
    """Load checkpoint mà state_dict không khớp kiến trúc khai báo phải lỗi."""
    _, _, save_path = _trained_model(tmp_path)
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
    # Đổi kiến trúc khai báo nhưng giữ nguyên state_dict của (32, 16):
    # load_state_dict phải báo lỗi lệch kích thước.
    checkpoint["hidden_layer_sizes"] = [8]
    mismatched_path = tmp_path / "wrong_arch.pt"
    torch.save(checkpoint, mismatched_path)

    with pytest.raises((RuntimeError, ValueError)):
        MLPTorchRegressorModel.load(mismatched_path)
