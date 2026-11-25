import inspect

import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from house_price import train_evaluate
from house_price.evaluation import regression_metrics
from house_price.models.mlp_torch import MLPTorchRegressorModel
from house_price.train_evaluate import (
    MODEL_NAMES,
    build_models,
    train_and_evaluate,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Đóng mọi figure matplotlib sau mỗi test (train_and_evaluate vẽ nhiều biểu đồ)."""
    yield
    plt.close("all")


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6))
    true_w = np.array([3.5, -2.0, 1.2, 0.8, -1.5, 2.2])
    y = X @ true_w + 10.0 + rng.normal(scale=0.1, size=80)

    return {
        "X_train": X[:50],
        "y_train": y[:50],
        "X_val": X[50:65],
        "y_val": y[50:65],
        "X_test": X[65:],
        "y_test": y[65:],
    }


@pytest.fixture
def shifted_val_dataset(synthetic_dataset):
    """Dữ liệu giả với y_val lệch +100 so với y_test.

    Nhờ độ lệch này, metric trên val và trên test khác nhau rõ rệt
    (MSE lệch khoảng 10000), nên test phân biệt được split nào đang
    được đánh giá. y_val chỉ dùng để theo dõi val_loss khi fit MLP
    và để đánh giá, không ảnh hưởng tới việc fit trên train.
    """
    data = dict(synthetic_dataset)
    data["y_val"] = synthetic_dataset["y_val"] + 100.0
    return data


def _assert_table_matches_metrics(table, predictions, y_true):
    """Mỗi dòng của bảng phải khớp metric tính tay từ cùng dự đoán."""
    assert list(table.columns) == ["model", "mse", "rmse", "mae", "r2", "mape"]
    assert len(table) == len(MODEL_NAMES)
    for name in MODEL_NAMES:
        row = table.loc[table["model"] == name].iloc[0]
        expected = regression_metrics(y_true, predictions[name])
        for metric, value in expected.items():
            if value is None:
                assert pd.isna(row[metric])
            else:
                assert row[metric] == pytest.approx(value)


def _assert_split_figure_files(output_dir, suffix):
    """Kiểm tra đủ file biểu đồ theo split (suffix '' cho test, '_val' cho val)."""
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        assert (output_dir / f"actual_vs_predicted_{slug}{suffix}.png").is_file()
    for metric in ("rmse", "mae", "r2"):
        assert (output_dir / f"comparison_{metric}{suffix}.png").is_file()


def _install_fake_pipeline(monkeypatch, captured):
    """Thay prepare_data/train_and_evaluate bằng bản giả để test main() nhanh.

    Bản giả không fit mô hình thật, chỉ ghi lại evaluation_split mà
    main() truyền xuống và trả về một bảng metric tối thiểu.
    """

    def fake_prepare_data(path):
        return {
            "y_val": np.array([100.0, 200.0]),
            "y_test": np.array([300.0, 400.0]),
        }

    def fake_train_and_evaluate(data, output_dir, evaluation_split="val"):
        captured["evaluation_split"] = evaluation_split
        table = pd.DataFrame(
            [
                {
                    "model": "MLP Regression",
                    "mse": 1.0,
                    "rmse": 1.0,
                    "mae": 1.0,
                    "r2": 0.9,
                    "mape": 5.0,
                }
            ]
        )
        return table, {"MLP Regression": np.array([101.0, 201.0])}

    monkeypatch.setattr(train_evaluate, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(train_evaluate, "train_and_evaluate", fake_train_and_evaluate)


def test_train_evaluate_module_has_callable_entry():
    """Xác minh module train_evaluate có hàm thực thi chính và không lỗi khi import."""
    public_callables = [
        name
        for name, obj in inspect.getmembers(train_evaluate, inspect.isfunction)
        if not name.startswith("_")
    ]
    assert len(public_callables) >= 1


def test_build_models_defaults_to_four_official_models_with_pytorch_mlp(tmp_path):
    """Factory mặc định tạo đúng 4 mô hình chính thức, MLP mặc định là bản PyTorch."""
    models = build_models(tmp_path)

    assert list(models) == MODEL_NAMES
    assert len(models) == 4
    assert isinstance(models["PyTorch MLP Regression"], MLPTorchRegressorModel)


def test_train_and_evaluate_returns_official_models(synthetic_dataset, tmp_path):
    """Run trả 4 mô hình chính thức và các artifact tương ứng."""
    table, predictions = train_and_evaluate(synthetic_dataset, tmp_path)

    assert len(table) == 4
    assert set(table["model"]) == set(MODEL_NAMES)
    assert set(predictions) == set(MODEL_NAMES)
    assert (tmp_path / "mlp_torch_parameters.pt").is_file()
    for filename in (
        "pytorch_mlp_regression_parameters.json",
        "linear_regression_parameters.json",
        "decision_tree_regression_parameters.json",
        "random_forest_regression_parameters.json",
    ):
        assert (tmp_path / filename).is_file()


def test_train_and_evaluate_validation_does_not_leak_into_linear_tree_forest_fit(
    shifted_val_dataset, synthetic_dataset, tmp_path
):
    """Đổi y_val không được đổi dự đoán của Linear/DT/RF trên test.

    Ba mô hình này chỉ fit trên X_train/y_train nên validation không
    lấn vào quá trình fit của chúng; chỉ MLP dùng X_val/y_val để theo
    dõi val_loss nên dự đoán của MLP được phép khác.
    """
    _, predictions_base = train_and_evaluate(
        synthetic_dataset, tmp_path / "base", evaluation_split="test"
    )
    _, predictions_shifted = train_and_evaluate(
        shifted_val_dataset, tmp_path / "shifted", evaluation_split="test"
    )

    for name in ("Linear Regression", "Decision Tree Regression", "Random Forest Regression"):
        np.testing.assert_array_equal(predictions_base[name], predictions_shifted[name])


def test_train_and_evaluate_defaults_to_validation_split(shifted_val_dataset, tmp_path):
    """Mặc định (không truyền evaluation_split) phải đánh giá trên val."""
    table, predictions = train_and_evaluate(shifted_val_dataset, tmp_path)

    _assert_table_matches_metrics(table, predictions, shifted_val_dataset["y_val"])
    _assert_split_figure_files(tmp_path, "_val")
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        assert not (tmp_path / f"actual_vs_predicted_{slug}.png").exists()
    assert not (tmp_path / "comparison_rmse.png").exists()


def test_train_and_evaluate_with_evaluation_split_val(shifted_val_dataset, tmp_path):
    """evaluation_split='val' đánh giá trên X_val/y_val, file có hậu tố _val."""
    table, predictions = train_and_evaluate(shifted_val_dataset, tmp_path, evaluation_split="val")

    _assert_table_matches_metrics(table, predictions, shifted_val_dataset["y_val"])
    _assert_split_figure_files(tmp_path, "_val")


def test_train_and_evaluate_with_evaluation_split_test(shifted_val_dataset, tmp_path):
    """evaluation_split='test' giữ hành vi cũ: đánh giá trên X_test/y_test."""
    table, predictions = train_and_evaluate(shifted_val_dataset, tmp_path, evaluation_split="test")

    _assert_table_matches_metrics(table, predictions, shifted_val_dataset["y_test"])
    _assert_split_figure_files(tmp_path, "")
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        assert not (tmp_path / f"actual_vs_predicted_{slug}_val.png").exists()


def test_train_and_evaluate_val_and_test_metrics_differ(shifted_val_dataset, tmp_path):
    """Metric trên val và trên test phải khác rõ rệt (y_val lệch +100)."""
    table_val, _ = train_and_evaluate(shifted_val_dataset, tmp_path / "val", evaluation_split="val")
    table_test, _ = train_and_evaluate(
        shifted_val_dataset, tmp_path / "test", evaluation_split="test"
    )

    merged = table_val.set_index("model")[["mse"]].join(
        table_test.set_index("model")[["mse"]], suffixes=("_val", "_test")
    )
    assert ((merged["mse_val"] - merged["mse_test"]).abs() > 1000).all()


def test_train_and_evaluate_rejects_invalid_evaluation_split(synthetic_dataset, tmp_path):
    """evaluation_split ngoài {'val', 'test'} phải lỗi rõ ràng trước khi fit."""
    with pytest.raises(ValueError, match="evaluation_split"):
        train_and_evaluate(synthetic_dataset, tmp_path, evaluation_split="train")


def test_main_defaults_to_validation_split(monkeypatch, tmp_path, capsys):
    """main() không truyền --evaluation-split thì mặc định đánh giá trên val."""
    captured = {}
    _install_fake_pipeline(monkeypatch, captured)

    assert train_evaluate.main(["--data", "dummy.csv", "--output-dir", str(tmp_path)]) == 0
    assert captured["evaluation_split"] == "val"
    assert (tmp_path / "metrics_val.csv").is_file()
    assert (tmp_path / "predictions_val.csv").is_file()
    assert not (tmp_path / "metrics.csv").exists()
    assert not (tmp_path / "predictions.csv").exists()

    saved = pd.read_csv(tmp_path / "predictions_val.csv")
    assert list(saved["y_true"]) == [100.0, 200.0]
    assert "validation" in capsys.readouterr().out


def test_main_writes_test_split_files(monkeypatch, tmp_path, capsys):
    """main() với --evaluation-split test ghi file tên cũ và y_true từ test."""
    captured = {}
    _install_fake_pipeline(monkeypatch, captured)

    assert (
        train_evaluate.main(
            [
                "--data",
                "dummy.csv",
                "--output-dir",
                str(tmp_path),
                "--evaluation-split",
                "test",
            ]
        )
        == 0
    )
    assert captured["evaluation_split"] == "test"
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert not (tmp_path / "metrics_val.csv").exists()
    assert not (tmp_path / "predictions_val.csv").exists()

    saved = pd.read_csv(tmp_path / "predictions.csv")
    assert list(saved["y_true"]) == [300.0, 400.0]
    assert "test (final)" in capsys.readouterr().out


def test_main_rejects_invalid_evaluation_split(tmp_path):
    """main() với --evaluation-split ngoài {'val', 'test'} phải lỗi (argparse)."""
    with pytest.raises(SystemExit):
        train_evaluate.main(["--evaluation-split", "train", "--output-dir", str(tmp_path)])
