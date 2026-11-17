import inspect
import numpy as np
import pytest

from house_price import train_evaluate
from house_price.models.mlp import MLPRegressorModel


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


def test_mlp_train_and_evaluate_end_to_end(synthetic_dataset, tmp_path):
    """Kiểm tra vòng đời build -> fit -> predict -> lưu tham số của MLPRegressorModel."""
    save_json = tmp_path / "mlp_params.json"
    model = MLPRegressorModel(
        hidden_layer_sizes=(8, 4),
        learning_rate=0.01,
        epochs=15,
        batch_size=16,
        random_state=42,
        save_path=save_json,
    )
    model.build_model(input_size=synthetic_dataset["X_train"].shape[1])
    model.fit(
        synthetic_dataset["X_train"],
        synthetic_dataset["y_train"],
        X_val=synthetic_dataset["X_val"],
        y_val=synthetic_dataset["y_val"],
    )

    preds = model.predict(synthetic_dataset["X_test"])

    assert preds.shape == synthetic_dataset["y_test"].shape
    assert np.isfinite(preds).all()
    assert save_json.exists()
    assert len(model.history_["loss"]) == 15
    assert len(model.history_["val_loss"]) == 15


def test_train_evaluate_module_has_callable_entry():
    """Xác minh module train_evaluate có hàm thực thi chính và không lỗi khi import."""
    public_callables = [
        name
        for name, obj in inspect.getmembers(train_evaluate, inspect.isfunction)
        if not name.startswith("_")
    ]
    assert len(public_callables) >= 1