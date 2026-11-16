import matplotlib

matplotlib.use("Agg")  # Chạy headless, không bật cửa sổ đồ họa khi test

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from house_price import visualization


@pytest.fixture(autouse=True)
def close_figures():
    """Tự động đóng toàn bộ figure sau mỗi test để tránh tràn bộ nhớ."""
    yield
    plt.close("all")


def test_plot_training_history_returns_figure(tmp_path):
    history = {
        "loss": [100.0, 60.0, 35.0, 20.0],
        "val_loss": [110.0, 65.0, 40.0, 28.0],
    }
    save_file = tmp_path / "training_history.png"

    try:
        fig = visualization.plot_training_history(history, save_path=save_file)
    except TypeError:
        fig = visualization.plot_training_history(history)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    assert len(fig.axes[0].lines) >= 2  # Có cả đường train loss và val_loss


def test_plot_actual_vs_predicted_returns_figure(tmp_path):
    y_true = np.array([200_000.0, 350_000.0, 500_000.0, 750_000.0])
    y_pred = np.array([210_000.0, 340_000.0, 490_000.0, 730_000.0])
    save_file = tmp_path / "actual_vs_pred.png"

    try:
        fig = visualization.plot_actual_vs_predicted(
            y_true, y_pred, save_path=save_file
        )
    except TypeError:
        fig = visualization.plot_actual_vs_predicted(y_true, y_pred)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_actual_vs_predicted_mismatched_length_raises():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 210.0])

    with pytest.raises((ValueError, AssertionError)):
        visualization.plot_actual_vs_predicted(y_true, y_pred)