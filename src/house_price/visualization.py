"""Bieu do danh gia Hoi quy (Sinh vien 3).

Khong sinh bieu do gia: cac ham chi ve khi duoc goi voi du doan that.
Neu truyen ``save_path`` thi luu file anh, neu khong thi tra ve figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    y_true, y_pred, title: str = "Actual vs Predicted", save_path: str | Path | None = None
):
    """Ve diem (gia that, gia du doan) kem duong ly tuong y = y_hat."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    limit = max(y_true.max(), y_pred.max())
    ax.plot([0, limit], [0, limit], color="red", linewidth=1)
    ax.set_xlabel("Gia that (USD)")
    ax.set_ylabel("Gia du doan (USD)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_training_history(history, save_path: str | Path | None = None):
    """Vẽ đường loss train/validation theo epoch để phân tích overfitting."""
    train_loss = np.asarray(history["loss"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)
    has_val = "val_loss" in history and len(history["val_loss"]) > 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Biểu đồ 1: Toàn bộ quá trình (Thang Log) ---
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
    if has_val:
        val_loss = np.asarray(history["val_loss"], dtype=float)
        ax1.plot(epochs, val_loss, label="Validation Loss (MSE)", color="#ff7f0e", linewidth=2, linestyle="--")

        best_epoch = int(np.argmin(val_loss)) + 1
        best_val = float(np.min(val_loss))
        ax1.axvline(best_epoch, color="red", linestyle=":", alpha=0.8, label=f"Best Epoch ({best_epoch})")
        ax1.scatter([best_epoch], [best_val], color="red", zorder=5)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (Log Scale)")
    ax1.set_title("Toàn bộ quá trình huấn luyện (Log Scale)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # --- Biểu đồ 2: Phóng to giai đoạn hội tụ (nửa sau số epoch) ---
    ax2 = axes[1]
    start_idx = len(epochs) // 4  # Bỏ qua 25% epoch đầu có loss quá lớn
    ax2.plot(epochs[start_idx:], train_loss[start_idx:], label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
    if has_val:
        ax2.plot(epochs[start_idx:], val_loss[start_idx:], label="Validation Loss (MSE)", color="#ff7f0e", linewidth=2, linestyle="--")
        if best_epoch > start_idx:
            ax2.axvline(best_epoch, color="red", linestyle=":", alpha=0.8, label=f"Best Epoch ({best_epoch})")
            ax2.scatter([best_epoch], [best_val], color="red", zorder=5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"Chi tiết hội tụ (Từ Epoch {start_idx + 1} đến {len(epochs)})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig

def plot_metric_comparison(
    table: pd.DataFrame, metric: str = "mse", save_path: str | Path | None = None
):
    """Ve bieu do cot so sanh mot chi so giua cac mo hinh."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(table["model"], table[metric])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model comparison by {metric.upper()}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig