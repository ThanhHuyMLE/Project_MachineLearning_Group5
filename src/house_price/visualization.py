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
    """Ve duong loss train/validation theo epoch.

    ``history`` la dict co khoa ``loss`` va ``val_loss`` (them ``mae``,
    ``val_mae`` neu co). Dung de phat hien overfitting: val loss tang
    trong khi train loss giam la dau hieu qua khop.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
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
