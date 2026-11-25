"""Sinh báo cáo Markdown và HTML từ bảng metrics đã lưu."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "outputs" / "metrics.csv"
DEFAULT_MARKDOWN = REPO_ROOT / "outputs" / "report.md"
DEFAULT_HTML = REPO_ROOT / "outputs" / "report.html"
REQUIRED_COLUMNS = ["model", "mse", "rmse", "mae", "r2", "mape"]


def _format_value(column: str, value) -> str:
    if pd.isna(value):
        return "N/A"
    if column == "model":
        return str(value)
    if column == "r2":
        return f"{float(value):.4f}"
    return f"{float(value):,.2f}"


def generate_report(
    metrics_path: str | Path = DEFAULT_METRICS,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    html_path: str | Path = DEFAULT_HTML,
) -> tuple[Path, Path]:
    """Đọc metrics hiện có và ghi báo cáo, không huấn luyện lại mô hình."""
    metrics_path = Path(metrics_path)
    frame = pd.read_csv(metrics_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"File metrics thiếu cột: {missing}")
    if frame.empty:
        raise ValueError("File metrics không có kết quả mô hình.")

    frame = frame[REQUIRED_COLUMNS].sort_values("r2", ascending=False).reset_index(drop=True)
    best = frame.iloc[0]
    header = ["Mô hình", "MSE", "RMSE", "MAE", "R2", "MAPE (%)"]
    lines = [
        "# Báo cáo kết quả dự đoán giá nhà",
        "",
        f"Nguồn số liệu: `{metrics_path}`. Báo cáo chỉ đọc kết quả đã lưu, không huấn luyện lại.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [_format_value(column, value) for column, value in zip(REQUIRED_COLUMNS, row)]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            (
                f"Mô hình tốt nhất theo R2 là **{best['model']}** "
                f"(R2 = {best['r2']:.4f}, RMSE = {best['rmse']:,.2f} USD)."
            ),
            "",
            (
                "MSE, RMSE và MAE càng thấp càng tốt; R2 càng cao càng tốt. "
                "Cần đọc RMSE cùng MAE vì giá nhà có phân bố lệch phải."
            ),
        ]
    )

    markdown_path = Path(markdown_path)
    html_path = Path(html_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    display_frame = frame.rename(
        columns={
            "model": "Mô hình",
            "mse": "MSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R2",
            "mape": "MAPE (%)",
        }
    )
    html = (
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        "<title>Báo cáo dự đoán giá nhà</title><body>"
        "<h1>Báo cáo kết quả dự đoán giá nhà</h1>"
        f"<p>Nguồn số liệu: <code>{escape(str(metrics_path))}</code>.</p>"
        + display_frame.to_html(index=False, float_format=lambda value: f"{value:,.4f}")
        + f"<p>Mô hình tốt nhất theo R2 là <strong>{escape(str(best['model']))}</strong> "
        f"(R2 = {best['r2']:.4f}, RMSE = {best['rmse']:,.2f} USD).</p>"
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return markdown_path, html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh báo cáo từ outputs/metrics.csv.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    args = parser.parse_args(argv)

    markdown_path, html_path = generate_report(args.metrics, args.markdown, args.html)
    print(f"Đã tạo {markdown_path} và {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
