from __future__ import annotations

import pandas as pd

from house_price.report import generate_report, main


def _metrics(path):
    pd.DataFrame(
        [
            {
                "model": "Model A",
                "mse": 100.0,
                "rmse": 10.0,
                "mae": 8.0,
                "r2": 0.9,
                "mape": 5.0,
            },
            {
                "model": "Model B",
                "mse": 225.0,
                "rmse": 15.0,
                "mae": 11.0,
                "r2": 0.8,
                "mape": 7.0,
            },
        ]
    ).to_csv(path, index=False)


def test_generate_report_uses_saved_metrics_without_training(tmp_path):
    metrics = tmp_path / "metrics.csv"
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    _metrics(metrics)

    generated_markdown, generated_html = generate_report(metrics, markdown, html)

    assert generated_markdown == markdown
    assert generated_html == html
    assert "Model A" in markdown.read_text(encoding="utf-8")
    assert "R2 = 0.9000" in markdown.read_text(encoding="utf-8")
    assert "<table" in html.read_text(encoding="utf-8")


def test_report_cli_writes_requested_files(tmp_path):
    metrics = tmp_path / "metrics.csv"
    markdown = tmp_path / "custom.md"
    html = tmp_path / "custom.html"
    _metrics(metrics)

    assert (
        main(
            [
                "--metrics",
                str(metrics),
                "--markdown",
                str(markdown),
                "--html",
                str(html),
            ]
        )
        == 0
    )
    assert markdown.is_file()
    assert html.is_file()
