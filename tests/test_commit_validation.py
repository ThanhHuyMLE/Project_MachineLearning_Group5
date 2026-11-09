"""Kiểm tra quy tắc commit message theo docs/commit-convention.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_commits import validate_commit_message


def test_valid_commit_passes():
    assert validate_commit_message("FEAT(model): add MLP regressor wrapper") == []


def test_lowercase_type_fails():
    errors = validate_commit_message("feat(model): add MLP regressor wrapper")
    assert any("uppercase" in e for e in errors)


def test_vague_summary_fails():
    errors = validate_commit_message("FIX(model): fix bug")
    assert errors


def test_past_tense_summary_fails():
    errors = validate_commit_message("FEAT(model): added MLP regressor wrapper")
    assert any("imperative" in e for e in errors)


def test_trailing_period_fails():
    errors = validate_commit_message("DOCS(readme): update setup instructions.")
    assert any("period" in e for e in errors)
