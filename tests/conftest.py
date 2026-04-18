"""Shared pytest fixtures for dao-bridge tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# Locate the fixtures directory relative to this file.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def jp_epub_path() -> Path:
    """Path to the Japanese mini test EPUB."""
    p = _FIXTURES_DIR / "ReZero-Vol5-mini-jp.epub"
    assert p.exists(), f"Test fixture missing: {p}"
    return p


@pytest.fixture
def eng_epub_path() -> Path:
    """Path to the English mini test EPUB."""
    p = _FIXTURES_DIR / "ReZero-Vol5-mini-eng.epub"
    assert p.exists(), f"Test fixture missing: {p}"
    return p


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """A temporary work directory."""
    wd = tmp_path / "work"
    wd.mkdir()
    return wd
