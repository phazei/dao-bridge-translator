"""Tests for dao_bridge.workdir — path helpers and atomic file operations."""

from pathlib import Path

import pytest

from dao_bridge.workdir import (
    assembled_path,
    atomic_write,
    chunk_dir,
    chunk_path,
    clean_path,
    ensure_dirs,
    format_chunk_id,
    glossary_path,
    log_dir,
    manifest_path,
    pad_spine,
    parse_chunk_id,
    raw_path,
    state_path,
    summary_path,
    translation_dir,
    translation_path,
)

# ---------------------------------------------------------------------------
# pad_spine / format_chunk_id / parse_chunk_id
# ---------------------------------------------------------------------------


class TestPadSpine:
    def test_single_digit(self):
        assert pad_spine(1) == "001"

    def test_double_digit(self):
        assert pad_spine(42) == "042"

    def test_triple_digit(self):
        assert pad_spine(999) == "999"

    def test_zero(self):
        assert pad_spine(0) == "000"


class TestChunkIdRoundTrip:
    def test_basic_round_trip(self):
        cid = format_chunk_id(3, 15)
        assert cid == "003.015"
        spine, chunk = parse_chunk_id(cid)
        assert spine == 3
        assert chunk == 15

    def test_zero_indices(self):
        cid = format_chunk_id(0, 0)
        assert cid == "000.000"
        assert parse_chunk_id(cid) == (0, 0)

    def test_large_indices(self):
        cid = format_chunk_id(100, 200)
        assert cid == "100.200"
        assert parse_chunk_id(cid) == (100, 200)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid chunk_id"):
            parse_chunk_id("bad")

    def test_too_many_dots_raises(self):
        with pytest.raises(ValueError, match="Invalid chunk_id"):
            parse_chunk_id("001.002.003")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_raw_path(self, tmp_path: Path):
        assert raw_path(tmp_path, 5) == tmp_path / "raw" / "005.xhtml"

    def test_clean_path(self, tmp_path: Path):
        assert clean_path(tmp_path, 12) == tmp_path / "clean" / "012.md"

    def test_chunk_dir(self, tmp_path: Path):
        assert chunk_dir(tmp_path, 7) == tmp_path / "chunks" / "007"

    def test_chunk_path(self, tmp_path: Path):
        assert chunk_path(tmp_path, "007.003") == tmp_path / "chunks" / "007" / "007.003.json"

    def test_translation_dir(self, tmp_path: Path):
        assert translation_dir(tmp_path, 2) == tmp_path / "translations" / "002"

    def test_translation_path(self, tmp_path: Path):
        p = translation_path(tmp_path, "002.010")
        assert p == tmp_path / "translations" / "002" / "002.010.json"

    def test_assembled_path(self, tmp_path: Path):
        assert assembled_path(tmp_path, 1) == tmp_path / "assembled" / "001.md"

    def test_summary_path(self, tmp_path: Path):
        assert summary_path(tmp_path) == tmp_path / "summaries" / "rolling_summary.json"

    def test_glossary_path(self, tmp_path: Path):
        assert glossary_path(tmp_path) == tmp_path / "glossary.json"

    def test_manifest_path(self, tmp_path: Path):
        assert manifest_path(tmp_path) == tmp_path / "manifest.json"

    def test_state_path(self, tmp_path: Path):
        assert state_path(tmp_path) == tmp_path / "state.json"

    def test_log_dir(self, tmp_path: Path):
        assert log_dir(tmp_path) == tmp_path / "logs"


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------


class TestEnsureDirs:
    def test_creates_all_subdirectories(self, tmp_path: Path):
        wd = tmp_path / "mywork"
        ensure_dirs(wd)
        for name in ["raw", "clean", "chunks", "translations", "assembled", "summaries", "logs"]:
            assert (wd / name).is_dir(), f"{name} directory not created"

    def test_idempotent(self, tmp_path: Path):
        wd = tmp_path / "mywork"
        ensure_dirs(wd)
        ensure_dirs(wd)  # should not raise


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_writes_string_content(self, tmp_path: Path):
        target = tmp_path / "test.json"
        atomic_write(target, '{"key": "value"}')
        assert target.read_text(encoding="utf-8") == '{"key": "value"}'

    def test_writes_bytes_content(self, tmp_path: Path):
        target = tmp_path / "test.bin"
        data = b"\x00\x01\x02\xff"
        atomic_write(target, data)
        assert target.read_bytes() == data

    def test_overwrites_existing_file(self, tmp_path: Path):
        target = tmp_path / "test.json"
        atomic_write(target, "old")
        atomic_write(target, "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_temp_file_cleaned_up_on_success(self, tmp_path: Path):
        target = tmp_path / "test.json"
        atomic_write(target, "data")
        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists(), "Temporary file should be removed after successful write"

    def test_temp_file_cleaned_up_on_failure(self, tmp_path: Path):
        target = tmp_path / "nonexistent_dir" / "test.json"
        # Writing to a path whose parent doesn't exist will fail
        with pytest.raises(OSError):
            atomic_write(target, "data")
        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists(), "Temporary file should be removed after failed write"

    def test_utf8_content(self, tmp_path: Path):
        target = tmp_path / "test.json"
        content = '{"name": "ゼロから始める異世界生活"}'
        atomic_write(target, content)
        assert target.read_text(encoding="utf-8") == content
