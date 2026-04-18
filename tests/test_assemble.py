"""Comprehensive tests for dao_bridge.assemble — chunk reassembly.

Tests use manually constructed chunk and translation JSON files
to exercise assembly logic without requiring the full pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dao_bridge.assemble import assemble_spine_item
from dao_bridge.schemas import Chunk, TranslatedChunk
from dao_bridge.workdir import (
    chunk_dir,
    chunk_path,
    format_chunk_id,
    translation_dir,
    translation_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_chunk(work_dir: Path, spine_index: int, chunk_index: int, text: str) -> Chunk:
    """Write a chunk JSON file and return the Chunk object."""
    chunk_id = format_chunk_id(spine_index, chunk_index)
    c = Chunk(
        chunk_id=chunk_id,
        spine_index=spine_index,
        chunk_index=chunk_index,
        source_file=f"clean/{spine_index:03d}.md",
        block_range=(chunk_index - 1, chunk_index - 1),  # simplified
        token_count=len(text.split()),
        text=text,
    )
    cp = chunk_path(work_dir, chunk_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(c.model_dump_json(indent=2), encoding="utf-8")
    return c


def _write_translation(
    work_dir: Path,
    spine_index: int,
    chunk_index: int,
    translated_text: str,
) -> TranslatedChunk:
    """Write a translation JSON file and return the TranslatedChunk object."""
    chunk_id = format_chunk_id(spine_index, chunk_index)
    tc = TranslatedChunk(
        chunk_id=chunk_id,
        source_text="original text",
        pass1_translation=translated_text,
        translated_text=translated_text,
        pass_count=1,
        total_attempts=1,
        model_used="test-model",
    )
    tp = translation_path(work_dir, chunk_id)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(tc.model_dump_json(indent=2), encoding="utf-8")
    return tc


def _setup_work_dir(tmp_path: Path) -> Path:
    work_dir = tmp_path / "work"
    for d in ["chunks", "translations", "assembled"]:
        (work_dir / d).mkdir(parents=True)
    return work_dir


# =========================================================================
# Multiple chunks assemble in correct order
# =========================================================================


class TestAssemblyOrder:
    def test_multiple_chunks_correct_order(self, tmp_path: Path):
        """Chunks are concatenated in chunk_index order."""
        work_dir = _setup_work_dir(tmp_path)

        _write_chunk(work_dir, 1, 1, "Source chunk 1")
        _write_chunk(work_dir, 1, 2, "Source chunk 2")
        _write_chunk(work_dir, 1, 3, "Source chunk 3")

        _write_translation(work_dir, 1, 1, "Translation one.")
        _write_translation(work_dir, 1, 2, "Translation two.")
        _write_translation(work_dir, 1, 3, "Translation three.")

        result = assemble_spine_item(work_dir, 1, 3)

        assert result == "Translation one.\n\nTranslation two.\n\nTranslation three."

    def test_out_of_order_files_still_correct(self, tmp_path: Path):
        """Chunk files on disk may be in any order — assembly sorts by chunk_index."""
        work_dir = _setup_work_dir(tmp_path)

        # Write chunks in reverse order.
        _write_chunk(work_dir, 2, 3, "Source C")
        _write_chunk(work_dir, 2, 1, "Source A")
        _write_chunk(work_dir, 2, 2, "Source B")

        _write_translation(work_dir, 2, 1, "Trans A.")
        _write_translation(work_dir, 2, 2, "Trans B.")
        _write_translation(work_dir, 2, 3, "Trans C.")

        result = assemble_spine_item(work_dir, 2, 3)
        assert result == "Trans A.\n\nTrans B.\n\nTrans C."


# =========================================================================
# Missing translation error
# =========================================================================


class TestMissingTranslation:
    def test_missing_translation_raises(self, tmp_path: Path):
        """Missing translation file raises FileNotFoundError with clear message."""
        work_dir = _setup_work_dir(tmp_path)

        _write_chunk(work_dir, 1, 1, "Source 1")
        _write_chunk(work_dir, 1, 2, "Source 2")

        # Only write translation for chunk 1, not chunk 2.
        _write_translation(work_dir, 1, 1, "Translation 1.")

        with pytest.raises(FileNotFoundError, match="Missing translations.*001\\.002"):
            assemble_spine_item(work_dir, 1, 2)

    def test_multiple_missing_translations_listed(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)

        _write_chunk(work_dir, 3, 1, "S1")
        _write_chunk(work_dir, 3, 2, "S2")
        _write_chunk(work_dir, 3, 3, "S3")

        # No translations at all.
        with pytest.raises(FileNotFoundError, match="Missing translations"):
            assemble_spine_item(work_dir, 3, 3)


# =========================================================================
# Single-chunk spine
# =========================================================================


class TestSingleChunk:
    def test_single_chunk_assembles(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)

        _write_chunk(work_dir, 5, 1, "Only source")
        _write_translation(work_dir, 5, 1, "Only translation.")

        result = assemble_spine_item(work_dir, 5, 1)
        assert result == "Only translation."


# =========================================================================
# Skipped items (chunk_count == 0)
# =========================================================================


class TestSkippedItems:
    def test_no_chunks_dir_raises(self, tmp_path: Path):
        """If chunk_count > 0 but no chunk dir exists, that's an error."""
        work_dir = _setup_work_dir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No chunk files found"):
            assemble_spine_item(work_dir, 99, 1)


# =========================================================================
# Assembled text is non-empty
# =========================================================================


class TestAssembledContent:
    def test_assembled_text_non_empty(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        _write_chunk(work_dir, 1, 1, "Source")
        _write_translation(work_dir, 1, 1, "Translated content here.")

        result = assemble_spine_item(work_dir, 1, 1)
        assert len(result.strip()) > 0

    def test_whitespace_only_translation_raises(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        _write_chunk(work_dir, 1, 1, "Source")
        _write_translation(work_dir, 1, 1, "   ")

        with pytest.raises(ValueError, match="empty"):
            assemble_spine_item(work_dir, 1, 1)
