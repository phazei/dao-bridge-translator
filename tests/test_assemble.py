"""Comprehensive tests for dao_bridge.assemble — chunk reassembly.

Tests use manually constructed chunk and translation JSON files
to exercise assembly logic without requiring the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dao_bridge.assemble import assemble_all, assemble_spine_item
from dao_bridge.config import AppConfig
from dao_bridge.schemas import Chunk, Manifest, ManifestItem, TranslatedChunk
from dao_bridge.state import is_stage_completed, load_state
from dao_bridge.workdir import (
    chunk_path,
    ensure_dirs,
    format_chunk_id,
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
        source_file=f"clean/{spine_index:04d}.md",
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

        with pytest.raises(FileNotFoundError, match="Missing translations.*0001\\.002"):
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


# =========================================================================
# Stage completion gating (deferred items)
# =========================================================================


def _make_config(work_dir: Path) -> AppConfig:
    """Create a minimal AppConfig pointing at work_dir."""
    return AppConfig(source_epub="dummy.epub", work_dir=str(work_dir))


class TestDeferredItemsPreventsStageCompletion:
    """assemble_all() must NOT mark the stage complete when items are deferred."""

    def test_deferred_prevents_completion(self, tmp_path: Path):
        """Stage is not marked complete when some translations are missing."""
        work_dir = _setup_work_dir(tmp_path)
        ensure_dirs(work_dir)

        # Two spine items, both chunkable with chunk_count > 0.
        item1 = ManifestItem(
            spine_index=1,
            original_href="text/001.xhtml",
            raw_path="raw/001.xhtml",
            clean_path="clean/001.md",
            classification="chapter",
            chunk_count=1,
        )
        item2 = ManifestItem(
            spine_index=2,
            original_href="text/002.xhtml",
            raw_path="raw/002.xhtml",
            clean_path="clean/002.md",
            classification="chapter",
            chunk_count=1,
        )
        manifest = Manifest(
            source_epub_path="dummy.epub",
            book_id="test",
            spine=[item1, item2],
        )

        # Write chunks for both items.
        _write_chunk(work_dir, 1, 1, "Source text one")
        _write_chunk(work_dir, 2, 1, "Source text two")

        # Write translation ONLY for item 1 — item 2 is missing.
        _write_translation(work_dir, 1, 1, "Translated text one.")

        config = _make_config(work_dir)
        state = load_state(work_dir)

        manifest = assemble_all(config, manifest, state, force=False)

        # Item 1 should be assembled.
        from dao_bridge.workdir import assembled_path

        ap1 = assembled_path(work_dir, 1)
        assert ap1.exists()
        assert "Translated text one." in ap1.read_text(encoding="utf-8")

        # Item 2 should NOT be assembled.
        ap2 = assembled_path(work_dir, 2)
        assert not ap2.exists()

        # Stage should NOT be marked complete.
        reloaded_state = load_state(work_dir)
        assert not is_stage_completed(reloaded_state, "assemble")

    def test_all_translated_marks_complete(self, tmp_path: Path):
        """Stage IS marked complete when all items have translations."""
        work_dir = _setup_work_dir(tmp_path)
        ensure_dirs(work_dir)

        item1 = ManifestItem(
            spine_index=1,
            original_href="text/001.xhtml",
            raw_path="raw/001.xhtml",
            clean_path="clean/001.md",
            classification="chapter",
            chunk_count=1,
        )
        manifest = Manifest(
            source_epub_path="dummy.epub",
            book_id="test",
            spine=[item1],
        )

        _write_chunk(work_dir, 1, 1, "Source text")
        _write_translation(work_dir, 1, 1, "Translated text.")

        config = _make_config(work_dir)
        state = load_state(work_dir)

        manifest = assemble_all(config, manifest, state, force=False)

        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "assemble")

    def test_rerun_after_translations_added(self, tmp_path: Path):
        """After adding missing translations and re-running, stage completes."""
        work_dir = _setup_work_dir(tmp_path)
        ensure_dirs(work_dir)

        item1 = ManifestItem(
            spine_index=1,
            original_href="text/001.xhtml",
            raw_path="raw/001.xhtml",
            clean_path="clean/001.md",
            classification="chapter",
            chunk_count=1,
        )
        item2 = ManifestItem(
            spine_index=2,
            original_href="text/002.xhtml",
            raw_path="raw/002.xhtml",
            clean_path="clean/002.md",
            classification="chapter",
            chunk_count=1,
        )
        manifest = Manifest(
            source_epub_path="dummy.epub",
            book_id="test",
            spine=[item1, item2],
        )

        _write_chunk(work_dir, 1, 1, "Source one")
        _write_chunk(work_dir, 2, 1, "Source two")

        # First run: only item 1 has a translation.
        _write_translation(work_dir, 1, 1, "Trans one.")

        config = _make_config(work_dir)
        state = load_state(work_dir)
        assemble_all(config, manifest, state, force=False)

        assert not is_stage_completed(load_state(work_dir), "assemble")

        # Now add the missing translation and re-run.
        _write_translation(work_dir, 2, 1, "Trans two.")

        state2 = load_state(work_dir)
        assemble_all(config, manifest, state2, force=False)

        reloaded = load_state(work_dir)
        assert is_stage_completed(reloaded, "assemble")

        # Both assembled files should exist.
        from dao_bridge.workdir import assembled_path

        assert assembled_path(work_dir, 1).exists()
        assert assembled_path(work_dir, 2).exists()


# ---------------------------------------------------------------------------
# Targeted --spine and targeted --force
# ---------------------------------------------------------------------------


class TestTargetedSpineAssemble:
    """Tests for --spine overriding completed state and targeted --force."""

    def _make_two_item_setup(self, tmp_path: Path):
        """Create a two-item manifest with chunks and translations."""
        work_dir = _setup_work_dir(tmp_path)
        ensure_dirs(work_dir)

        item1 = ManifestItem(
            spine_index=1,
            original_href="text/001.xhtml",
            raw_path="raw/001.xhtml",
            clean_path="clean/001.md",
            classification="chapter",
            chunk_count=1,
        )
        item2 = ManifestItem(
            spine_index=2,
            original_href="text/002.xhtml",
            raw_path="raw/002.xhtml",
            clean_path="clean/002.md",
            classification="chapter",
            chunk_count=1,
        )
        manifest = Manifest(
            source_epub_path="dummy.epub",
            book_id="test",
            spine=[item1, item2],
        )

        _write_chunk(work_dir, 1, 1, "Source one")
        _write_chunk(work_dir, 2, 1, "Source two")
        _write_translation(work_dir, 1, 1, "Trans one.")
        _write_translation(work_dir, 2, 1, "Trans two.")

        config = _make_config(work_dir)
        return work_dir, manifest, config

    def test_spine_overrides_completed_state(self, tmp_path: Path):
        """--spine N reassembles even if the item is already completed."""
        work_dir, manifest, config = self._make_two_item_setup(tmp_path)

        # First run: assemble all.
        state = load_state(work_dir)
        assemble_all(config, manifest, state, force=False)

        assert is_stage_completed(load_state(work_dir), "assemble")

        # Update translation for item 1.
        _write_translation(work_dir, 1, 1, "Updated trans one.")

        # Run with --spine 1 (no --force). Should reassemble item 1.
        state2 = load_state(work_dir)
        assemble_all(config, manifest, state2, spine_filter=1)

        from dao_bridge.workdir import assembled_path

        content = assembled_path(work_dir, 1).read_text(encoding="utf-8")
        assert "Updated trans one." in content

    def test_spine_preserves_other_items_state(self, tmp_path: Path):
        """--spine N does not reset state for other items."""
        work_dir, manifest, config = self._make_two_item_setup(tmp_path)

        state = load_state(work_dir)
        assemble_all(config, manifest, state, force=False)

        # Run with --spine 1.
        state2 = load_state(work_dir)
        assemble_all(config, manifest, state2, spine_filter=1)

        # Item 2's state should still be completed.
        state3 = load_state(work_dir)
        assert state3.items["assemble:0002"].status == "completed"

    def test_force_with_spine_is_targeted(self, tmp_path: Path):
        """--force --spine N only resets the targeted item."""
        work_dir, manifest, config = self._make_two_item_setup(tmp_path)

        state = load_state(work_dir)
        assemble_all(config, manifest, state, force=False)

        # Force reassemble only item 1.
        state2 = load_state(work_dir)
        assemble_all(config, manifest, state2, force=True, spine_filter=1)

        # Item 2's state should still be completed.
        state3 = load_state(work_dir)
        assert state3.items["assemble:0002"].status == "completed"
