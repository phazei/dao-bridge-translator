"""Integration tests: EPUB -> init -> extract -> clean -> classify -> chunk -> assemble.

Uses the Japanese mini EPUB fixture to exercise the full pipeline
end-to-end through chunk and assemble stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dao_bridge.classify import run_classify_stage
from dao_bridge.clean import clean_all
from dao_bridge.config import load_config
from dao_bridge.extract import extract_epub
from dao_bridge.schemas import ClassificationResponse, Manifest, TranslatedChunk
from dao_bridge.state import (
    is_stage_completed,
    load_state,
)
from dao_bridge.workdir import (
    assembled_path,
    chunk_dir,
    ensure_dirs,
    format_chunk_id,
    manifest_path,
    state_path,
    translation_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(work_dir: Path, epub_path: Path) -> Path:
    """Write a minimal config.yaml and return its path."""
    import yaml

    cfg_path = work_dir / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {"source_epub": str(epub_path), "work_dir": str(work_dir)},
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Full pipeline: extract -> clean
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_extract_and_clean_jp_epub(self, jp_epub_path: Path, tmp_path: Path):
        """End-to-end: extract JP EPUB, then clean, verify output structure."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        # --- Extract ---
        manifest = extract_epub(config, state, force=False)

        # Verify raw files exist.
        assert len(manifest.spine) == 7, f"Expected 7 spine items, got {len(manifest.spine)}"
        for item in manifest.spine:
            rp = work_dir / item.raw_path
            assert rp.exists(), f"Raw file missing: {rp}"
            assert rp.suffix == ".xhtml"

        # Verify manifest persisted.
        mp = manifest_path(work_dir)
        assert mp.exists()

        # Verify images recorded.
        assert len(manifest.images) >= 1  # At least cover image

        # Verify book_id derived.
        assert manifest.book_id, "book_id should be non-empty"

        # Verify opf_dir extracted.
        assert isinstance(manifest.opf_dir, str)  # may be "" if OPF at root

        # Verify metadata.
        assert "title" in manifest.metadata
        assert "language" in manifest.metadata

        # Verify extract stage completed in state.
        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "extract")

        # --- Clean ---
        manifest = clean_all(config, manifest, state, force=False)

        # Verify clean files exist.
        for item in manifest.spine:
            assert item.clean_path is not None, f"clean_path not set for spine {item.spine_index}"
            cp = work_dir / item.clean_path
            assert cp.exists(), f"Clean file missing: {cp}"
            assert cp.suffix == ".md"

        # Verify counts populated.
        for item in manifest.spine:
            assert item.token_count is not None, f"token_count not set for spine {item.spine_index}"
            assert item.token_count > 0
            assert item.paragraph_count is not None
            assert item.paragraph_count >= 0  # Some items (e.g. cover) may have 0 paragraphs

        # Verify clean stage completed.
        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "clean")

        # Verify state.json exists and is valid.
        sp = state_path(work_dir)
        assert sp.exists()
        state_data = json.loads(sp.read_text(encoding="utf-8"))
        assert state_data["stages"]["extract"]["status"] == "completed"
        assert state_data["stages"]["clean"]["status"] == "completed"

    def test_extract_and_clean_eng_epub(self, eng_epub_path: Path, tmp_path: Path):
        """End-to-end with English EPUB — simpler content, no ruby."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, eng_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest = extract_epub(config, state, force=False)
        assert len(manifest.spine) == 7

        manifest = clean_all(config, manifest, state, force=False)
        for item in manifest.spine:
            assert item.token_count is not None
            assert item.token_count > 0


# ---------------------------------------------------------------------------
# Idempotency: re-running without --force is a no-op
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_extract_noop_without_force(self, jp_epub_path: Path, tmp_path: Path):
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        # First run.
        manifest1 = extract_epub(config, state, force=False)

        # Capture modification time of a raw file.
        rp = work_dir / manifest1.spine[0].raw_path
        mtime1 = rp.stat().st_mtime

        # Second run without force — should be a no-op.
        state2 = load_state(work_dir)
        manifest2 = extract_epub(config, state2, force=False)
        mtime2 = rp.stat().st_mtime

        # File should not have been rewritten.
        assert mtime1 == mtime2
        assert len(manifest2.spine) == len(manifest1.spine)

    def test_extract_reruns_with_force(self, jp_epub_path: Path, tmp_path: Path):
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest1 = extract_epub(config, state, force=False)
        rp = work_dir / manifest1.spine[0].raw_path
        mtime1 = rp.stat().st_mtime

        import time

        time.sleep(0.05)  # Ensure different mtime

        # With force — should re-extract.
        state2 = load_state(work_dir)
        extract_epub(config, state2, force=True)
        mtime2 = rp.stat().st_mtime

        assert mtime2 > mtime1


# ---------------------------------------------------------------------------
# Verify clean output quality with real fixture
# ---------------------------------------------------------------------------


class TestCleanOutputQuality:
    def test_ruby_text_in_jp_output(self, jp_epub_path: Path, tmp_path: Path):
        """Verify ruby annotations are converted to {base|reading} in real EPUB."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest = extract_epub(config, state)
        manifest = clean_all(config, manifest, state)

        # Read the prologue (spine index 4 = p-005.xhtml in the JP fixture).
        # Find it by looking at the original_href.
        prologue_item = None
        for item in manifest.spine:
            if "005" in item.original_href or "prologue" in item.original_href.lower():
                prologue_item = item
                break

        assert prologue_item is not None, "Could not find prologue in manifest"
        assert prologue_item.clean_path is not None
        md_content = (work_dir / prologue_item.clean_path).read_text(encoding="utf-8")

        # Should contain ruby annotations in {base|reading} format.
        assert "{" in md_content and "|" in md_content, (
            "Expected ruby annotations in {base|reading} format"
        )

        # Should NOT contain any koboSpan artefacts.
        assert "koboSpan" not in md_content
        assert "kobo." not in md_content

    def test_no_script_or_style_in_output(self, jp_epub_path: Path, tmp_path: Path):
        """Verify scripts and styles are stripped from all cleaned files."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest = extract_epub(config, state)
        manifest = clean_all(config, manifest, state)

        for item in manifest.spine:
            if item.clean_path:
                md = (work_dir / item.clean_path).read_text(encoding="utf-8")
                assert "<script" not in md.lower()
                assert "<style" not in md.lower()


# ---------------------------------------------------------------------------
# Full pipeline: extract -> clean -> classify -> chunk -> assemble
# ---------------------------------------------------------------------------


class TestChunkAssemblePipeline:
    """End-to-end: extract -> clean -> classify -> chunk -> assemble."""

    def test_full_chunk_assemble_pipeline(self, jp_epub_path: Path, tmp_path: Path):
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        # --- Extract ---
        manifest = extract_epub(config, state, force=False)

        # --- Clean ---
        manifest = clean_all(config, manifest, state, force=False)

        # --- Classify (structural hints + mocked LLM) ---
        # The LLM mock classifies anything not caught by structural hints
        # as "chapter".  Structural hints will handle cover images, TOC, etc.
        state = load_state(work_dir)
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = ClassificationResponse(
                classification="chapter",
                title=None,
                confidence="high",
                reasoning="integration test default",
            )
            mock_llm_cls.return_value = mock_client

            manifest = run_classify_stage(work_dir, config, state, force=False)

        # Verify all items have a classification set.
        for item in manifest.spine:
            assert item.classification is not None, (
                f"Spine {item.spine_index} has no classification after classify stage"
            )

        # Verify classify stage completed.
        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "classify")

        # Ensure at least one illustration was detected (cover page).
        classifications = [i.classification for i in manifest.spine]
        assert "illustration" in classifications or "frontmatter" in classifications, (
            f"Expected at least one structural hint classification, got: {classifications}"
        )

        # --- Chunk ---
        from dao_bridge.chunk import chunk_all

        state = load_state(work_dir)
        manifest = chunk_all(config, manifest, state, force=False)

        # Verify chunk stage completed.
        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "chunk")

        # Verify chunk_count set for all items.
        for item in manifest.spine:
            assert item.chunk_count is not None

        # Non-chunkable items (illustration, toc_auto) should have chunk_count=0.
        non_chunkable = [
            i for i in manifest.spine if i.classification in ("illustration", "toc_auto")
        ]
        for item in non_chunkable:
            assert item.chunk_count == 0, (
                f"Spine {item.spine_index} ({item.classification}) should have chunk_count=0"
            )

        # At least some items should have chunks.
        chunked_items = [i for i in manifest.spine if (i.chunk_count or 0) > 0]
        assert len(chunked_items) > 0

        # Verify chunk files exist.
        for item in chunked_items:
            cd = chunk_dir(work_dir, item.spine_index)
            chunk_files = list(cd.glob("*.json"))
            assert len(chunk_files) == item.chunk_count

        # --- Inject mock translations ---
        for item in manifest.spine:
            n = item.chunk_count or 0
            for ci in range(1, n + 1):
                chunk_id = format_chunk_id(item.spine_index, ci)
                # Read the source chunk to get its text.
                from dao_bridge.workdir import chunk_path

                cp = chunk_path(work_dir, chunk_id)
                chunk_data = json.loads(cp.read_text(encoding="utf-8"))
                source_text = chunk_data["text"]

                # Create a mock translation.
                tc = TranslatedChunk(
                    chunk_id=chunk_id,
                    source_text=source_text,
                    pass1_translation=f"[Translated] {source_text[:100]}...",
                    translated_text=f"[Translated] {source_text[:100]}...",
                    pass_count=1,
                    total_attempts=1,
                    model_used="test-model",
                )
                tp = translation_path(work_dir, chunk_id)
                tp.parent.mkdir(parents=True, exist_ok=True)
                tp.write_text(tc.model_dump_json(indent=2), encoding="utf-8")

        # --- Assemble ---
        from dao_bridge.assemble import assemble_all

        state = load_state(work_dir)
        manifest = assemble_all(config, manifest, state, force=False)

        # Verify assemble stage completed.
        reloaded_state = load_state(work_dir)
        assert is_stage_completed(reloaded_state, "assemble")

        # Verify assembled files exist for chunked items.
        for item in chunked_items:
            ap = assembled_path(work_dir, item.spine_index)
            assert ap.exists(), f"Assembled file missing for spine {item.spine_index}"
            content = ap.read_text(encoding="utf-8")
            assert len(content.strip()) > 0
            assert "[Translated]" in content

        # Non-chunkable items should NOT have assembled files.
        for item in non_chunkable:
            ap_nc = assembled_path(work_dir, item.spine_index)
            assert not ap_nc.exists(), (
                f"Non-chunkable spine {item.spine_index} ({item.classification}) "
                f"should not have an assembled file"
            )

    def test_chunk_idempotent_without_force(self, jp_epub_path: Path, tmp_path: Path):
        """Running chunk twice without --force is a no-op."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest = extract_epub(config, state, force=False)
        manifest = clean_all(config, manifest, state, force=False)

        # Classify using structural hints + mocked LLM.
        state = load_state(work_dir)
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = ClassificationResponse(
                classification="chapter",
                title=None,
                confidence="high",
                reasoning="test default",
            )
            mock_llm_cls.return_value = mock_client
            manifest = run_classify_stage(work_dir, config, state, force=False)

        from dao_bridge.chunk import chunk_all

        state = load_state(work_dir)
        manifest = chunk_all(config, manifest, state, force=False)
        first_counts = [i.chunk_count for i in manifest.spine]

        # Second run — should be no-op.
        state2 = load_state(work_dir)
        mp = manifest_path(work_dir)
        manifest2 = Manifest(**json.loads(mp.read_text(encoding="utf-8")))
        manifest2 = chunk_all(config, manifest2, state2, force=False)
        second_counts = [i.chunk_count for i in manifest2.spine]

        assert first_counts == second_counts
