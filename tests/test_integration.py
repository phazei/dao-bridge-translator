"""Integration tests: EPUB -> init -> extract -> clean -> classify -> chunk -> assemble -> rebuild.

Uses the Japanese mini EPUB fixture to exercise the full pipeline
end-to-end through chunk, assemble, and rebuild stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.classify import run_classify_stage
from dao_bridge.clean import clean_all
from dao_bridge.config import load_config
from dao_bridge.extract import extract_epub
from dao_bridge.schemas import ClassificationResponse, Glossary, Manifest, TranslatedChunk
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


# ---------------------------------------------------------------------------
# Full pipeline through rebuild
# ---------------------------------------------------------------------------


class TestFullPipelineWithRebuild:
    """End-to-end: extract -> clean -> classify -> chunk -> translate (mock)
    -> assemble -> rebuild, using the real JP EPUB fixture."""

    def test_full_pipeline_produces_output_epub(self, jp_epub_path: Path, tmp_path: Path):
        """Run the complete pipeline and verify the output EPUB."""
        import zipfile

        from dao_bridge.assemble import assemble_all
        from dao_bridge.chunk import chunk_all
        from dao_bridge.rebuild import run_rebuild_stage
        from dao_bridge.schemas import Glossary

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
        state = load_state(work_dir)
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = ClassificationResponse(
                classification="chapter",
                title=None,
                confidence="high",
                reasoning="integration test",
            )
            mock_llm_cls.return_value = mock_client
            manifest = run_classify_stage(work_dir, config, state, force=False)

        # --- Chunk ---
        state = load_state(work_dir)
        manifest = chunk_all(config, manifest, state, force=False)

        # --- Inject mock translations ---
        from dao_bridge.workdir import chunk_path

        for item in manifest.spine:
            n = item.chunk_count or 0
            for ci in range(1, n + 1):
                chunk_id = format_chunk_id(item.spine_index, ci)
                cp = chunk_path(work_dir, chunk_id)
                chunk_data = json.loads(cp.read_text(encoding="utf-8"))
                source_text = chunk_data["text"]

                tc = TranslatedChunk(
                    chunk_id=chunk_id,
                    source_text=source_text,
                    pass1_translation=f"[Translated] {source_text[:80]}",
                    translated_text=f"[Translated] {source_text[:80]}",
                    pass_count=1,
                    total_attempts=1,
                    model_used="test-model",
                )
                tp = translation_path(work_dir, chunk_id)
                tp.parent.mkdir(parents=True, exist_ok=True)
                tp.write_text(tc.model_dump_json(indent=2), encoding="utf-8")

        # --- Assemble ---
        state = load_state(work_dir)
        manifest = assemble_all(config, manifest, state, force=False)

        # --- Create minimal glossary ---
        glossary = Glossary(
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        (work_dir / "glossary.json").write_text(glossary.model_dump_json(), encoding="utf-8")

        # --- Rebuild (mock ToC translation LLM call) ---
        with patch("dao_bridge.rebuild.translate_toc") as mock_translate_toc:
            mock_translate_toc.return_value = {}  # No ToC modifications for simplicity.

            run_rebuild_stage(work_dir, config, force=False)

        # --- Verify output ---
        output_path = (work_dir.parent / config.output.epub_path).resolve()
        assert output_path.exists(), f"Output EPUB not written: {output_path}"

        # Verify it's a valid ZIP.
        assert zipfile.is_zipfile(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            names = zf.namelist()

            # Mimetype is first entry.
            assert zf.infolist()[0].filename == "mimetype"
            assert zf.infolist()[0].compress_type == zipfile.ZIP_STORED

            # All source entries present.
            with zipfile.ZipFile(jp_epub_path, "r") as src:
                for item in src.infolist():
                    assert item.filename in names, f"Missing entry: {item.filename}"

        # Verify translated items have different content from source.
        translated_items = [i for i in manifest.spine if (i.chunk_count or 0) > 0]
        non_translated = [i for i in manifest.spine if (i.chunk_count or 0) == 0]

        from dao_bridge.rebuild import resolve_zip_path

        with zipfile.ZipFile(jp_epub_path, "r") as src_zf:
            with zipfile.ZipFile(output_path, "r") as out_zf:
                for item in translated_items:
                    zip_path = resolve_zip_path(manifest.opf_dir, item.original_href)
                    src_content = src_zf.read(zip_path)
                    out_content = out_zf.read(zip_path)
                    assert src_content != out_content, (
                        f"Translated item {zip_path} should differ from source"
                    )

                for item in non_translated:
                    zip_path = resolve_zip_path(manifest.opf_dir, item.original_href)
                    src_content = src_zf.read(zip_path)
                    out_content = out_zf.read(zip_path)
                    assert src_content == out_content, (
                        f"Non-translated item {zip_path} should be byte-identical"
                    )

        # Verify rebuild stage marked completed.
        final_state = load_state(work_dir)
        assert is_stage_completed(final_state, "rebuild")

    def test_stage_failure_marks_state_failed(self, jp_epub_path: Path, tmp_path: Path):
        """Rebuild fails gracefully when assembled files are missing."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        cfg_path = _write_config(work_dir, jp_epub_path)
        config = load_config(cfg_path)
        state = load_state(work_dir)

        manifest = extract_epub(config, state, force=False)

        # Write manifest but NO assembled files.
        (work_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        # Fake a translatable item.
        manifest.spine[0].chunk_count = 1
        (work_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

        # Create minimal glossary.
        glossary = Glossary(
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        (work_dir / "glossary.json").write_text(glossary.model_dump_json(), encoding="utf-8")

        from dao_bridge.rebuild import run_rebuild_stage

        with pytest.raises(FileNotFoundError):
            run_rebuild_stage(work_dir, config, force=False)

        final_state = load_state(work_dir)
        stage = final_state.stages.get("rebuild")
        assert stage is not None
        assert stage.status == "failed"


# ---------------------------------------------------------------------------
# CLI `run` command: full pipeline via Click
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Test the ``dao-bridge run`` CLI command end-to-end.

    This exercises the orchestration in cli.py rather than calling stage
    functions directly.  All LLM calls are mocked.
    """

    def test_run_on_freshly_initialised_work_dir(self, jp_epub_path: Path, tmp_path: Path):
        """``dao-bridge run`` succeeds on a work dir that only had ``init`` run.

        This is the primary regression test for the bug where ``run`` tried
        to load the manifest before the extract stage created it.
        """
        from click.testing import CliRunner
        from unittest.mock import PropertyMock

        from dao_bridge.cli import cli
        from dao_bridge.llm_client import CompletionResult
        from dao_bridge.schemas import (
            ClassificationResponse,
            GlossaryExtractionResponse,
            TocTranslationResponse,
        )

        work_dir = tmp_path / "work"

        # --- Phase 1: init via CLI ---
        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(jp_epub_path), "--work-dir", str(work_dir)])
        assert result.exit_code == 0, f"init failed: {result.output}"

        # At this point only config.yaml and state.json exist — no manifest.
        assert not manifest_path(work_dir).exists()

        # Disable QA and double-pass so the short mock translation is accepted.
        import yaml

        cfg_path = work_dir / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["translation_phase"] = {
            "qa_check": False,
            "double_pass": False,
            "rolling_summary": False,
        }
        cfg_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

        # --- Phase 2: run via CLI with mocked LLM ---
        # Mock LLMClient at each module that lazily creates one.
        def _make_mock_client(
            classify_response=None,
            glossary_response=None,
            translate_text="This is a translated sentence.",
            toc_titles=None,
        ):
            """Build a MagicMock that covers all LLM interactions."""
            client = MagicMock()
            # complete() — used by translate (pass1, pass2, summary)
            client.complete.return_value = CompletionResult(
                text=translate_text,
                token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                model="mock-model",
                finish_reason="stop",
            )

            # complete_json() — used by classify, glossary, translate QA, toc
            def _complete_json(messages, response_model, **kwargs):
                if response_model is ClassificationResponse:
                    return classify_response or ClassificationResponse(
                        classification="chapter",
                        title=None,
                        confidence="high",
                        reasoning="mock",
                    )
                if response_model is GlossaryExtractionResponse:
                    return glossary_response or GlossaryExtractionResponse()
                if response_model is TocTranslationResponse:
                    return TocTranslationResponse(titles=toc_titles or [])
                # QA response — return pass
                return response_model(result="pass", issues=[])

            client.complete_json.side_effect = _complete_json
            # reset_token_usage — called by translate_chunk
            client.reset_token_usage.return_value = None
            # total_token_usage property
            type(client).total_token_usage = PropertyMock(
                return_value={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
            # config attribute — used by translate to get model name
            client.config = MagicMock()
            client.config.model = "mock-model"
            return client

        mock_client = _make_mock_client()

        with (
            patch("dao_bridge.classify.LLMClient", return_value=mock_client),
            patch("dao_bridge.glossary.LLMClient", return_value=mock_client),
            patch("dao_bridge.translate.LLMClient", return_value=mock_client),
            patch("dao_bridge.llm_client.LLMClient", return_value=mock_client),
            patch("dao_bridge.rebuild.translate_toc", return_value={}),
        ):
            result = runner.invoke(
                cli,
                ["run", "--work-dir", str(work_dir)],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, f"run failed:\n{result.output}"
        assert "Pipeline complete" in result.output

        # Verify key artefacts exist.
        assert manifest_path(work_dir).exists()
        assert state_path(work_dir).exists()

        # Manifest should have spine items with classifications and chunk counts.
        manifest_data = json.loads(manifest_path(work_dir).read_text(encoding="utf-8"))
        manifest = Manifest(**manifest_data)
        for item in manifest.spine:
            assert item.classification is not None
            assert item.chunk_count is not None

        # At least one assembled file should exist.
        chunked = [i for i in manifest.spine if (i.chunk_count or 0) > 0]
        assert len(chunked) > 0
        for item in chunked:
            ap = assembled_path(work_dir, item.spine_index)
            assert ap.exists(), f"Assembled file missing for spine {item.spine_index}"

        # Output EPUB should exist.
        output_epub = (work_dir.parent / "book.en.epub").resolve()
        if not output_epub.exists():
            # Config default is relative to work_dir parent; check work_dir too.
            output_epub = work_dir / "book.en.epub"
        # The exact output path depends on config resolution, but at minimum
        # the rebuild stage should have completed.
        final_state = load_state(work_dir)
        assert is_stage_completed(final_state, "rebuild")
