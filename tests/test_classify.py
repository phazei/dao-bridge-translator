"""Tests for dao_bridge.classify — structural hints, LLM classification, stage orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.classify import (
    apply_structural_hints,
    classify_item,
    llm_classify,
    resolve_language_name,
    run_classify_stage,
)
from dao_bridge.config import AppConfig
from dao_bridge.llm_client import LLMStructuredOutputError
from dao_bridge.schemas import ClassificationResponse, Manifest, ManifestItem
from dao_bridge.state import load_state
from dao_bridge.workdir import atomic_write, ensure_dirs, manifest_path, pad_spine

# ---------------------------------------------------------------------------
# XHTML fixtures
# ---------------------------------------------------------------------------

TOC_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>TOC</title></head>
<body>
<nav epub:type="toc">
  <ol>
    <li><a href="ch1.xhtml">Chapter 1</a></li>
    <li><a href="ch2.xhtml">Chapter 2</a></li>
  </ol>
</nav>
</body>
</html>"""

TOC_EPUB_TYPE_ON_ELEMENT_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>TOC</title></head>
<body epub:type="toc">
  <ol>
    <li><a href="ch1.xhtml">Chapter 1</a></li>
  </ol>
</body>
</html>"""

ILLUSTRATION_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title></title></head>
<body>
<div class="illust"><img src="images/cover.jpg" alt="Cover"/></div>
</body>
</html>"""

ILLUSTRATION_WITH_CAPTION_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title></title></head>
<body>
<div class="illust">
  <img src="images/insert01.jpg" alt=""/>
  <p>Color illustration</p>
</div>
</body>
</html>"""

FRONTMATTER_TITLE_ONLY_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Prologue</title></head>
<body>
<h1>Prologue</h1>
</body>
</html>"""

FRONTMATTER_LENIENT_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title></title></head>
<body>
<h1>Copyright</h1>
<p>All rights reserved.</p>
</body>
</html>"""

PROSE_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
<h1>Chapter 1: The Beginning</h1>
<p>This is a long paragraph of narrative prose content that would be found
in a typical chapter of a light novel. It continues for many sentences
and contains dialogue and description. "Hello," said the protagonist,
looking out at the vast landscape before them. The wind swept through
the tall grass, carrying with it the scent of wildflowers and rain.</p>
<p>Another paragraph follows with more content to ensure this is clearly
above the 30-word threshold for structural hint detection. The protagonist
stepped forward, determined to face whatever challenges lay ahead.</p>
</body>
</html>"""

PROSE_CLEAN_MD = """\
# Chapter 1: The Beginning

This is a long paragraph of narrative prose content that would be found
in a typical chapter of a light novel. It continues for many sentences
and contains dialogue and description.

Another paragraph follows with more content.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    work_dir: Path,
    n_items: int = 3,
    classifications: list[str | None] | None = None,
) -> Manifest:
    """Create and persist a manifest with *n_items* spine items."""
    spine = []
    for i in range(n_items):
        cls = classifications[i] if classifications else None
        item = ManifestItem(
            spine_index=i,
            original_href=f"text/{pad_spine(i)}.xhtml",
            raw_path=f"raw/{pad_spine(i)}.xhtml",
            clean_path=f"clean/{pad_spine(i)}.md",
            classification=cls,
        )
        spine.append(item)

    manifest = Manifest(
        source_epub_path=str(work_dir / "test.epub"),
        book_id="test-book",
        spine=spine,
    )
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))
    return manifest


def _write_raw_and_clean(work_dir: Path, spine_index: int, raw: str, clean: str = "") -> None:
    """Write raw XHTML and clean markdown files for a spine item."""
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{pad_spine(spine_index)}.xhtml").write_text(raw, encoding="utf-8")

    if clean:
        clean_dir = work_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        (clean_dir / f"{pad_spine(spine_index)}.md").write_text(clean, encoding="utf-8")


def _make_config(work_dir: Path) -> AppConfig:
    """Create a minimal AppConfig for testing."""
    return AppConfig(
        source_epub=str(work_dir / "test.epub"),
        work_dir=str(work_dir),
    )


def _mock_llm_response(
    classification: str = "chapter",
    title: str | None = None,
    confidence: str = "high",
    reasoning: str = "test classification",
) -> ClassificationResponse:
    """Create a ClassificationResponse for mocking complete_json."""
    return ClassificationResponse(
        classification=classification,
        title=title,
        confidence=confidence,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Language name resolution
# ---------------------------------------------------------------------------


class TestLanguageResolution:
    def test_known_code_returns_name(self):
        assert resolve_language_name("ja") == "Japanese"
        assert resolve_language_name("en") == "English"
        assert resolve_language_name("zh") == "Chinese"

    def test_unknown_code_returns_code(self):
        assert resolve_language_name("xx") == "xx"
        assert resolve_language_name("unknown") == "unknown"


# ---------------------------------------------------------------------------
# Structural hints — Layer 1
# ---------------------------------------------------------------------------


class TestStructuralHints:
    def test_toc_nav_detected(self):
        """epub:type='toc' on <nav> element produces toc_auto."""
        result = apply_structural_hints(TOC_XHTML, "")
        assert result is not None
        assert result.classification == "toc_auto"
        assert result.source == "structural"
        assert result.confidence == "high"

    def test_toc_epub_type_on_body(self):
        """epub:type='toc' on <body> element also produces toc_auto."""
        result = apply_structural_hints(TOC_EPUB_TYPE_ON_ELEMENT_XHTML, "")
        assert result is not None
        assert result.classification == "toc_auto"

    def test_illustration_detected(self):
        """XHTML with <img> and minimal text produces illustration."""
        result = apply_structural_hints(ILLUSTRATION_XHTML, "")
        assert result is not None
        assert result.classification == "illustration"
        assert result.source == "structural"
        assert result.title is None

    def test_illustration_with_short_caption(self):
        """Illustration page with a brief caption still matches."""
        result = apply_structural_hints(ILLUSTRATION_WITH_CAPTION_XHTML, "")
        assert result is not None
        assert result.classification == "illustration"

    def test_frontmatter_title_only(self):
        """XHTML with only an h1 heading produces frontmatter with title."""
        result = apply_structural_hints(FRONTMATTER_TITLE_ONLY_XHTML, "")
        assert result is not None
        assert result.classification == "frontmatter"
        assert result.title == "Prologue"
        assert result.source == "structural"

    def test_frontmatter_lenient(self):
        """XHTML with heading + small non-heading text (< 10 words) → frontmatter."""
        result = apply_structural_hints(FRONTMATTER_LENIENT_XHTML, "")
        assert result is not None
        assert result.classification == "frontmatter"
        assert result.title == "Copyright"

    def test_no_hint_on_prose(self):
        """Normal prose XHTML does not trigger any structural hint."""
        result = apply_structural_hints(PROSE_XHTML, PROSE_CLEAN_MD)
        assert result is None

    def test_no_hint_without_img(self):
        """Low word count but no <img> tag should NOT classify as illustration."""
        short_xhtml = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title></title></head>
<body><p>Short text.</p></body>
</html>"""
        result = apply_structural_hints(short_xhtml, "")
        # Should hit frontmatter hint (heading check fails, but p text < 10 words)
        # Actually no heading → won't match frontmatter either.
        # No img → won't match illustration.
        # No epub:type=toc → won't match toc.
        # This has no headings, so frontmatter hint requires headings → None.
        assert result is None


# ---------------------------------------------------------------------------
# LLM classification — Layer 2
# ---------------------------------------------------------------------------


class TestLLMClassify:
    @patch("dao_bridge.classify.LLMClient")
    def test_prompt_includes_excerpts_and_position(self, mock_llm_cls, tmp_path):
        """Verify the LLM prompt contains raw excerpt, clean excerpt, and position."""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = _mock_llm_response()

        config = _make_config(tmp_path)

        result = llm_classify(
            raw_excerpt="<html>raw content here</html>",
            clean_excerpt="# Chapter\n\nClean markdown content.",
            position=(3, 10),
            config=config,
            llm_client=mock_client,
        )

        # Verify complete_json was called.
        mock_client.complete_json.assert_called_once()
        call_args = mock_client.complete_json.call_args

        # Check messages structure.
        messages = call_args[0][0]  # first positional arg
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        prompt = messages[0]["content"]
        assert "<html>raw content here</html>" in prompt
        assert "Clean markdown content" in prompt
        assert "3" in prompt  # spine position
        assert "10" in prompt  # total items

        # Check response model (passed as keyword arg).
        assert call_args[1]["response_model"] is ClassificationResponse

        # Verify result.
        assert result.classification == "chapter"
        assert result.source == "llm"

    @patch("dao_bridge.classify.LLMClient")
    def test_llm_response_parsed_correctly(self, mock_llm_cls, tmp_path):
        """LLM response fields are correctly transferred to ClassificationResult."""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = _mock_llm_response(
            classification="frontmatter",
            title="Prologue",
            confidence="medium",
            reasoning="Appears to be a prologue section",
        )

        config = _make_config(tmp_path)
        result = llm_classify("raw", "clean", (0, 5), config, mock_client)

        assert result.classification == "frontmatter"
        assert result.title == "Prologue"
        assert result.confidence == "medium"
        assert result.reasoning == "Appears to be a prologue section"
        assert result.source == "llm"

    @patch("dao_bridge.classify.LLMClient")
    def test_prompt_includes_source_language(self, mock_llm_cls, tmp_path):
        """The prompt template should include the resolved source language name."""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = _mock_llm_response()

        config = _make_config(tmp_path)
        llm_classify("raw", "clean", (0, 5), config, mock_client)

        messages = mock_client.complete_json.call_args[0][0]
        prompt = messages[0]["content"]
        # Default config has source="ja" → "Japanese"
        assert "Japanese" in prompt


# ---------------------------------------------------------------------------
# classify_item — Layer 1 + 2 integration
# ---------------------------------------------------------------------------


class TestClassifyItem:
    def test_structural_hint_short_circuits_llm(self):
        """When structural hint matches, LLM client should not be called."""
        mock_client = MagicMock()
        item = ManifestItem(
            spine_index=0,
            original_href="text/0000.xhtml",
            raw_path="raw/0000.xhtml",
        )
        config = MagicMock(spec=AppConfig)

        result = classify_item(item, TOC_XHTML, "", (0, 5), config, mock_client)

        assert result.classification == "toc_auto"
        assert result.source == "structural"
        # LLM should NOT have been called.
        mock_client.complete_json.assert_not_called()

    def test_falls_back_to_llm_when_no_hint(self, tmp_path):
        """When no structural hint matches, LLM is called."""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = _mock_llm_response(
            classification="chapter",
            title="Chapter 1: The Beginning",
        )

        item = ManifestItem(
            spine_index=0,
            original_href="text/0000.xhtml",
            raw_path="raw/0000.xhtml",
        )
        config = _make_config(tmp_path)

        result = classify_item(item, PROSE_XHTML, PROSE_CLEAN_MD, (0, 5), config, mock_client)

        assert result.classification == "chapter"
        assert result.source == "llm"
        mock_client.complete_json.assert_called_once()


# ---------------------------------------------------------------------------
# Manual override preservation
# ---------------------------------------------------------------------------


class TestManualOverride:
    def test_existing_classification_skipped(self, tmp_path):
        """Items with existing classification are preserved without --force."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)

        # Create manifest with item 0 already classified as "backmatter".
        _make_manifest(work_dir, n_items=2, classifications=["backmatter", None])

        # Write raw/clean files for both items.
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(classification="chapter")
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        # Item 0 should still be "backmatter" (preserved).
        assert result.spine[0].classification == "backmatter"
        # Item 1 should be classified by LLM as "chapter".
        assert result.spine[1].classification == "chapter"

    def test_force_overrides_existing(self, tmp_path):
        """With --force, existing classifications are cleared and reclassified."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)

        _make_manifest(work_dir, n_items=1, classifications=["backmatter"])
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(classification="chapter")
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=True)

        # Classification should be updated to "chapter".
        assert result.spine[0].classification == "chapter"


# ---------------------------------------------------------------------------
# Force reclassification
# ---------------------------------------------------------------------------


class TestForceReclassify:
    def test_force_reclassifies_structural_hint_items(self, tmp_path):
        """--force reclassifies even items that were classified by structural hints."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)

        # Start with toc_auto classification set.
        _make_manifest(work_dir, n_items=1, classifications=["toc_auto"])
        # But write a prose XHTML file — with force, it should be reclassified.
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(classification="chapter")
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=True)

        assert result.spine[0].classification == "chapter"


# ---------------------------------------------------------------------------
# Confidence handling
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_low_confidence_logged_but_saved(self, tmp_path):
        """Items with low confidence are classified and saved normally."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=1)
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(
                classification="chapter",
                confidence="low",
                reasoning="Ambiguous content",
            )
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        # Classification should still be saved.
        assert result.spine[0].classification == "chapter"

        # Verify manifest on disk also has the classification.
        mp = manifest_path(work_dir)
        disk_manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))
        assert disk_manifest.spine[0].classification == "chapter"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_llm_failure_leaves_classification_none(self, tmp_path):
        """LLM failure should leave classification as None for retry, not crash the stage."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=2)
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            # First call fails, second succeeds.
            mock_client.complete_json.side_effect = [
                LLMStructuredOutputError("Failed after retries. Model: test"),
                _mock_llm_response(classification="chapter"),
            ]
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        # Item 0 should be None (LLM failed, retryable on re-run).
        assert result.spine[0].classification is None
        # Item 1 should be "chapter" (LLM succeeded).
        assert result.spine[1].classification == "chapter"

    def test_unexpected_error_leaves_classification_none(self, tmp_path):
        """Unexpected exceptions leave classification as None for retry."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=1)
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = RuntimeError("connection timeout")
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        assert result.spine[0].classification is None

    def test_missing_raw_file_leaves_classification_none(self, tmp_path):
        """Missing raw file should leave classification as None without crashing."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        # Create manifest but do NOT write raw files.
        _make_manifest(work_dir, n_items=1)

        state = load_state(work_dir)

        result = run_classify_stage(work_dir, config, state, force=False)

        assert result.spine[0].classification is None


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


class TestRunClassifyStage:
    def test_idempotent_without_force(self, tmp_path):
        """Running classify twice without --force skips already-classified items."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=2)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response()
            mock_llm_cls.return_value = mock_client

            # First run.
            result1 = run_classify_stage(work_dir, config, state, force=False)

        assert result1.spine[0].classification == "toc_auto"
        assert result1.spine[1].classification == "chapter"

        # Second run — should skip everything.
        state2 = load_state(work_dir)
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls2:
            mock_client2 = MagicMock()
            mock_llm_cls2.return_value = mock_client2

            result2 = run_classify_stage(work_dir, config, state2, force=False)

        # LLM should NOT have been called at all on second run.
        mock_client2.complete_json.assert_not_called()
        assert result2.spine[0].classification == "toc_auto"
        assert result2.spine[1].classification == "chapter"

    def test_state_tracking(self, tmp_path):
        """Verify state items are marked started → completed for each item."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=1)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")

        state = load_state(work_dir)
        run_classify_stage(work_dir, config, state, force=False)

        # Reload state from disk.
        final_state = load_state(work_dir)

        # Stage should be completed.
        assert final_state.stages.get("classify") is not None
        assert final_state.stages["classify"].status == "completed"

        # Item should be completed.
        assert "classify:0000" in final_state.items
        assert final_state.items["classify:0000"].status == "completed"

    def test_manifest_persisted_atomically(self, tmp_path):
        """After classification, manifest on disk should reflect all changes."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=2)
        _write_raw_and_clean(work_dir, 0, ILLUSTRATION_XHTML, "")
        _write_raw_and_clean(work_dir, 1, TOC_XHTML, "")

        state = load_state(work_dir)
        run_classify_stage(work_dir, config, state, force=False)

        # Read manifest from disk.
        mp = manifest_path(work_dir)
        disk_manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

        assert disk_manifest.spine[0].classification == "illustration"
        assert disk_manifest.spine[1].classification == "toc_auto"

    def test_spine_filter(self, tmp_path):
        """--spine N classifies only that one item."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=3)
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 1, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 2, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response()
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False, spine_filter=1)

        # Only item 1 should be classified.
        assert result.spine[1].classification == "toc_auto"
        # Items 0 and 2 should remain unclassified.
        assert result.spine[0].classification is None
        assert result.spine[2].classification is None

    def test_spine_filter_not_found(self, tmp_path):
        """--spine with invalid index raises ValueError."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=2)

        state = load_state(work_dir)

        with pytest.raises(ValueError, match="Spine index 99 not found"):
            run_classify_stage(work_dir, config, state, spine_filter=99)

    def test_on_progress_callback(self, tmp_path):
        """Progress callback is called for each item processed."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=3)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 1, ILLUSTRATION_XHTML, "")
        _write_raw_and_clean(work_dir, 2, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)
        progress_calls = []

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response()
            mock_llm_cls.return_value = mock_client

            run_classify_stage(
                work_dir,
                config,
                state,
                on_progress=lambda pid: progress_calls.append(pid),
            )

        assert len(progress_calls) == 3

    def test_mixed_structural_and_llm(self, tmp_path):
        """Stage correctly mixes structural hints and LLM classification."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=4)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")  # structural: toc_auto
        _write_raw_and_clean(work_dir, 1, ILLUSTRATION_XHTML, "")  # structural: illustration
        _write_raw_and_clean(work_dir, 2, FRONTMATTER_TITLE_ONLY_XHTML, "")  # structural: front
        _write_raw_and_clean(work_dir, 3, PROSE_XHTML, PROSE_CLEAN_MD)  # LLM: chapter

        state = load_state(work_dir)

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(
                classification="chapter",
                title="Chapter 1: The Beginning",
            )
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        assert result.spine[0].classification == "toc_auto"
        assert result.spine[1].classification == "illustration"
        assert result.spine[2].classification == "frontmatter"
        assert result.spine[2].title == "Prologue"
        assert result.spine[3].classification == "chapter"

        # LLM should have been called exactly once (for item 3).
        assert mock_client.complete_json.call_count == 1


# ---------------------------------------------------------------------------
# ClassificationResponse model validation
# ---------------------------------------------------------------------------


class TestClassificationResponse:
    def test_valid_response(self):
        resp = ClassificationResponse(
            classification="chapter",
            title="Chapter 1",
            confidence="high",
            reasoning="Main content",
        )
        assert resp.classification == "chapter"

    def test_invalid_classification_rejected(self):
        with pytest.raises(Exception):
            ClassificationResponse(
                classification="invalid_value",
                title=None,
                confidence="high",
                reasoning="test",
            )

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            ClassificationResponse(
                classification="chapter",
                title=None,
                confidence="very_high",
                reasoning="test",
            )

    def test_null_title_allowed(self):
        resp = ClassificationResponse(
            classification="illustration",
            title=None,
            confidence="high",
            reasoning="Just an image",
        )
        assert resp.title is None


# ---------------------------------------------------------------------------
# Targeted --spine and consistency auto-repair
# ---------------------------------------------------------------------------


class TestTargetedSpine:
    """Tests for --spine overriding completed state and targeted --force."""

    def test_spine_overrides_completed_state(self, tmp_path):
        """--spine N reclassifies the item even if it's already completed in state."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=3)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 2, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        # First run: classify all items.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(classification="chapter")
            mock_llm_cls.return_value = mock_client
            run_classify_stage(work_dir, config, state, force=False)

        # All items should be completed now.
        state2 = load_state(work_dir)
        assert state2.items["classify:0001"].status == "completed"

        # Now run with --spine 1 (no --force).  Should reclassify item 1.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls2:
            mock_client2 = MagicMock()
            mock_client2.complete_json.return_value = _mock_llm_response(
                classification="frontmatter"
            )
            mock_llm_cls2.return_value = mock_client2
            result = run_classify_stage(work_dir, config, state2, spine_filter=1)

        # Item 1 should be reclassified.
        assert result.spine[1].classification == "frontmatter"
        # LLM should have been called exactly once.
        assert mock_client2.complete_json.call_count == 1

    def test_spine_preserves_other_items_state(self, tmp_path):
        """--spine N does not reset state for other items."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=3)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 2, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        # First run: classify all.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response()
            mock_llm_cls.return_value = mock_client
            run_classify_stage(work_dir, config, state, force=False)

        state2 = load_state(work_dir)

        # Run with --spine 1.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls2:
            mock_client2 = MagicMock()
            mock_client2.complete_json.return_value = _mock_llm_response()
            mock_llm_cls2.return_value = mock_client2
            run_classify_stage(work_dir, config, state2, spine_filter=1)

        # Other items' state should still be completed.
        state3 = load_state(work_dir)
        assert state3.items["classify:0000"].status == "completed"
        assert state3.items["classify:0002"].status == "completed"

    def test_force_with_spine_is_targeted(self, tmp_path):
        """--force --spine N only resets the targeted item, not all items."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)
        _make_manifest(work_dir, n_items=3)
        _write_raw_and_clean(work_dir, 0, TOC_XHTML, "")
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 2, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        # First run: classify all.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response()
            mock_llm_cls.return_value = mock_client
            run_classify_stage(work_dir, config, state, force=False)

        state2 = load_state(work_dir)

        # Force reclassify only item 1.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls2:
            mock_client2 = MagicMock()
            mock_client2.complete_json.return_value = _mock_llm_response(
                classification="backmatter"
            )
            mock_llm_cls2.return_value = mock_client2
            result = run_classify_stage(work_dir, config, state2, force=True, spine_filter=1)

        # Item 1 reclassified.
        assert result.spine[1].classification == "backmatter"
        # Other items' state preserved.
        state3 = load_state(work_dir)
        assert state3.items["classify:0000"].status == "completed"
        assert state3.items["classify:0002"].status == "completed"


class TestConsistencyAutoRepair:
    """Tests for Option A: state=completed but classification=None auto-repair."""

    def test_completed_but_null_classification_is_repaired(self, tmp_path):
        """Item with state=completed but classification=None is reclassified."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)

        # Create manifest with item 0 unclassified (classification=None).
        _make_manifest(work_dir, n_items=2)
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)
        _write_raw_and_clean(work_dir, 1, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        # Manually mark item 0 as completed in state (simulating the bug).
        from dao_bridge.state import mark_item_completed, mark_stage_started

        mark_stage_started(work_dir, state, "classify")
        mark_item_completed(work_dir, state, "classify", "0000")
        mark_item_completed(work_dir, state, "classify", "0001")

        # Verify the inconsistency: state says completed, manifest says None.
        assert state.items["classify:0000"].status == "completed"

        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _mock_llm_response(classification="chapter")
            mock_llm_cls.return_value = mock_client

            result = run_classify_stage(work_dir, config, state, force=False)

        # Both items should now be classified (auto-repair triggered for both).
        assert result.spine[0].classification == "chapter"
        assert result.spine[1].classification == "chapter"

    def test_completed_with_classification_not_repaired(self, tmp_path):
        """Item with state=completed and valid classification is NOT reclassified."""
        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        config = _make_config(work_dir)

        # Item 0 has a valid classification.
        _make_manifest(work_dir, n_items=1, classifications=["backmatter"])
        _write_raw_and_clean(work_dir, 0, PROSE_XHTML, PROSE_CLEAN_MD)

        state = load_state(work_dir)

        # Classify normally.
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_llm_cls.return_value = mock_client
            run_classify_stage(work_dir, config, state, force=False)

        # On second run, item should be skipped (not reclassified).
        state2 = load_state(work_dir)
        with patch("dao_bridge.classify.LLMClient") as mock_llm_cls2:
            mock_client2 = MagicMock()
            mock_llm_cls2.return_value = mock_client2
            result = run_classify_stage(work_dir, config, state2, force=False)

        assert result.spine[0].classification == "backmatter"
        # LLM should not have been called.
        mock_client2.complete_json.assert_not_called()
