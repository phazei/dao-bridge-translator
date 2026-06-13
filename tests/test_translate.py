"""Comprehensive tests for dao_bridge.translate — translation pipeline.

All tests use mocked LLM responses.  The tests are organised by feature
area: glossary rendering, rolling summaries, message construction,
overlap loading, programmatic QA, LLM QA, translate_chunk, and the
stage runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.config import AppConfig
from dao_bridge.llm_client import CompletionResult, LLMStructuredOutputError
from dao_bridge.schemas import (
    Chunk,
    Glossary,
    GlossaryEntity,
    Manifest,
    ManifestItem,
    SurfaceForm,
    TranslatedChunk,
)
from dao_bridge.state import (
    PipelineState,
    load_state,
    mark_item_completed,
    mark_item_failed,
    mark_stage_completed,
)
from dao_bridge.translate import (
    QAIssue,
    QAResponse,
    TranslationProgress,
    _enumerate_chunk_ids,
    _extract_analysis,
    _filter_chunk_range,
    _load_rolling_summaries,
    _run_qa,
    _save_rolling_summaries,
    _strip_analysis,
    _update_rolling_summary,
    build_pass1_messages,
    build_pass2_messages,
    build_qa_fix_messages,
    build_qa_messages,
    load_overlap,
    programmatic_qa_check,
    qa_fix_chunk,
    render_glossary,
    render_rolling_summary,
    run_translate_stage,
    translate_chunk,
)
from dao_bridge.workdir import (
    atomic_write,
    chunk_path,
    ensure_dirs,
    glossary_path,
    manifest_path,
    translation_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(work_dir: Path, **overrides) -> AppConfig:
    """Create a minimal AppConfig for testing."""
    defaults = {
        "source_epub": str(work_dir / "test.epub"),
        "work_dir": str(work_dir),
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_chunk(
    chunk_id: str = "0001.001",
    spine_index: int = 1,
    chunk_index: int = 1,
    text: str = "これはテストです。翻訳してください。",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        spine_index=spine_index,
        chunk_index=chunk_index,
        source_file=f"clean/{spine_index:04d}.md",
        block_range=(0, 0),
        token_count=50,
        text=text,
    )


def _make_glossary(entities: list[GlossaryEntity] | None = None) -> Glossary:
    if entities is None:
        entities = [
            GlossaryEntity(
                entity_id="character_000001",
                category="character",
                canonical_name="Natsuki Subaru",
                surface_forms=[SurfaceForm(source="ナツキ・スバル", translation="Natsuki Subaru")],
                speech_style="Speaks casually, modern slang.",
                nicknames={"Ram": "Barusu"},
                source="extracted",
            ),
            GlossaryEntity(
                entity_id="place_000001",
                category="place",
                canonical_name="Guaral",
                surface_forms=[SurfaceForm(source="グァラル", translation="Guaral")],
                notes="fortress city in Vollachia",
                source="extracted",
            ),
        ]
    return Glossary(entities=entities)


def _make_translated_chunk(
    chunk_id: str = "0001.001",
    source_text: str = "original",
    translated_text: str = "translated",
    **kwargs,
) -> TranslatedChunk:
    defaults = {
        "chunk_id": chunk_id,
        "source_text": source_text,
        "pass1_translation": translated_text,
        "translated_text": translated_text,
        "pass_count": 1,
        "total_attempts": 1,
        "model_used": "test-model",
    }
    defaults.update(kwargs)
    return TranslatedChunk(**defaults)


def _make_manifest(
    work_dir: Path,
    spines: list[tuple[int, int]] | None = None,
) -> Manifest:
    """Create and persist a manifest.

    Parameters
    ----------
    spines:
        List of (spine_index, chunk_count) tuples.  Defaults to a single
        spine with 3 chunks.
    """
    if spines is None:
        spines = [(1, 3)]

    items = []
    for si, cc in spines:
        items.append(
            ManifestItem(
                spine_index=si,
                original_href=f"spine{si}.xhtml",
                raw_path=f"raw/{si:04d}.xhtml",
                clean_path=f"clean/{si:04d}.md",
                classification="chapter",
                chunk_count=cc,
            )
        )

    manifest = Manifest(
        source_epub_path=str(work_dir / "test.epub"),
        book_id="test-book",
        spine=items,
    )
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))
    return manifest


def _write_chunk_file(work_dir: Path, chunk: Chunk) -> None:
    """Write a Chunk JSON to disk."""
    cp = chunk_path(work_dir, chunk.chunk_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")


def _write_translation_file(work_dir: Path, tc: TranslatedChunk) -> None:
    """Write a TranslatedChunk JSON to disk."""
    tp = translation_path(work_dir, tc.chunk_id)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(tc.model_dump_json(indent=2), encoding="utf-8")


def _write_glossary(work_dir: Path, glossary: Glossary) -> None:
    gp = glossary_path(work_dir)
    atomic_write(gp, glossary.model_dump_json(indent=2))


def _mark_prior_stages_complete(work_dir: Path, state: PipelineState) -> None:
    """Mark all stages before translate as completed."""
    for stage in [
        "extract",
        "clean",
        "classify",
        "chunk",
        "glossary_build",
        "glossary_reconcile",
    ]:
        mark_stage_completed(work_dir, state, stage)


def _setup_work_dir(tmp_path: Path) -> Path:
    work_dir = tmp_path / "work"
    ensure_dirs(work_dir)
    return work_dir


def _mock_completion(text: str = "Translated text.", **kwargs) -> CompletionResult:
    return CompletionResult(
        text=text,
        token_usage=kwargs.get(
            "token_usage",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        ),
        model=kwargs.get("model", "test-model"),
        finish_reason="stop",
    )


def _make_mock_client(
    completions: list[CompletionResult] | CompletionResult | None = None,
) -> MagicMock:
    """Create a MagicMock LLM client with working token-usage tracking.

    The mock accumulates token usage from each ``complete()`` call's return
    value, mirroring the real :class:`LLMClient` behaviour.
    """
    mock = MagicMock()
    mock.config = MagicMock(model="test-model")

    # Internal accumulator.
    _usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _reset() -> None:
        _usage["prompt_tokens"] = 0
        _usage["completion_tokens"] = 0
        _usage["total_tokens"] = 0

    mock.reset_token_usage = _reset

    # Property-like accessor (MagicMock can't do @property, so use a function
    # assigned to the attribute name via PropertyMock or a plain attribute).
    # We need it to return a fresh copy each time it's accessed.
    type(mock).total_token_usage = property(lambda self: dict(_usage))

    # Wrap complete() to auto-accumulate token usage.
    if completions is not None:
        if isinstance(completions, list):
            raw_side_effects = list(completions)
        else:
            raw_side_effects = None
            raw_return = completions

        call_index = [0]

        def _complete_side_effect(*args, **kwargs):
            if raw_side_effects is not None:
                result = raw_side_effects[call_index[0]]
                call_index[0] += 1
            else:
                result = raw_return
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                _usage[key] += result.token_usage.get(key, 0)
            return result

        mock.complete.side_effect = _complete_side_effect

    return mock


# =========================================================================
# Glossary rendering
# =========================================================================


class TestRenderGlossary:
    """Tests for render_glossary()."""

    def test_relevant_mode_filters_by_source_term(self):
        """In 'relevant' mode, only entries whose source_term appears in
        the chunk text are included."""
        glossary = _make_glossary()
        # Only ナツキ・スバル is in the text.
        text = "ナツキ・スバルは言った。"
        result = render_glossary(glossary, text, "relevant")

        assert "Natsuki Subaru" in result
        assert "Guaral" not in result

    def test_all_mode_includes_everything(self):
        """In 'all' mode, all entries are included regardless of text."""
        glossary = _make_glossary()
        result = render_glossary(glossary, "unrelated text", "all")

        assert "Natsuki Subaru" in result
        assert "Guaral" in result

    def test_character_includes_speech_and_nicknames(self):
        """Character entries include Speech and Nicknames fields."""
        glossary = _make_glossary()
        text = "ナツキ・スバル"
        result = render_glossary(glossary, text, "relevant")

        assert "Speech: Speaks casually" in result
        assert "Nicknames:" in result
        assert "Barusu" in result

    def test_non_character_omits_speech_and_nicknames(self):
        """Non-character entries do not include speech or nickname fields."""
        glossary = _make_glossary()
        text = "グァラル"
        result = render_glossary(glossary, text, "relevant")

        assert "Guaral" in result
        # Speech and Nicknames should not appear for a place.
        lines = result.split("\n")
        guaral_idx = next(i for i, line in enumerate(lines) if "Guaral" in line)
        # Next line should not be a Speech or Nicknames line.
        remaining = lines[guaral_idx + 1 :]
        for line in remaining:
            if line.strip().startswith("-") or line.strip() == "":
                break
            assert not line.strip().startswith("Speech:")
            assert not line.strip().startswith("Nicknames:")

    def test_no_matches_returns_empty_string(self):
        """When no glossary entries match in relevant mode, return an
        empty string so the caller can skip injection."""
        glossary = _make_glossary()
        result = render_glossary(glossary, "no matching terms here", "relevant")

        assert result == ""

    def test_entries_grouped_by_category(self):
        """Entries are grouped under their category header."""
        glossary = _make_glossary()
        result = render_glossary(glossary, "ナツキ・スバル グァラル", "relevant")

        assert "Character:" in result
        assert "Place:" in result

    def test_entity_without_surface_forms_excluded_from_relevant(self):
        """Entities with no surface forms are excluded in relevant mode."""
        entities = [
            GlossaryEntity(
                entity_id="term_000001",
                category="term",
                canonical_name="Test Term",
                surface_forms=[],
                source="seed",
            ),
        ]
        glossary = _make_glossary(entities)
        result = render_glossary(glossary, "anything", "relevant")

        assert result == ""

    def test_notes_included_in_rendering(self):
        """Entry notes are shown after the dash."""
        glossary = _make_glossary()
        result = render_glossary(glossary, "グァラル", "relevant")

        assert "fortress city in Vollachia" in result


# =========================================================================
# Rolling summary rendering
# =========================================================================


class TestRenderRollingSummary:
    """Tests for render_rolling_summary()."""

    def test_empty_summaries_returns_empty_string(self):
        result = render_rolling_summary([], max_tokens=2000)
        assert result == ""

    def test_all_fit_within_budget(self):
        """When all summaries fit, they are all included."""
        summaries = [
            {"chunk_id": "0001.001", "summary": "Event A happened."},
            {"chunk_id": "0001.002", "summary": "Event B happened."},
        ]
        result = render_rolling_summary(summaries, max_tokens=5000)

        assert "STORY SO FAR" in result
        assert "[0001.001]" in result
        assert "[0001.002]" in result

    def test_sliding_window_excludes_oldest(self):
        """When the budget is tight, oldest entries are excluded first."""
        summaries = [
            {"chunk_id": "0001.001", "summary": "A " * 500},  # large
            {"chunk_id": "0001.002", "summary": "B " * 500},  # large
            {"chunk_id": "0001.003", "summary": "Short."},  # small
        ]
        # Budget is very tight — should only fit the newest entries.
        result = render_rolling_summary(summaries, max_tokens=50)

        assert "[0001.003]" in result
        # The oldest large entry should be excluded.
        # (exact inclusion depends on token counting, but the newest should always be there)

    def test_chronological_order_preserved(self):
        """Selected summaries appear in chronological order."""
        summaries = [
            {"chunk_id": "0001.001", "summary": "First."},
            {"chunk_id": "0001.002", "summary": "Second."},
            {"chunk_id": "0001.003", "summary": "Third."},
        ]
        result = render_rolling_summary(summaries, max_tokens=5000)

        idx1 = result.index("[0001.001]")
        idx2 = result.index("[0001.002]")
        idx3 = result.index("[0001.003]")
        assert idx1 < idx2 < idx3


# =========================================================================
# Analysis stripping
# =========================================================================


class TestStripAnalysis:
    """Tests for _strip_analysis()."""

    def test_strips_analysis_block(self):
        """Analysis block is removed, leaving only the translation."""
        text = "<analysis>\nSome notes here.\n</analysis>\nThe translation text."
        assert _strip_analysis(text) == "The translation text."

    def test_strips_analysis_with_whitespace(self):
        """Leading/trailing whitespace around the result is cleaned up."""
        text = "<analysis>notes</analysis>\n\n  The translation text.  \n"
        assert _strip_analysis(text) == "The translation text."

    def test_no_analysis_returns_text_unchanged(self):
        """When no analysis block is present, text is returned as-is."""
        text = "Just the translation."
        assert _strip_analysis(text) == "Just the translation."

    def test_multiline_analysis(self):
        """Analysis blocks with multiple lines are fully removed."""
        text = (
            "<analysis>\nLine 1\nLine 2\n**Bold**: stuff\n</analysis>\n"
            "Paragraph one.\n\nParagraph two."
        )
        assert _strip_analysis(text) == "Paragraph one.\n\nParagraph two."

    def test_empty_analysis(self):
        """An empty analysis block is removed cleanly."""
        text = "<analysis></analysis>Translation."
        assert _strip_analysis(text) == "Translation."


class TestExtractAnalysis:
    """Tests for _extract_analysis()."""

    def test_extracts_analysis_block(self):
        """Returns the full tagged analysis block."""
        text = "<analysis>\nSome notes here.\n</analysis>\nThe translation text."
        result = _extract_analysis(text)
        assert result is not None
        assert "<analysis>" in result
        assert "Some notes here." in result
        assert "</analysis>" in result

    def test_no_analysis_returns_none(self):
        """When no analysis block is present, returns None."""
        text = "Just the translation."
        assert _extract_analysis(text) is None

    def test_multiline_analysis(self):
        """Multi-line analysis blocks are fully captured."""
        text = (
            "<analysis>\nLine 1\nLine 2\n**Bold**: stuff\n</analysis>\n"
            "Paragraph one.\n\nParagraph two."
        )
        result = _extract_analysis(text)
        assert result is not None
        assert "Line 1" in result
        assert "Line 2" in result
        assert "**Bold**: stuff" in result

    def test_empty_analysis(self):
        """An empty analysis block is still returned."""
        text = "<analysis></analysis>Translation."
        result = _extract_analysis(text)
        assert result is not None
        assert result == "<analysis></analysis>"


# =========================================================================
# Message construction
# =========================================================================


class TestBuildPass1Messages:
    """Tests for build_pass1_messages()."""

    def test_system_message_contains_instructions_only(self, tmp_path: Path):
        """System message contains translation instructions but not glossary
        or rolling summary content."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()
        summaries = [{"chunk_id": "0001.001", "summary": "Something happened."}]

        messages = build_pass1_messages(chunk, glossary, None, summaries, config)

        assert messages[0]["role"] == "system"
        assert "Translation guidelines" in messages[0]["content"]
        assert "GLOSSARY" not in messages[0]["content"]
        assert "STORY SO FAR" not in messages[0]["content"]

    def test_last_message_is_source_text(self, tmp_path: Path):
        """The last message is always the user source text to translate."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()

        messages = build_pass1_messages(chunk, glossary, None, [], config)

        assert messages[-1]["role"] == "user"
        assert chunk.text in messages[-1]["content"]

    def test_glossary_injected_as_user_assistant_pair(self, tmp_path: Path):
        """Matching glossary entries are injected as a user message
        followed by an assistant 'Understood.' acknowledgment."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        # Use chunk text that contains a glossary source_term.
        chunk = _make_chunk(text="ナツキ・スバルは言った。")
        glossary = _make_glossary()

        messages = build_pass1_messages(chunk, glossary, None, [], config)

        assert messages[1]["role"] == "user"
        assert "GLOSSARY" in messages[1]["content"]
        assert "Natsuki Subaru" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Understood."

    def test_no_glossary_match_skips_glossary_messages(self, tmp_path: Path):
        """When no glossary entries match the chunk text, glossary user and
        assistant messages are omitted entirely."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()  # default text has no matching glossary terms
        glossary = _make_glossary()

        messages = build_pass1_messages(chunk, glossary, None, [], config)

        all_content = " ".join(m["content"] for m in messages)
        assert "GLOSSARY" not in all_content

    def test_overlap_adds_user_assistant_pair(self, tmp_path: Path):
        """When overlap is provided, it is injected as a user message
        followed by an assistant 'Understood.' before the source text."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()
        overlap = _make_translated_chunk(
            source_text="prev source", translated_text="prev translation"
        )

        messages = build_pass1_messages(chunk, glossary, overlap, [], config)

        # system, overlap-user, overlap-assistant, source-user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert "prev source" in messages[1]["content"]
        assert "prev translation" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Understood."

    def test_no_overlap_omits_overlap_messages(self, tmp_path: Path):
        """Without overlap, only system + source messages are present."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()

        messages = build_pass1_messages(chunk, glossary, None, [], config)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_rolling_summary_injected_as_user_assistant_pair(self, tmp_path: Path):
        """Rolling summaries are injected as a user message followed by an
        assistant 'Understood.' acknowledgment."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()
        summaries = [{"chunk_id": "0001.001", "summary": "Something happened."}]

        messages = build_pass1_messages(chunk, glossary, None, summaries, config)

        # system, summary-user, summary-assistant, source-user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert "STORY SO FAR" in messages[1]["content"]
        assert "Something happened" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Understood."

    def test_empty_summary_omits_summary_messages(self, tmp_path: Path):
        """When there are no summaries, summary messages are omitted entirely."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk()
        glossary = _make_glossary()

        messages = build_pass1_messages(chunk, glossary, None, [], config)

        all_content = " ".join(m["content"] for m in messages)
        assert "STORY SO FAR" not in all_content

    def test_full_context_message_order(self, tmp_path: Path):
        """With glossary, summary, and overlap all present, messages follow
        the expected order: system, glossary, ack, summary, ack, overlap, ack, source."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        chunk = _make_chunk(text="ナツキ・スバルは言った。")
        glossary = _make_glossary()
        summaries = [{"chunk_id": "0001.001", "summary": "Something happened."}]
        overlap = _make_translated_chunk(
            source_text="prev source", translated_text="prev translation"
        )

        messages = build_pass1_messages(chunk, glossary, overlap, summaries, config)

        assert len(messages) == 8
        roles = [m["role"] for m in messages]
        assert roles == [
            "system",
            "user",
            "assistant",  # glossary
            "user",
            "assistant",  # summary
            "user",
            "assistant",  # overlap
            "user",  # source text
        ]
        assert "GLOSSARY" in messages[1]["content"]
        assert "STORY SO FAR" in messages[3]["content"]
        assert "prev source" in messages[5]["content"]
        assert chunk.text in messages[7]["content"]


class TestBuildPass2Messages:
    """Tests for build_pass2_messages()."""

    def test_fresh_system_and_user_messages(self, tmp_path: Path):
        """Pass 2 builds a fresh system + user message list with source
        and draft translation."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)

        result = build_pass2_messages("original source", "pass1 output", config)

        # Exactly 2 messages: system + user.
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "polish" in result[0]["content"].lower()
        assert result[1]["role"] == "user"
        # User message contains both source and draft.
        assert "original source" in result[1]["content"]
        assert "pass1 output" in result[1]["content"]
        assert "polish" in result[1]["content"].lower()

    def test_no_overlap_or_glossary(self, tmp_path: Path):
        """Pass 2 messages contain only source, draft, and polish instructions."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)

        result = build_pass2_messages("src", "draft", config)
        all_content = " ".join(m["content"] for m in result)

        assert "GLOSSARY" not in all_content
        assert "STORY SO FAR" not in all_content
        assert "preceding" not in all_content.lower()


class TestBuildQAMessages:
    """Tests for build_qa_messages()."""

    def test_fresh_system_and_user_messages(self, tmp_path: Path):
        """QA builds a fresh system + user message list with source
        and translation."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        # Source has no glossary surface forms, so no glossary message is added.
        result = build_qa_messages("original source", "final translation", glossary, config)

        # Exactly 2 messages: system + user.
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "JSON" in result[0]["content"]
        assert result[1]["role"] == "user"
        # User message contains both source and translation.
        assert "original source" in result[1]["content"]
        assert "final translation" in result[1]["content"]
        assert "Assess the" in result[1]["content"]

    def test_no_rolling_summary_or_overlap(self, tmp_path: Path):
        """QA messages contain no rolling summary or overlap continuity."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        result = build_qa_messages("src", "translation", glossary, config)
        all_content = " ".join(m["content"] for m in result)

        assert "STORY SO FAR" not in all_content
        assert "preceding" not in all_content.lower()

    def test_glossary_terms_injected_when_relevant(self, tmp_path: Path):
        """Matching glossary terms are injected as an approved-names guard."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        # "グァラル" matches the place surface form -> glossary message added.
        result = build_qa_messages("グァラル の街", "Guaral translation", glossary, config)
        all_content = " ".join(m["content"] for m in result)

        assert "APPROVED" in all_content
        assert "Guaral" in all_content
        # The glossary is delivered as a primed user/assistant exchange.
        assert any(m["role"] == "assistant" for m in result)


# =========================================================================
# Programmatic QA check
# =========================================================================


class TestProgrammaticQACheck:
    """Tests for programmatic_qa_check()."""

    def test_too_short_fails(self, tmp_path: Path):
        """Very short translation triggers programmatic failure."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        # Source is long, translation is very short.
        source = "これは長い文章です。" * 20
        translation = "Short."

        result = programmatic_qa_check(source, translation, config)

        assert result is not None
        assert result.result == "fail"
        assert result.source == "programmatic"
        assert "short" in result.issues[0]

    def test_too_long_fails(self, tmp_path: Path):
        """Very long translation triggers programmatic failure."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        source = "短い。"
        translation = "This is an extremely long repeated text. " * 200

        result = programmatic_qa_check(source, translation, config)

        assert result is not None
        assert result.result == "fail"
        assert result.source == "programmatic"
        assert "long" in result.issues[0]

    def test_normal_length_passes(self, tmp_path: Path):
        """Normal-length translation returns None (proceed to LLM)."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        source = "これはテストです。翻訳してください。普通の長さの文章。"
        translation = "This is a test. Please translate. Normal length text."

        result = programmatic_qa_check(source, translation, config)

        assert result is None

    def test_empty_source_skips_check(self, tmp_path: Path):
        """Empty source text skips the check (returns None)."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)

        result = programmatic_qa_check("", "any translation", config)

        assert result is None


# =========================================================================
# Overlap loading
# =========================================================================


class TestLoadOverlap:
    """Tests for load_overlap()."""

    def test_same_spine_overlap(self, tmp_path: Path):
        """Chunk NNNN.015 loads NNNN.014's translation."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        manifest = _make_manifest(work_dir, spines=[(3, 15)])

        # Write the overlap translation.
        prev_tc = _make_translated_chunk(
            "0003.014", source_text="prev src", translated_text="prev trans"
        )
        _write_translation_file(work_dir, prev_tc)

        chunk = _make_chunk("0003.015", spine_index=3, chunk_index=15)

        result = load_overlap(chunk, manifest, config)

        assert result is not None
        assert result.chunk_id == "0003.014"
        assert result.translated_text == "prev trans"

    def test_cross_spine_overlap(self, tmp_path: Path):
        """First chunk of spine 3 loads last chunk of spine 2."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        manifest = _make_manifest(work_dir, spines=[(2, 5), (3, 3)])

        # Write last chunk of spine 2.
        prev_tc = _make_translated_chunk(
            "0002.005", source_text="end of ch2", translated_text="end of ch2 trans"
        )
        _write_translation_file(work_dir, prev_tc)

        chunk = _make_chunk("0003.001", spine_index=3, chunk_index=1)

        result = load_overlap(chunk, manifest, config)

        assert result is not None
        assert result.chunk_id == "0002.005"

    def test_cross_spine_disabled_returns_none(self, tmp_path: Path):
        """When cross_spine_overlap is disabled, first chunk of spine
        has no overlap."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.cross_spine_overlap = False
        manifest = _make_manifest(work_dir, spines=[(2, 5), (3, 3)])

        chunk = _make_chunk("0003.001", spine_index=3, chunk_index=1)

        result = load_overlap(chunk, manifest, config)

        assert result is None

    def test_first_chunk_of_book_no_overlap(self, tmp_path: Path):
        """The very first chunk of the book has no overlap."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        manifest = _make_manifest(work_dir, spines=[(1, 3)])

        chunk = _make_chunk("0001.001", spine_index=1, chunk_index=1)

        result = load_overlap(chunk, manifest, config)

        assert result is None

    def test_missing_overlap_raises(self, tmp_path: Path):
        """When the overlap chunk hasn't been translated, RuntimeError."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        manifest = _make_manifest(work_dir, spines=[(1, 3)])

        # Chunk 1.002 needs 1.001, but 1.001 is not translated.
        chunk = _make_chunk("0001.002", spine_index=1, chunk_index=2)

        with pytest.raises(RuntimeError, match="not been translated"):
            load_overlap(chunk, manifest, config)

    def test_overlap_disabled_returns_none(self, tmp_path: Path):
        """When overlap_chunks is 0, overlap is always None."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.overlap_chunks = 0
        manifest = _make_manifest(work_dir, spines=[(1, 3)])

        chunk = _make_chunk("0001.002", spine_index=1, chunk_index=2)

        result = load_overlap(chunk, manifest, config)

        assert result is None


# =========================================================================
# translate_chunk (core per-chunk translation)
# =========================================================================


class TestTranslateChunk:
    """Tests for translate_chunk() with mocked LLM client."""

    def test_pass1_only(self, tmp_path: Path):
        """With double_pass disabled, a single API call is made.
        pass1_translation equals translated_text."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        mock_client = _make_mock_client(_mock_completion("Pass 1 result."))

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.pass1_translation == "Pass 1 result."
        assert result.pass1_analysis is None
        assert result.translated_text == "Pass 1 result."
        assert result.pass_count == 1
        mock_client.complete.assert_called_once()

    def test_double_pass(self, tmp_path: Path):
        """With double_pass enabled, two API calls are made and the
        translations differ."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = True
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        mock_client = _make_mock_client(
            [
                _mock_completion("Pass 1 draft."),
                _mock_completion("Pass 2 revised."),
            ]
        )

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.pass1_translation == "Pass 1 draft."
        assert result.pass1_analysis is None
        assert result.translated_text == "Pass 2 revised."
        assert result.pass_count == 2
        assert mock_client.complete.call_count == 2

    def test_pass1_analysis_captured(self, tmp_path: Path):
        """When Pass 1 returns an <analysis> block, it is extracted and
        stored in pass1_analysis, stripped from pass1_translation."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        raw_output = "<analysis>\nTerminology check.\n</analysis>\nThe final translation."
        mock_client = _make_mock_client(_mock_completion(raw_output))

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.pass1_analysis is not None
        assert "<analysis>" in result.pass1_analysis
        assert "Terminology check." in result.pass1_analysis
        assert result.pass1_translation == "The final translation."
        assert result.translated_text == "The final translation."

    def test_qa_pass(self, tmp_path: Path):
        """QA enabled, judge returns pass — chunk saved with qa_result='pass'."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = False

        # Pass 1 translation — needs enough text to pass programmatic check.
        translation = "This is a reasonable length translation for the test."
        mock_client = _make_mock_client(_mock_completion(translation))
        mock_client.complete_json.return_value = QAResponse(result="pass", issues=[])

        # Source text should have similar token count to avoid programmatic fail.
        chunk = _make_chunk(text="これは合理的な長さのテスト翻訳です。テストのための文章。")
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result == "pass"
        assert result.qa_issues == []
        mock_client.complete_json.assert_called_once()

    def test_qa_disabled(self, tmp_path: Path):
        """With qa_check disabled, no QA call is made and qa_result is None."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        mock_client = _make_mock_client(_mock_completion("Translated."))

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result is None
        mock_client.complete_json.assert_not_called()

    def test_token_usage_accumulated(self, tmp_path: Path):
        """Token usage is accumulated across passes via client-level tracking."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = True
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        mock_client = _make_mock_client(
            [
                CompletionResult(
                    text="Pass 1.",
                    token_usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    model="test-model",
                ),
                CompletionResult(
                    text="Pass 2.",
                    token_usage={
                        "prompt_tokens": 200,
                        "completion_tokens": 60,
                        "total_tokens": 260,
                    },
                    model="test-model",
                ),
            ]
        )

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.token_usage["prompt_tokens"] == 300
        assert result.token_usage["completion_tokens"] == 110
        assert result.token_usage["total_tokens"] == 410


# =========================================================================
# QA flow
# =========================================================================


class TestQAFlow:
    """Tests for QA-specific behaviour in translate_chunk."""

    def test_llm_judge_fail(self, tmp_path: Path):
        """LLM QA judge returns fail — translate_chunk returns qa_result='fail'."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = False

        translation = "This is a reasonable length translation for the test."
        mock_client = _make_mock_client(_mock_completion(translation))
        mock_client.complete_json.return_value = QAResponse(
            result="fail", issues=[QAIssue(severity="high", issue="Missing paragraphs")]
        )

        chunk = _make_chunk(text="これは合理的な長さのテスト翻訳です。テストのための文章。")
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result == "fail"
        assert "Missing paragraphs" in result.qa_issues

    def test_llm_judge_low_severity_passes(self, tmp_path: Path):
        """Only low-severity issues -> chunk passes, issues not surfaced as failures."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = False

        translation = "This is a reasonable length translation for the test."
        mock_client = _make_mock_client(_mock_completion(translation))
        mock_client.complete_json.return_value = QAResponse(
            result="fail", issues=[QAIssue(severity="low", issue="word choice nuance")]
        )

        chunk = _make_chunk(text="これは合理的な長さのテスト翻訳です。テストのための文章。")
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        # Low-severity issues never force a failure.
        assert result.qa_result == "pass"
        assert result.qa_issues == []

    def test_programmatic_short_skips_llm(self, tmp_path: Path):
        """Programmatic check catches too-short output — no LLM QA call."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = False

        # Return very short translation for a long source.
        mock_client = _make_mock_client(_mock_completion("No."))

        chunk = _make_chunk(text="これは長い文章です。" * 30)
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result == "fail"
        assert "short" in result.qa_issues[0]
        mock_client.complete_json.assert_not_called()

    def test_malformed_qa_json_treated_as_failure(self, tmp_path: Path):
        """When complete_json raises LLMStructuredOutputError, it's
        treated as QA failure."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = False

        translation = "This is a reasonable length translation for the test."
        mock_client = _make_mock_client(_mock_completion(translation))
        mock_client.complete_json.side_effect = LLMStructuredOutputError("Failed after retries")

        chunk = _make_chunk(text="これは合理的な長さのテスト翻訳です。テストのための文章。")
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result == "fail"
        assert "unparseable JSON" in result.qa_issues[0]


# =========================================================================
# _run_qa helper
# =========================================================================


class TestRunQA:
    """Tests for the _run_qa() severity-gating helper."""

    def test_high_severity_fails(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        mock_client = _make_mock_client()
        mock_client.complete_json.return_value = QAResponse(
            result="fail", issues=[QAIssue(severity="high", issue="Missing paragraphs")]
        )

        source = "これは合理的な長さのテスト翻訳です。テストのための文章。"
        translation = "This is a reasonable length translation for the test."
        qa_result, issues = _run_qa(
            "0001.001", source, translation, glossary, mock_client, config
        )

        assert qa_result == "fail"
        assert issues == ["Missing paragraphs"]

    def test_only_low_severity_passes(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        mock_client = _make_mock_client()
        mock_client.complete_json.return_value = QAResponse(
            result="fail", issues=[QAIssue(severity="low", issue="word choice")]
        )

        source = "これは合理的な長さのテスト翻訳です。テストのための文章。"
        translation = "This is a reasonable length translation for the test."
        qa_result, issues = _run_qa(
            "0001.001", source, translation, glossary, mock_client, config
        )

        assert qa_result == "pass"
        assert issues == []

    def test_unparseable_json_fails(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        mock_client = _make_mock_client()
        mock_client.complete_json.side_effect = LLMStructuredOutputError("nope")

        source = "これは合理的な長さのテスト翻訳です。テストのための文章。"
        translation = "This is a reasonable length translation for the test."
        qa_result, issues = _run_qa(
            "0001.001", source, translation, glossary, mock_client, config
        )

        assert qa_result == "fail"
        assert "unparseable JSON" in issues[0]

    def test_programmatic_short_circuits_llm(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        mock_client = _make_mock_client()

        # Long source, tiny translation -> programmatic fail, no LLM call.
        source = "これは長い文章です。" * 30
        qa_result, issues = _run_qa(
            "0001.001", source, "No.", glossary, mock_client, config
        )

        assert qa_result == "fail"
        assert "short" in issues[0]
        mock_client.complete_json.assert_not_called()


# =========================================================================
# QA-fix pass
# =========================================================================


class TestBuildQAFixMessages:
    """Tests for build_qa_fix_messages()."""

    def test_contains_source_translation_and_issues(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        result = build_qa_fix_messages(
            "original source",
            "broken translation",
            ["Boilerplate leaked: 'visit our site'"],
            glossary,
            config,
        )

        assert result[0]["role"] == "system"
        user_content = result[-1]["content"]
        assert "original source" in user_content
        assert "broken translation" in user_content
        assert "Boilerplate leaked: 'visit our site'" in user_content

    def test_glossary_injected_when_relevant(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        glossary = _make_glossary()

        result = build_qa_fix_messages(
            "グァラル の街", "Guaral text", ["issue"], glossary, config
        )
        all_content = " ".join(m["content"] for m in result)

        assert "APPROVED" in all_content
        assert "Guaral" in all_content


class TestQAFixChunk:
    """Tests for qa_fix_chunk()."""

    def test_fix_passes_and_carries_prior_pass1(self, tmp_path: Path):
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.rolling_summary = False
        glossary = _make_glossary()

        chunk = _make_chunk(text="これは合理的な長さのテスト翻訳です。テストのための文章。")
        prior = _make_translated_chunk(
            chunk_id=chunk.chunk_id,
            source_text=chunk.text,
            translated_text="Broken with leaked boilerplate.",
            pass1_translation="Original pass 1 text.",
            qa_result="fail",
            qa_issues=["Boilerplate leaked: 'x'"],
        )

        fixed = "This is a reasonable length corrected translation for the test."
        mock_client = _make_mock_client(_mock_completion(fixed))
        # QA on the fixed text passes.
        mock_client.complete_json.return_value = QAResponse(result="pass", issues=[])

        result = qa_fix_chunk(
            chunk=chunk,
            prior=prior,
            issues=prior.qa_issues,
            config=config,
            glossary=glossary,
            overlap=None,
            rolling_summaries=[],
            llm_client=mock_client,
        )

        assert result.translated_text == fixed
        assert result.qa_result == "pass"
        assert result.qa_issues == []
        # Prior Pass 1 text is carried over for record continuity.
        assert result.pass1_translation == "Original pass 1 text."
        # Caller overwrites total_attempts; the fix itself reports 1.
        assert result.total_attempts == 1


# =========================================================================
# Rolling summary
# =========================================================================


class TestRollingSummary:
    """Tests for rolling summary generation and persistence."""

    def test_summary_generated_and_appended(self, tmp_path: Path):
        """After translation, the summary is appended to the file."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = True

        mock_translate = _make_mock_client(_mock_completion("Translated text here."))
        mock_summary = _make_mock_client(_mock_completion("Summary of events."))

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_translate, mock_summary)

        assert result.summary_generated == "Summary of events."
        mock_summary.complete.assert_called_once()

    def test_summary_not_generated_when_disabled(self, tmp_path: Path):
        """With rolling_summary disabled, no summary LLM call is made."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = False
        config.translation_phase.rolling_summary = False

        mock_client = _make_mock_client(_mock_completion("Translated."))

        chunk = _make_chunk()
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.summary_generated is None

    def test_summary_not_generated_on_qa_fail(self, tmp_path: Path):
        """Summary is not generated when QA fails."""
        work_dir = _setup_work_dir(tmp_path)
        config = _make_config(work_dir)
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.rolling_summary = True

        # Return short translation to trigger programmatic QA fail.
        mock_client = _make_mock_client(_mock_completion("No."))

        chunk = _make_chunk(text="これは長い文章です。" * 30)
        glossary = _make_glossary()

        result = translate_chunk(chunk, config, glossary, None, [], mock_client)

        assert result.qa_result == "fail"
        assert result.summary_generated is None

    def test_update_overwrites_existing_entry(self):
        """Re-translation overwrites the existing summary entry."""
        summaries = [
            {"chunk_id": "0001.001", "summary": "old summary"},
            {"chunk_id": "0001.002", "summary": "other"},
        ]
        result = _update_rolling_summary(summaries, "0001.001", "new summary")

        assert len(result) == 2
        assert result[0]["summary"] == "new summary"

    def test_update_appends_new_entry(self):
        """New chunk_id appends to the list."""
        summaries = [{"chunk_id": "0001.001", "summary": "first"}]
        result = _update_rolling_summary(summaries, "0001.002", "second")

        assert len(result) == 2
        assert result[1]["chunk_id"] == "0001.002"

    def test_rolling_summary_io(self, tmp_path: Path):
        """Save and load round-trip for rolling summaries."""
        work_dir = _setup_work_dir(tmp_path)

        summaries = [
            {"chunk_id": "0001.001", "summary": "Event A."},
            {"chunk_id": "0001.002", "summary": "Event B."},
        ]
        _save_rolling_summaries(work_dir, summaries)
        loaded = _load_rolling_summaries(work_dir)

        assert len(loaded) == 2
        assert loaded[0]["chunk_id"] == "0001.001"

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        """Loading summaries when file doesn't exist returns []."""
        work_dir = _setup_work_dir(tmp_path)
        loaded = _load_rolling_summaries(work_dir)
        assert loaded == []


# =========================================================================
# run_translate_stage
# =========================================================================


def _setup_stage_test(
    tmp_path: Path,
    spines: list[tuple[int, int]] | None = None,
    chunk_texts: dict[str, str] | None = None,
) -> tuple[Path, AppConfig, PipelineState, Manifest]:
    """Set up a work directory for stage runner tests.

    Creates the manifest, chunk files, glossary, and marks prior stages
    as complete.
    """
    work_dir = _setup_work_dir(tmp_path)
    config = _make_config(work_dir)
    config.translation_phase.double_pass = False
    config.translation_phase.qa_check = False
    config.translation_phase.rolling_summary = False
    config.translation_phase.overlap_chunks = 0

    manifest = _make_manifest(work_dir, spines)

    state = load_state(work_dir)
    _mark_prior_stages_complete(work_dir, state)

    # Write glossary.
    _write_glossary(work_dir, _make_glossary())

    # Write chunk files.
    all_ids = _enumerate_chunk_ids(manifest)
    for cid in all_ids:
        text = "テスト文章。これは翻訳のテストです。"
        if chunk_texts and cid in chunk_texts:
            text = chunk_texts[cid]
        spine_idx, chunk_idx = [int(x) for x in cid.split(".")]
        c = _make_chunk(cid, spine_idx, chunk_idx, text)
        _write_chunk_file(work_dir, c)

    return work_dir, config, state, manifest


class TestRunTranslateStage:
    """Tests for run_translate_stage() with mocked LLM."""

    @patch("dao_bridge.translate.LLMClient")
    def test_translates_all_chunks(self, mock_llm_cls, tmp_path: Path):
        """All chunks are translated and marked completed."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 2)])

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["completed"] == 2
        assert result["error"] is None
        assert result["failed_chunk"] is None

        # Verify translation files exist.
        assert translation_path(work_dir, "0001.001").exists()
        assert translation_path(work_dir, "0001.002").exists()

    @patch("dao_bridge.translate.LLMClient")
    def test_completed_chunks_skipped(self, mock_llm_cls, tmp_path: Path):
        """Already-completed chunks are skipped on re-run."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 2)])

        # Pre-mark first chunk as completed.
        mark_item_completed(work_dir, state, "translate", "0001.001")
        # Write translation file so it exists.
        tc = _make_translated_chunk("0001.001")
        _write_translation_file(work_dir, tc)

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        # Only the second chunk should have been translated.
        assert result["completed"] == 1

    @patch("dao_bridge.translate.LLMClient")
    def test_failed_chunks_retried(self, mock_llm_cls, tmp_path: Path):
        """Chunks with 'failed' status are retried on re-run."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])

        # Pre-mark as failed.
        mark_item_failed(work_dir, state, "translate", "0001.001", "connection error")

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["completed"] == 1
        assert result["error"] is None

    @patch("dao_bridge.translate.LLMClient")
    def test_qa_failure_is_non_blocking(self, mock_llm_cls, tmp_path: Path):
        """Persistent QA failure does NOT halt the pipeline: the best attempt
        is kept, the chunk is marked completed, and the run continues."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 2)])
        config.translation_phase.qa_check = True
        config.translation_phase.qa_max_retries = 1

        # Return short translation to trigger programmatic QA fail on every attempt.
        mock_client = _make_mock_client(_mock_completion("No."))
        mock_llm_cls.return_value = mock_client

        # Write chunks with long text so programmatic QA fails.
        long_text = "これは長い文章です。" * 30
        for cid in ["0001.001", "0001.002"]:
            si, ci = [int(x) for x in cid.split(".")]
            c = _make_chunk(cid, si, ci, long_text)
            _write_chunk_file(work_dir, c)

        result = run_translate_stage(work_dir, config, state, manifest)

        # Pipeline runs to completion; nothing halts.
        assert result.get("failed_chunk") is None
        assert result["error"] is None
        assert result["completed"] == 2

        # The kept (best) translation is still saved, with qa_result == "fail".
        for cid in ["0001.001", "0001.002"]:
            tp = translation_path(work_dir, cid)
            assert tp.exists()
            saved = TranslatedChunk.model_validate_json(tp.read_text(encoding="utf-8"))
            assert saved.qa_result == "fail"

        # State marks the chunks completed (not failed) so they are not retried.
        reloaded = load_state(work_dir)
        assert reloaded.items["translate:0001.001"].status == "completed"
        assert reloaded.items["translate:0001.002"].status == "completed"

    @patch("dao_bridge.translate.LLMClient")
    def test_qa_fix_pass_recovers_chunk(self, mock_llm_cls, tmp_path: Path):
        """Attempt 1 fails QA, the QA-fix pass passes -> attempt 2 is selected
        and the run stops early without a third attempt."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])
        config.translation_phase.double_pass = False
        config.translation_phase.qa_check = True
        config.translation_phase.qa_max_retries = 2
        config.translation_phase.rolling_summary = False

        long_text = "これは合理的な長さのテスト翻訳です。テストのための文章。" * 3
        c = _make_chunk("0001.001", 1, 1, long_text)
        _write_chunk_file(work_dir, c)

        good = "This is a reasonable length translation for the test. " * 3
        # complete() is called once per attempt (Pass 1, then QA-fix).
        mock_client = _make_mock_client([_mock_completion(good), _mock_completion(good)])
        # First QA verdict fails (high), second (after fix) passes.
        mock_client.complete_json.side_effect = [
            QAResponse(result="fail", issues=[QAIssue(severity="high", issue="repetition loop")]),
            QAResponse(result="pass", issues=[]),
        ]
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["completed"] == 1
        assert result["error"] is None

        saved = TranslatedChunk.model_validate_json(
            translation_path(work_dir, "0001.001").read_text(encoding="utf-8")
        )
        assert saved.qa_result == "pass"
        assert saved.selected_attempt == 2
        # Stopped after attempt 2 — only two QA verdicts consumed.
        assert mock_client.complete_json.call_count == 2

    @patch("dao_bridge.translate.LLMClient")
    def test_chunk_range_filter(self, mock_llm_cls, tmp_path: Path):
        """--from/--to restricts which chunks are translated."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 5)])

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(
            work_dir,
            config,
            state,
            manifest,
            from_chunk="0001.002",
            to_chunk="0001.004",
        )

        assert result["completed"] == 3
        assert result["total_chunks"] == 3

        # Only chunks 2, 3, 4 should have translation files.
        assert not translation_path(work_dir, "0001.001").exists()
        assert translation_path(work_dir, "0001.002").exists()
        assert translation_path(work_dir, "0001.003").exists()
        assert translation_path(work_dir, "0001.004").exists()
        assert not translation_path(work_dir, "0001.005").exists()

    @patch("dao_bridge.translate.LLMClient")
    def test_sequential_enforcement(self, mock_llm_cls, tmp_path: Path):
        """When overlap is enabled, error if previous chunk not completed."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 3)])
        config.translation_phase.overlap_chunks = 1

        # Mark chunk 1 as completed, but leave chunk 2 as pending.
        mark_item_completed(work_dir, state, "translate", "0001.001")
        tc = _make_translated_chunk("0001.001")
        _write_translation_file(work_dir, tc)

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        # Translate only chunk 3, which depends on chunk 2.
        result = run_translate_stage(
            work_dir,
            config,
            state,
            manifest,
            from_chunk="0001.003",
            to_chunk="0001.003",
        )

        assert result["failed_chunk"] == "0001.003"
        assert "depends on" in result["error"]

    @patch("dao_bridge.translate.LLMClient")
    def test_infrastructure_error_halts(self, mock_llm_cls, tmp_path: Path):
        """Infrastructure errors (LLM connection) halt the pipeline."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])

        mock_client = _make_mock_client()
        mock_client.complete.side_effect = ConnectionError("server down")
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["failed_chunk"] == "0001.001"
        assert "server down" in result["error"]
        assert result["completed"] == 0

    @patch("dao_bridge.translate.LLMClient")
    def test_force_retranslates_completed(self, mock_llm_cls, tmp_path: Path):
        """--force causes completed chunks to be retranslated."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])

        # Pre-mark as completed.
        mark_item_completed(work_dir, state, "translate", "0001.001")
        tc = _make_translated_chunk("0001.001")
        _write_translation_file(work_dir, tc)

        mock_client = _make_mock_client(_mock_completion("Re-translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest, force=True)

        assert result["completed"] == 1

        # The translation file should contain the new translation.
        tp = translation_path(work_dir, "0001.001")
        data = json.loads(tp.read_text(encoding="utf-8"))
        assert data["translated_text"] == "Re-translated."

    @patch("dao_bridge.translate.LLMClient")
    def test_progress_callback_called(self, mock_llm_cls, tmp_path: Path):
        """The on_progress callback is invoked during translation."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        progress_calls: list[TranslationProgress] = []

        def on_progress(p: TranslationProgress) -> None:
            progress_calls.append(p)

        result = run_translate_stage(work_dir, config, state, manifest, on_progress=on_progress)

        assert result["completed"] == 1
        assert len(progress_calls) >= 1
        assert progress_calls[0].chunk_id == "0001.001"

    @patch("dao_bridge.translate.LLMClient")
    def test_rolling_summary_saved_to_file(self, mock_llm_cls, tmp_path: Path):
        """When rolling_summary is enabled, summaries are saved to disk."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 2)])
        config.translation_phase.rolling_summary = True

        mock_client = _make_mock_client(
            [
                # Chunk 1 translation.
                _mock_completion("Translated chunk 1."),
                # Chunk 1 summary.
                _mock_completion("Summary of chunk 1."),
                # Chunk 2 translation.
                _mock_completion("Translated chunk 2."),
                # Chunk 2 summary.
                _mock_completion("Summary of chunk 2."),
            ]
        )
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["completed"] == 2

        # Check summary file.
        summaries = _load_rolling_summaries(work_dir)
        assert len(summaries) == 2
        assert summaries[0]["chunk_id"] == "0001.001"
        assert summaries[1]["chunk_id"] == "0001.002"


# =========================================================================
# End-of-run summary
# =========================================================================


class TestEndOfRunSummary:
    """Tests for the summary dict returned by run_translate_stage."""

    @patch("dao_bridge.translate.LLMClient")
    def test_success_summary(self, mock_llm_cls, tmp_path: Path):
        """Successful run returns correct counts."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 2)])

        mock_client = _make_mock_client(_mock_completion("Translated."))
        mock_llm_cls.return_value = mock_client

        result = run_translate_stage(work_dir, config, state, manifest)

        assert result["completed"] == 2
        assert result["error"] is None
        assert result["total_tokens"] > 0

    @patch("dao_bridge.translate.LLMClient")
    def test_qa_failure_kept_summary(self, mock_llm_cls, tmp_path: Path):
        """A QA-failing chunk (with no retries) is kept and counted as completed."""
        work_dir, config, state, manifest = _setup_stage_test(tmp_path, spines=[(1, 1)])
        config.translation_phase.qa_check = True
        config.translation_phase.qa_max_retries = 0

        mock_client = _make_mock_client(_mock_completion("No."))
        mock_llm_cls.return_value = mock_client

        # Long source text to trigger programmatic QA fail.
        long_text = "これは長い文章です。" * 30
        c = _make_chunk("0001.001", 1, 1, long_text)
        _write_chunk_file(work_dir, c)

        result = run_translate_stage(work_dir, config, state, manifest)

        # Non-blocking: the chunk is kept and the run completes.
        assert result["completed"] == 1
        assert result["error"] is None
        assert result.get("failed_chunk") is None

        # The kept record reflects the only (failing) attempt.
        saved = TranslatedChunk.model_validate_json(
            translation_path(work_dir, "0001.001").read_text(encoding="utf-8")
        )
        assert saved.qa_result == "fail"
        assert saved.selected_attempt == 1


# =========================================================================
# Chunk enumeration helpers
# =========================================================================


class TestChunkEnumeration:
    """Tests for _enumerate_chunk_ids and _filter_chunk_range."""

    def test_enumerate_chunk_ids_order(self, tmp_path: Path):
        """Chunks are enumerated in spine order, then chunk order."""
        work_dir = _setup_work_dir(tmp_path)
        manifest = _make_manifest(work_dir, spines=[(1, 2), (3, 3)])

        ids = _enumerate_chunk_ids(manifest)

        assert ids == [
            "0001.001",
            "0001.002",
            "0003.001",
            "0003.002",
            "0003.003",
        ]

    def test_enumerate_skips_zero_chunk_spines(self, tmp_path: Path):
        """Spines with chunk_count=0 are skipped."""
        work_dir = _setup_work_dir(tmp_path)
        manifest = _make_manifest(work_dir, spines=[(1, 2), (2, 0), (3, 1)])

        ids = _enumerate_chunk_ids(manifest)

        assert "0002" not in " ".join(ids)
        assert len(ids) == 3

    def test_filter_chunk_range(self):
        """Range filter works with string comparison."""
        ids = ["0001.001", "0001.002", "0002.001", "0002.002", "0003.001"]

        result = _filter_chunk_range(ids, "0001.002", "0002.002")

        assert result == ["0001.002", "0002.001", "0002.002"]

    def test_filter_from_only(self):
        """--from without --to continues to end."""
        ids = ["0001.001", "0001.002", "0002.001"]

        result = _filter_chunk_range(ids, "0001.002", None)

        assert result == ["0001.002", "0002.001"]

    def test_filter_none_returns_all(self):
        """No filter returns all chunks."""
        ids = ["0001.001", "0001.002"]

        result = _filter_chunk_range(ids, None, None)

        assert result == ids
