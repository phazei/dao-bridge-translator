"""Tests for dao_bridge.glossary — build, reconcile, export, and integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.config import AppConfig
from dao_bridge.glossary import (
    _BuildMeta,
    _load_build_meta,
    _load_glossary,
    _merge_extraction_into_glossary,
    _pack_batches,
    _save_build_meta,
    _save_glossary,
    glossary_build,
    glossary_export,
    glossary_reconcile,
    validate_glossary_categories,
)
from dao_bridge.schemas import (
    Chunk,
    Glossary,
    GlossaryCorrectionEntry,
    GlossaryEntry,
    GlossaryExtractionEntry,
    GlossaryExtractionResponse,
    GlossaryReconcileResponse,
    GlossarySpeechMergeResponse,
    Manifest,
    ManifestItem,
)
from dao_bridge.state import (
    PipelineState,
    load_state,
    mark_item_completed,
    mark_stage_completed,
    mark_stage_started,
)
from dao_bridge.workdir import (
    atomic_write,
    chunk_dir,
    glossary_path,
    manifest_path,
    pad_spine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(work_dir: Path, **overrides) -> AppConfig:
    """Create a minimal AppConfig pointing at work_dir."""
    defaults = {
        "source_epub": str(work_dir / "test.epub"),
        "work_dir": str(work_dir),
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_manifest(
    work_dir: Path,
    n_spines: int = 2,
    chunks_per_spine: int = 2,
    spine_width: int = 4,
) -> Manifest:
    """Create and persist a manifest with chunkable spine items."""
    items = []
    for i in range(n_spines):
        items.append(
            ManifestItem(
                spine_index=i,
                original_href=f"ch{i}.xhtml",
                raw_path=f"raw/{pad_spine(i, spine_width)}.xhtml",
                clean_path=f"clean/{pad_spine(i, spine_width)}.md",
                classification="chapter",
                chunk_count=chunks_per_spine,
            )
        )
    manifest = Manifest(
        source_epub_path=str(work_dir / "test.epub"),
        book_id="test-book",
        spine=items,
        spine_padding_width=spine_width,
    )
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))
    return manifest


def _write_chunks(
    work_dir: Path,
    n_spines: int = 2,
    chunks_per_spine: int = 2,
    tokens_per_chunk: int = 500,
    spine_width: int = 4,
) -> list[Chunk]:
    """Write chunk JSON files to disk, returning the chunk objects."""
    all_chunks = []
    for si in range(n_spines):
        cd = chunk_dir(work_dir, si, spine_width)
        cd.mkdir(parents=True, exist_ok=True)
        for ci in range(1, chunks_per_spine + 1):
            chunk_id = f"{pad_spine(si, spine_width)}.{ci:03d}"
            text = f"Sample text for chunk {chunk_id}. " * 20
            c = Chunk(
                chunk_id=chunk_id,
                spine_index=si,
                chunk_index=ci,
                source_file=f"clean/{pad_spine(si, spine_width)}.md",
                block_range=(0, 5),
                token_count=tokens_per_chunk,
                text=text,
            )
            chunk_path = cd / f"{chunk_id}.json"
            atomic_write(chunk_path, c.model_dump_json(indent=2))
            all_chunks.append(c)
    return all_chunks


def _mock_extraction_response(
    entries: list[dict] | None = None,
    corrections: list[dict] | None = None,
) -> GlossaryExtractionResponse:
    """Build a GlossaryExtractionResponse from simple dicts."""
    entry_objs = []
    for e in entries or []:
        entry_objs.append(GlossaryExtractionEntry(**e))
    corr_objs = []
    for c in corrections or []:
        corr_objs.append(GlossaryCorrectionEntry(**c))
    return GlossaryExtractionResponse(entries=entry_objs, corrections=corr_objs)


def _setup_work_dir(tmp_path: Path) -> Path:
    """Create a standard work directory with chunks/ subdirectory."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "chunks").mkdir()
    return work


def _mark_prior_stages_complete(work_dir: Path, state: PipelineState) -> None:
    """Mark all stages before glossary_build as completed."""
    for stage in ("extract", "clean", "classify", "chunk"):
        mark_stage_started(work_dir, state, stage)
        mark_stage_completed(work_dir, state, stage)


# ---------------------------------------------------------------------------
# TestBatchPacking
# ---------------------------------------------------------------------------


class TestBatchPacking:
    """Tests for the greedy batch packing algorithm."""

    def test_single_batch_all_fit(self):
        """All chunks fit in one batch."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            )
            for i in range(1, 4)
        ]
        batches = _pack_batches(chunks, target_tokens=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_multiple_batches(self):
        """Chunks are split across multiple batches."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=400,
                text="text",
            )
            for i in range(1, 6)
        ]
        batches = _pack_batches(chunks, target_tokens=1000)
        # 5 chunks @ 400 tokens each = 2000 total.
        # Batch 1: 400 + 400 = 800 (add third -> 1200 > 1000, emit)
        # Batch 2: 400 + 400 = 800 (add fifth -> 1200 > 1000, emit)
        # Batch 3: 400 (last batch)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_last_batch_smaller(self):
        """The last batch can be smaller than target."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=600,
                text="text",
            )
            for i in range(1, 4)
        ]
        batches = _pack_batches(chunks, target_tokens=1000)
        # Batch 1: 600 (add second -> 1200 > 1000, emit)
        # Batch 2: 600 (add third -> 1200 > 1000, emit)
        # Batch 3: 600 (last batch)
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_single_chunk(self):
        """Single chunk returns one batch."""
        chunks = [
            Chunk(
                chunk_id="0000.001",
                spine_index=0,
                chunk_index=1,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=5000,
                text="text",
            )
        ]
        batches = _pack_batches(chunks, target_tokens=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_empty_input(self):
        """Empty chunk list returns no batches."""
        batches = _pack_batches([], target_tokens=1000)
        assert batches == []

    def test_exact_fit(self):
        """Chunks that exactly fill the target go in one batch."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=500,
                text="text",
            )
            for i in range(1, 3)
        ]
        batches = _pack_batches(chunks, target_tokens=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 2


# ---------------------------------------------------------------------------
# TestCategoryValidation
# ---------------------------------------------------------------------------


class TestCategoryValidation:
    """Tests for validate_glossary_categories."""

    def test_valid_categories(self):
        """No error when all categories are valid."""
        g = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="X", category="character", source="extracted"
                ),
                GlossaryEntry(source_term="Y", english="Y", category="place", source="extracted"),
            ]
        )
        # Should not raise.
        validate_glossary_categories(
            g,
            [
                "character",
                "place",
                "ability",
                "title",
                "term",
                "item",
                "species",
                "clan",
                "organization",
                "other",
            ],
        )

    def test_invalid_category_raises(self):
        """Clear error with invalid category."""
        g = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="Foo", category="weapon", source="extracted"
                ),
                GlossaryEntry(source_term="Y", english="Bar", category="magic", source="extracted"),
            ]
        )
        with pytest.raises(ValueError, match="weapon"):
            validate_glossary_categories(g, ["character", "place"])

    def test_error_lists_affected_entries(self):
        """Error message lists which entries use the invalid category."""
        g = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="Foo", category="weapon", source="extracted"
                ),
            ]
        )
        with pytest.raises(ValueError, match="Foo"):
            validate_glossary_categories(g, ["character"])

    def test_empty_glossary_passes(self):
        """Empty glossary always passes."""
        validate_glossary_categories(Glossary(), ["character"])


# ---------------------------------------------------------------------------
# TestGlossaryBuild
# ---------------------------------------------------------------------------


class TestGlossaryBuild:
    """Tests for the glossary_build function."""

    def test_entries_merged_correctly(self, tmp_path):
        """New entries are added with correct fields."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "ナツキ・スバル",
                    "reading": "ナツキ・スバル",
                    "english_proposed": "Natsuki Subaru",
                    "category": "character",
                    "aliases": ["スバル"],
                    "nicknames": {},
                    "speech_style": "Casual modern speech.",
                    "notes": "Protagonist.",
                },
                {
                    "source_term": "エミリア",
                    "english_proposed": "Emilia",
                    "category": "character",
                    "aliases": [],
                    "nicknames": {"Subaru": "Emilia-tan"},
                    "speech_style": "Polite, earnest.",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        assert len(glossary.entries) == 2
        subaru = next(e for e in glossary.entries if e.english == "Natsuki Subaru")
        assert subaru.source_term == "ナツキ・スバル"
        assert subaru.reading == "ナツキ・スバル"
        assert subaru.source == "extracted"
        assert "スバル" in subaru.aliases
        assert subaru.speech_style == "Casual modern speech."
        assert subaru.first_seen_chunk is not None

        emilia = next(e for e in glossary.entries if e.english == "Emilia")
        assert emilia.nicknames["Subaru"] == "Emilia-tan"

        # book_id and book_metadata populated from manifest.
        assert glossary.book_id == "test-book"

    def test_aliases_unioned_across_batches(self, tmp_path):
        """Aliases from later batches are merged with existing."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # 4 chunks @ 500 tokens, target 1200 -> 2 batches of 2.
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "english_proposed": "Subaru",
                    "category": "character",
                    "aliases": ["バルス"],
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "english_proposed": "Subaru",
                    "category": "character",
                    "aliases": ["バルス", "ナツキ"],
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entries if e.english == "Subaru")
        assert "バルス" in subaru.aliases
        assert "ナツキ" in subaru.aliases
        # No duplicates.
        assert subaru.aliases.count("バルス") == 1

    def test_speech_styles_accumulated(self, tmp_path):
        """Speech style observations are accumulated with newline delimiter."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # 4 chunks @ 500 tokens, target 1200 -> 2 batches of 2.
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "english_proposed": "Subaru",
                    "category": "character",
                    "speech_style": "Casual speech.",
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "english_proposed": "Subaru",
                    "category": "character",
                    "speech_style": "Uses modern slang.",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entries if e.english == "Subaru")
        assert "Casual speech." in subaru.speech_style
        assert "Uses modern slang." in subaru.speech_style
        assert "\n" in subaru.speech_style

    def test_corrections_logged_to_conflicts(self, tmp_path):
        """Corrections from the LLM are logged as conflicts, not applied."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "プリシラ",
                    "english_proposed": "Priscilla",
                    "category": "character",
                },
            ],
            corrections=[
                {
                    "existing_english": "Priscilla",
                    "source_term": "プリシラ・バーリエル",
                    "corrected_english": "Priscilla Barielle",
                    "reason": "Full name appears.",
                }
            ],
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        # The correction should not have been applied.
        priscilla = next(e for e in glossary.entries if e.source_term == "プリシラ")
        assert priscilla.english == "Priscilla"

        # But it should be in the build meta conflicts.
        meta = _load_build_meta(work)
        assert len(meta.corrections) == 1
        assert meta.corrections[0]["corrected_english"] == "Priscilla Barielle"
        # Also recorded as a conflict.
        assert any(c.source_term == "プリシラ・バーリエル" for c in meta.conflicts)

    def test_user_sourced_entries_never_modified(self, tmp_path):
        """Entries with source='user' are never modified by build."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        # Pre-seed a user entry.
        pre_glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="スバル",
                    english="Subaru (custom)",
                    category="character",
                    source="user",
                    aliases=["original_alias"],
                )
            ]
        )
        _save_glossary(work, pre_glossary)

        mock_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "english_proposed": "Natsuki Subaru",
                    "category": "character",
                    "aliases": ["バルス"],
                    "speech_style": "Casual.",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=False)

        # User entry should be unchanged.
        subaru = next(e for e in glossary.entries if e.source_term == "スバル")
        assert subaru.english == "Subaru (custom)"
        assert subaru.source == "user"
        assert subaru.aliases == ["original_alias"]
        assert subaru.speech_style is None

    def test_state_tracking_per_batch(self, tmp_path):
        """Batch item IDs are tracked in state."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(entries=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, force=True)

        # Check state has batch items.
        batch_items = [k for k in state.items.keys() if k.startswith("glossary_build:")]
        assert len(batch_items) >= 1
        for item_key in batch_items:
            assert state.items[item_key].status == "completed"

    def test_english_conflict_recorded(self, tmp_path):
        """Differing English proposals create a conflict record."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # 4 chunks @ 500 tokens, target 1200 -> 2 batches of 2.
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "ルグニカ",
                    "english_proposed": "Lugnica",
                    "category": "place",
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "ルグニカ",
                    "english_proposed": "Lugunica",
                    "category": "place",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        # Original english preserved.
        entry = next(e for e in glossary.entries if e.source_term == "ルグニカ")
        assert entry.english == "Lugnica"

        # Conflict recorded.
        meta = _load_build_meta(work)
        conflict = next(c for c in meta.conflicts if c.source_term == "ルグニカ")
        assert any(a["english"] == "Lugunica" for a in conflict.alternatives)

    def test_missing_reading_backfilled_on_merge(self):
        """A later extraction can fill in a previously missing reading."""
        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="スバル",
                    reading=None,
                    english="Subaru",
                    category="character",
                    source="extracted",
                )
            ]
        )
        response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "reading": "すばる",
                    "english_proposed": "Subaru",
                    "category": "character",
                }
            ]
        )
        meta = _BuildMeta()

        _merge_extraction_into_glossary(
            glossary,
            response,
            "glossary_build.batch.001",
            "0000.001",
            meta,
        )

        assert glossary.entries[0].reading == "すばる"


# ---------------------------------------------------------------------------
# TestBuildResume
# ---------------------------------------------------------------------------


class TestBuildResume:
    """Tests for build-stage resumability."""

    def test_resume_after_partial(self, tmp_path):
        """Re-running from a partial glossary.json picks up at next batch."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # 4 chunks @ 500 tokens, target 1200 -> 2 batches of 2.
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        # First run: only process first batch then "crash".
        batch1_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "エミリア",
                    "english_proposed": "Emilia",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [
                batch1_response,
                RuntimeError("LLM crashed"),
            ]
            mock_llm_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="LLM crashed"):
                glossary_build(work, config, state, force=True)

        # Verify first batch was saved.
        glossary = _load_glossary(work)
        assert len(glossary.entries) == 1
        assert glossary.entries[0].english == "Emilia"

        # Second run: resume from batch 2.
        batch2_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "レム",
                    "english_proposed": "Rem",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = batch2_response
            mock_llm_cls.return_value = mock_client

            # Reset the failed state item so iter_pending_items finds it.
            glossary = glossary_build(work, config, state, force=False)

        assert len(glossary.entries) == 2
        assert any(e.english == "Emilia" for e in glossary.entries)
        assert any(e.english == "Rem" for e in glossary.entries)


# ---------------------------------------------------------------------------
# TestGlossaryReconcile
# ---------------------------------------------------------------------------


class TestGlossaryReconcile:
    """Tests for the glossary_reconcile function."""

    def test_raises_when_build_not_completed(self, tmp_path):
        """Reconcile raises RuntimeError if glossary_build stage is not completed."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # glossary_build is NOT started/completed.

        with pytest.raises(RuntimeError, match="Glossary build stage not completed"):
            glossary_reconcile(work, config, state, force=False)

    def _setup_with_conflicts(self, tmp_path):
        """Set up a work dir with a glossary and conflicts for reconcile."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        # Create glossary with an entry.
        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="ルグニカ",
                    reading="るぐにか",
                    english="Lugnica",
                    category="place",
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary)

        # Create build meta with a conflict.
        meta = _BuildMeta(
            conflicts=[
                {
                    "source_term": "ルグニカ",
                    "reading": "るぐにか",
                    "current_english": "Lugnica",
                    "alternatives": [
                        {
                            "english": "Lugunica",
                            "context_snippet": "Batch 002",
                            "batch_id": "glossary_build.batch.002",
                        }
                    ],
                    "category_variants": [],
                }
            ],
        )
        _save_build_meta(work, meta)

        return work, config, state

    def test_llm_picks_winner(self, tmp_path):
        """Reconcile applies the LLM's chosen English form."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_english="Lugunica",
            reasoning="More common romanization.",
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entry = next(e for e in glossary.entries if e.source_term == "ルグニカ")
        assert entry.english == "Lugunica"

    def test_term_change_persisted_before_item_completion(self, tmp_path):
        """Disk glossary is updated before a term item is marked complete."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_english="Lugunica",
            reasoning="More common romanization.",
        )

        def checking_mark_item_completed(work_dir, pipeline_state, stage, item_id):
            if item_id.startswith("glossary_reconcile.term."):
                disk_glossary = _load_glossary(work_dir)
                entry = next(e for e in disk_glossary.entries if e.source_term == "ルグニカ")
                assert entry.english == "Lugunica"
            return mark_item_completed(work_dir, pipeline_state, stage, item_id)

        with (
            patch("dao_bridge.glossary.LLMClient") as mock_llm_cls,
            patch(
                "dao_bridge.glossary.mark_item_completed", side_effect=checking_mark_item_completed
            ),
        ):
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entry = next(e for e in glossary.entries if e.source_term == "ルグニカ")
        assert entry.english == "Lugunica"

    def test_report_generated(self, tmp_path):
        """Reconcile report markdown is written."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_english="Lugunica",
            reasoning="More common.",
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary_reconcile(work, config, state, force=True)

        report_path = work / "glossary_reconcile_report.md"
        assert report_path.exists()
        report = report_path.read_text(encoding="utf-8")
        assert "ルグニカ" in report
        assert "Lugunica" in report
        assert "More common." in report

    def test_speech_style_consolidation(self, tmp_path):
        """Speech-style observations are consolidated by LLM."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        # Entry with multiple speech_style observations.
        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="スバル",
                    english="Subaru",
                    category="character",
                    source="extracted",
                    speech_style="Casual speech.\nUses modern slang.\nFrequent sarcasm.",
                ),
            ]
        )
        _save_glossary(work, glossary)
        _save_build_meta(work, _BuildMeta())

        mock_speech = GlossarySpeechMergeResponse(
            consolidated_speech_style="Speaks casually with modern slang and frequent sarcasm."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_speech
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        subaru = next(e for e in glossary.entries if e.english == "Subaru")
        assert subaru.speech_style == "Speaks casually with modern slang and frequent sarcasm."
        assert "\n" not in subaru.speech_style

    def test_speech_change_persisted_before_item_completion(self, tmp_path):
        """Disk glossary is updated before a speech item is marked complete."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="スバル",
                    english="Subaru",
                    category="character",
                    source="extracted",
                    speech_style="Casual speech.\nUses modern slang.",
                ),
            ]
        )
        _save_glossary(work, glossary)
        _save_build_meta(work, _BuildMeta())

        mock_speech = GlossarySpeechMergeResponse(
            consolidated_speech_style="Speaks casually with modern slang."
        )

        def checking_mark_item_completed(work_dir, pipeline_state, stage, item_id):
            if item_id.startswith("glossary_reconcile.speech."):
                disk_glossary = _load_glossary(work_dir)
                entry = next(e for e in disk_glossary.entries if e.english == "Subaru")
                assert entry.speech_style == "Speaks casually with modern slang."
            return mark_item_completed(work_dir, pipeline_state, stage, item_id)

        with (
            patch("dao_bridge.glossary.LLMClient") as mock_llm_cls,
            patch(
                "dao_bridge.glossary.mark_item_completed", side_effect=checking_mark_item_completed
            ),
        ):
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_speech
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        subaru = next(e for e in glossary.entries if e.english == "Subaru")
        assert subaru.speech_style == "Speaks casually with modern slang."

    def test_no_conflicts_completes_immediately(self, tmp_path):
        """Stage completes as no-op when no conflicts exist."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X",
                    english="X",
                    category="character",
                    source="extracted",
                    speech_style="Single observation only.",
                ),
            ]
        )
        _save_glossary(work, glossary)
        _save_build_meta(work, _BuildMeta())

        # No LLM calls should be made.
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)
            mock_llm_cls.assert_not_called()

        report_path = work / "glossary_reconcile_report.md"
        assert report_path.exists()

    def test_category_only_conflict_logged(self, tmp_path):
        """Category-only conflicts are logged to the report without LLM call."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="聖剣",
                    english="Holy Sword",
                    category="item",
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary)

        # Category-only conflict (no English alternatives).
        meta = _BuildMeta(
            conflicts=[
                {
                    "source_term": "聖剣",
                    "reading": None,
                    "current_english": "Holy Sword",
                    "alternatives": [],
                    "category_variants": ["ability"],
                }
            ],
        )
        _save_build_meta(work, meta)

        # No LLM calls should be needed for category-only conflicts.
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)
            mock_llm_cls.assert_not_called()

        # Report should mention the category conflict.
        report = (work / "glossary_reconcile_report.md").read_text(encoding="utf-8")
        assert "聖剣" in report
        assert "Category" in report or "category" in report

    def test_build_validates_preexisting_categories(self, tmp_path):
        """Build stage validates categories on pre-existing glossary entries."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        # Pre-seed glossary with invalid category.
        bad_glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X",
                    english="X",
                    category="weapon",  # not in default categories
                    source="user",
                ),
            ]
        )
        _save_glossary(work, bad_glossary)

        with pytest.raises(ValueError, match="weapon"):
            glossary_build(work, config, state, force=False)


# ---------------------------------------------------------------------------
# TestGlossaryExport
# ---------------------------------------------------------------------------


class TestGlossaryExport:
    """Tests for the glossary_export function."""

    def test_grouped_by_category_sorted_alphabetically(self, tmp_path):
        """Entries are grouped by category and sorted by english."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="B-char", english="Zorro", category="character", source="extracted"
                ),
                GlossaryEntry(
                    source_term="A-char", english="Alice", category="character", source="extracted"
                ),
                GlossaryEntry(
                    source_term="Place1", english="Kingdom", category="place", source="extracted"
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        # Character section should come before place (config order).
        char_pos = md.index("## Character")
        place_pos = md.index("## Place")
        assert char_pos < place_pos

        # Within characters, Alice before Zorro.
        alice_pos = md.index("Alice")
        zorro_pos = md.index("Zorro")
        assert alice_pos < zorro_pos

    def test_optional_fields_rendered_when_present(self, tmp_path):
        """All optional fields are rendered when present."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="スバル",
                    reading="すばる",
                    english="Subaru",
                    category="character",
                    source="extracted",
                    aliases=["バルス"],
                    nicknames={"Rem": "Subaru-kun"},
                    speech_style="Casual.",
                    notes="Main character.",
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        assert "Reading: すばる" in md
        assert "Aliases: バルス" in md
        assert "Nicknames:" in md
        assert "Subaru-kun" in md
        assert "Speech style: Casual." in md
        assert "Notes: Main character." in md

    def test_optional_fields_omitted_when_null(self, tmp_path):
        """Null/empty optional fields are not rendered."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X",
                    english="Xterm",
                    category="term",
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        assert "Reading:" not in md
        assert "Aliases:" not in md
        assert "Nicknames:" not in md
        assert "Speech style:" not in md
        assert "Notes:" not in md

    def test_stdout_mode(self, tmp_path):
        """In stdout mode, no file is written."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="X", category="character", source="extracted"
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)
        assert "# Glossary" in md
        assert not (work / "glossary.md").exists()

    def test_file_mode(self, tmp_path):
        """In file mode, glossary.md is written."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="X", category="character", source="extracted"
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=False)
        assert (work / "glossary.md").exists()
        written = (work / "glossary.md").read_text(encoding="utf-8")
        assert written == md

    def test_custom_output_path(self, tmp_path):
        """Custom output path is used when specified."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(
                    source_term="X", english="X", category="character", source="extracted"
                ),
            ]
        )
        _save_glossary(work, glossary)

        custom_path = tmp_path / "custom_glossary.md"
        glossary_export(work, config, stdout=False, output_path=custom_path)
        assert custom_path.exists()

    def test_empty_glossary(self, tmp_path):
        """Empty glossary produces a minimal markdown."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        _save_glossary(work, Glossary())

        md = glossary_export(work, config, stdout=True)
        assert "No entries" in md

    def test_entry_without_source_term(self, tmp_path):
        """Entry without source_term still renders."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entries=[
                GlossaryEntry(english="Some Term", category="term", source="extracted"),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)
        assert "**Some Term**" in md


# ---------------------------------------------------------------------------
# TestLanguageAgnostic
# ---------------------------------------------------------------------------


class TestLanguageAgnostic:
    """Tests that prompts use resolved language names, not hardcoded values."""

    def test_prompt_uses_source_language(self, tmp_path):
        """The extraction prompt contains the resolved language name."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        # Override languages to Chinese.
        config.languages.source = "zh"
        config.languages.target = "en"
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        captured_messages = []

        def capture_complete_json(messages, response_model=None, **kwargs):
            captured_messages.append(messages)
            return _mock_extraction_response(entries=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = capture_complete_json
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, force=True)

        assert len(captured_messages) == 1
        prompt_content = captured_messages[0][0]["content"]
        assert "Chinese" in prompt_content
        assert "English" in prompt_content
        # Should NOT contain hardcoded "Japanese".
        assert "Japanese" not in prompt_content


# ---------------------------------------------------------------------------
# TestGlossaryIntegration
# ---------------------------------------------------------------------------


class TestGlossaryIntegration:
    """Integration test: full mini-pipeline through glossary stages."""

    def test_full_pipeline_with_mocked_llm(self, jp_epub_path, tmp_work_dir):
        """init -> extract -> clean -> classify -> chunk -> glossary-build -> glossary-reconcile."""
        import yaml

        from dao_bridge.chunk import chunk_all
        from dao_bridge.classify import run_classify_stage
        from dao_bridge.clean import clean_all
        from dao_bridge.extract import extract_epub
        from dao_bridge.schemas import ClassificationResponse

        work = tmp_work_dir

        # Write config.
        cfg_dict = {
            "source_epub": str(jp_epub_path),
            "work_dir": str(work),
            "languages": {"source": "ja", "target": "en"},
        }
        cfg_path = work / "config.yaml"
        cfg_path.write_text(
            yaml.dump(cfg_dict, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        config = AppConfig(**cfg_dict)
        state = load_state(work)

        # --- extract ---
        manifest = extract_epub(config, state, force=True)
        assert len(manifest.spine) > 0

        # --- clean ---
        manifest = clean_all(config, manifest, state, force=True)

        # --- classify (mocked) ---
        mock_classify_response = ClassificationResponse(
            classification="chapter",
            title="Test Chapter",
            confidence="high",
            reasoning="Prose content.",
        )

        with patch("dao_bridge.classify.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_classify_response
            mock_cls.return_value = mock_client
            manifest = run_classify_stage(work, config, state, force=True)

        # --- chunk ---
        manifest = chunk_all(config, manifest, state, force=True)
        total_chunks = sum(i.chunk_count or 0 for i in manifest.spine)
        assert total_chunks > 0

        # --- glossary-build (mocked) ---
        mock_build_response = _mock_extraction_response(
            entries=[
                {
                    "source_term": "スバル",
                    "reading": "すばる",
                    "english_proposed": "Subaru",
                    "category": "character",
                    "aliases": ["ナツキ・スバル"],
                    "speech_style": "Casual speech.",
                },
                {
                    "source_term": "エミリア",
                    "english_proposed": "Emilia",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_build_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        assert len(glossary.entries) >= 2
        assert any(e.english == "Subaru" for e in glossary.entries)
        assert any(e.english == "Emilia" for e in glossary.entries)

        # Verify glossary.json exists on disk.
        gp = glossary_path(work)
        assert gp.exists()
        disk_glossary = Glossary(**json.loads(gp.read_text(encoding="utf-8")))
        assert len(disk_glossary.entries) >= 2

        # --- glossary-reconcile (no conflicts, so no-op) ---
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)

        # State should show both glossary stages completed.
        assert state.stages.get("glossary_build") is not None
        assert state.stages["glossary_build"].status == "completed"
        assert state.stages.get("glossary_reconcile") is not None
        assert state.stages["glossary_reconcile"].status == "completed"

        # Report should exist.
        report_path = work / "glossary_reconcile_report.md"
        assert report_path.exists()
