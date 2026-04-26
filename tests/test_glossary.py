"""Tests for dao_bridge.glossary — build, reconcile, export, and integration.

Rewritten for entity-centric v2 glossary schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.config import AppConfig
from dao_bridge.glossary import (
    _BuildMeta,
    _GlossaryBatch,
    _build_work_items,
    _group_chunks_by_spine,
    _load_build_meta,
    _load_glossary,
    _merge_extraction_into_glossary,
    _pack_spine_batches,
    _rebalance_final_two_batches,
    _save_build_meta,
    _save_glossary,
    add_or_update_surface_form,
    find_entity_for_mention,
    glossary_build,
    glossary_export,
    glossary_reconcile,
    merge_aliases_nicknames_speech_notes,
    merge_entity_summary,
    next_entity_id,
    validate_glossary_categories,
)
from dao_bridge.schemas import (
    Chunk,
    ExtractedMention,
    Glossary,
    GlossaryCorrectionEntry,
    GlossaryEntity,
    GlossaryExtractionResponse,
    GlossaryReconcileResponse,
    GlossarySpeechMergeResponse,
    Manifest,
    ManifestItem,
    SurfaceForm,
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
    mentions: list[dict] | None = None,
    corrections: list[dict] | None = None,
) -> GlossaryExtractionResponse:
    """Build a GlossaryExtractionResponse from simple dicts."""
    mention_objs = []
    for m in mentions or []:
        mention_objs.append(ExtractedMention(**m))
    corr_objs = []
    for c in corrections or []:
        corr_objs.append(GlossaryCorrectionEntry(**c))
    return GlossaryExtractionResponse(mentions=mention_objs, corrections=corr_objs)


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


def _make_entity(
    entity_id: str = "character_000001",
    category: str = "character",
    canonical_english: str = "Subaru",
    surface_forms: list[dict] | None = None,
    **kwargs,
) -> GlossaryEntity:
    """Helper to create a GlossaryEntity with sensible defaults."""
    sfs = []
    for sf_data in surface_forms or []:
        sfs.append(SurfaceForm(**sf_data))
    return GlossaryEntity(
        entity_id=entity_id,
        category=category,
        canonical_english=canonical_english,
        surface_forms=sfs,
        source="extracted",
        **kwargs,
    )


def _make_mention(
    source: str = "スバル",
    english: str = "Subaru",
    category: str = "character",
    **kwargs,
) -> ExtractedMention:
    """Helper to create an ExtractedMention with sensible defaults."""
    return ExtractedMention(source=source, english=english, category=category, **kwargs)


# ---------------------------------------------------------------------------
# TestBatchPacking
# ---------------------------------------------------------------------------


class TestSpineBatchPacking:
    """Tests for the spine-aware batch packing algorithm."""

    # Default balancing params used across tests unless overridden.
    TARGET = 1000
    MIN_BATCH = 100
    THRESHOLD = 0.4  # 400 tokens

    def _make_chunks(self, count: int, token_count: int, spine_index: int = 0):
        """Create a list of chunks for a single spine."""
        return [
            Chunk(
                chunk_id=f"{spine_index:04d}.{i:03d}",
                spine_index=spine_index,
                chunk_index=i,
                source_file=f"clean/{spine_index:04d}.md",
                block_range=(0, 5),
                token_count=token_count,
                text="text",
            )
            for i in range(1, count + 1)
        ]

    def test_single_batch_all_fit(self):
        """All chunks fit in one batch."""
        chunks = self._make_chunks(3, token_count=100)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_multiple_batches(self):
        """Chunks are split across multiple batches."""
        chunks = self._make_chunks(5, token_count=400)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_last_batch_larger_than_threshold(self):
        """Last batch above threshold is left as-is."""
        chunks = self._make_chunks(3, token_count=600)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_single_chunk(self):
        """Single chunk returns one batch."""
        chunks = self._make_chunks(1, token_count=5000)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_empty_input(self):
        """Empty chunk list returns no batches."""
        batches = _pack_spine_batches([], self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert batches == []

    def test_exact_fit(self):
        """Chunks that exactly fill the target go in one batch."""
        chunks = self._make_chunks(2, token_count=500)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    # --- Remainder balancing tests ---

    def test_absorb_tiny_final_batch(self):
        """Final batch below min_batch_tokens is absorbed into previous."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=tc,
                text="text",
            )
            for i, tc in enumerate([800, 800, 50], start=1)
        ]
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 2
        assert len(batches[0]) == 1  # [800]
        assert len(batches[1]) == 2  # [800, 50]

    def test_redistribute_small_final_batch(self):
        """Final batch between min_batch and threshold is redistributed."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=tc,
                text="text",
            )
            for i, tc in enumerate([450, 450, 450, 450, 450, 450, 200], start=1)
        ]
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 4
        assert len(batches[0]) == 2  # [450, 450]
        assert len(batches[1]) == 2  # [450, 450]
        assert len(batches[2]) == 1  # [450]
        assert len(batches[3]) == 2  # [450, 200]
        last_two_tokens = [
            sum(c.token_count for c in batches[2]),
            sum(c.token_count for c in batches[3]),
        ]
        assert last_two_tokens == [450, 650]

    def test_no_balancing_with_single_batch(self):
        """Single-batch spine has no balancing (nothing to absorb into)."""
        chunks = self._make_chunks(1, token_count=50)
        batches = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_balancing_is_deterministic(self):
        """Same input always produces the same batches."""
        chunks = [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=tc,
                text="text",
            )
            for i, tc in enumerate([450, 450, 450, 450, 450, 450, 200], start=1)
        ]
        result1 = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        result2 = _pack_spine_batches(chunks, self.TARGET, self.MIN_BATCH, self.THRESHOLD)
        assert len(result1) == len(result2)
        for b1, b2 in zip(result1, result2):
            assert [c.chunk_id for c in b1] == [c.chunk_id for c in b2]


class TestRebalanceFinalTwoBatches:
    """Tests for _rebalance_final_two_batches."""

    def _make_chunks(self, token_counts: list[int]):
        return [
            Chunk(
                chunk_id=f"0000.{i:03d}",
                spine_index=0,
                chunk_index=i,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=tc,
                text="text",
            )
            for i, tc in enumerate(token_counts, start=1)
        ]

    def test_even_split(self):
        """Perfectly even token totals split in the middle."""
        prev = self._make_chunks([500, 500])
        last = self._make_chunks([500, 500])
        new_prev, new_last = _rebalance_final_two_batches(prev, last)
        total_left = sum(c.token_count for c in new_prev)
        total_right = sum(c.token_count for c in new_last)
        assert total_left == total_right

    def test_tie_prefers_earlier_split(self):
        """Equal split candidates choose the earlier chunk boundary."""
        chunks = self._make_chunks([2000, 2000, 2000, 2000])
        prev = chunks[:3]
        last = chunks[3:]
        new_prev, new_last = _rebalance_final_two_batches(prev, last)
        assert len(new_prev) == 2
        assert len(new_last) == 2

    def test_uneven_picks_closest(self):
        """Uneven chunks pick the boundary closest to half."""
        prev = self._make_chunks([800, 800])
        last = self._make_chunks([300])
        new_prev, new_last = _rebalance_final_two_batches(prev, last)
        assert len(new_prev) == 1
        assert len(new_last) == 2

    def test_single_chunk_each_unchanged(self):
        """Single-chunk batches can't be rebalanced further."""
        prev = self._make_chunks([800])
        last = self._make_chunks([200])
        new_prev, new_last = _rebalance_final_two_batches(prev, last)
        assert len(new_prev) == 1
        assert len(new_last) == 1

    def test_single_chunk_total_returned_unchanged(self):
        """A single chunk in combined returns the original batches."""
        prev = self._make_chunks([800])
        new_prev, new_last = _rebalance_final_two_batches(prev, [])
        assert new_prev == prev
        assert new_last == []


class TestGroupChunksBySpine:
    """Tests for _group_chunks_by_spine."""

    def test_groups_by_spine(self):
        """Chunks from different spines are grouped correctly."""
        chunks = [
            Chunk(
                chunk_id="0000.001",
                spine_index=0,
                chunk_index=1,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            ),
            Chunk(
                chunk_id="0000.002",
                spine_index=0,
                chunk_index=2,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            ),
            Chunk(
                chunk_id="0001.001",
                spine_index=1,
                chunk_index=1,
                source_file="clean/0001.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            ),
        ]
        groups = _group_chunks_by_spine(chunks)
        assert len(groups) == 2
        assert groups[0][0] == 0
        assert len(groups[0][1]) == 2
        assert groups[1][0] == 1
        assert len(groups[1][1]) == 1

    def test_empty_input(self):
        """Empty chunk list returns no groups."""
        assert _group_chunks_by_spine([]) == []


class TestBuildWorkItems:
    """Tests for _build_work_items which creates spine-aligned _GlossaryBatch objects."""

    def test_single_spine_single_batch(self):
        """One spine with chunks fitting in one batch."""
        chunks = [
            Chunk(
                chunk_id="0000.001",
                spine_index=0,
                chunk_index=1,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            ),
        ]
        items = _build_work_items(
            chunks,
            target_tokens=1000,
            min_batch_tokens=100,
            redistribute_threshold=0.4,
            spine_width=4,
        )
        assert len(items) == 1
        batch = items[0]
        assert isinstance(batch, _GlossaryBatch)
        assert batch.item_id == "0000.b1"
        assert len(batch.chunks) == 1
        assert batch.spine_batch_count == 1
        assert batch.spine_index == 0
        assert batch.token_count == 100
        assert batch.chunk_range_label == "0000.001"

    def test_multi_spine_multi_batch(self):
        """Multiple spines, some needing multiple batches."""
        chunks = []
        for i in range(1, 4):
            chunks.append(
                Chunk(
                    chunk_id=f"0000.{i:03d}",
                    spine_index=0,
                    chunk_index=i,
                    source_file="clean/0000.md",
                    block_range=(0, 5),
                    token_count=400,
                    text="text",
                )
            )
        chunks.append(
            Chunk(
                chunk_id="0001.001",
                spine_index=1,
                chunk_index=1,
                source_file="clean/0001.md",
                block_range=(0, 5),
                token_count=200,
                text="text",
            )
        )

        items = _build_work_items(
            chunks,
            target_tokens=1000,
            min_batch_tokens=100,
            redistribute_threshold=0.4,
            spine_width=4,
        )
        assert len(items) == 3
        assert items[0].item_id == "0000.b1"
        assert items[0].spine_batch_count == 2
        assert items[0].chunk_range_label == "0000.001-0000.002"
        assert items[1].item_id == "0000.b2"
        assert items[1].spine_batch_count == 2
        assert items[1].chunk_range_label == "0000.003"
        assert items[2].item_id == "0001.b1"
        assert items[2].spine_batch_count == 1

    def test_item_ids_are_deterministic(self):
        """Same input always produces same item IDs."""
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
        items1 = _build_work_items(
            chunks,
            target_tokens=1000,
            min_batch_tokens=100,
            redistribute_threshold=0.4,
            spine_width=4,
        )
        items2 = _build_work_items(
            chunks,
            target_tokens=1000,
            min_batch_tokens=100,
            redistribute_threshold=0.4,
            spine_width=4,
        )
        ids1 = [b.item_id for b in items1]
        ids2 = [b.item_id for b in items2]
        assert ids1 == ids2

    def test_chunks_stored_as_tuples(self):
        """Batch chunks are immutable tuples, not lists."""
        chunks = [
            Chunk(
                chunk_id="0000.001",
                spine_index=0,
                chunk_index=1,
                source_file="clean/0000.md",
                block_range=(0, 5),
                token_count=100,
                text="text",
            ),
        ]
        items = _build_work_items(
            chunks,
            target_tokens=1000,
            min_batch_tokens=100,
            redistribute_threshold=0.4,
            spine_width=4,
        )
        assert isinstance(items[0].chunks, tuple)


# ---------------------------------------------------------------------------
# TestCategoryValidation
# ---------------------------------------------------------------------------


class TestCategoryValidation:
    """Tests for validate_glossary_categories."""

    def test_valid_categories(self):
        """No error when all categories are valid."""
        g = Glossary(
            entities=[
                _make_entity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="X",
                    surface_forms=[{"source": "X", "english": "X"}],
                ),
                _make_entity(
                    entity_id="place_000001",
                    category="place",
                    canonical_english="Y",
                    surface_forms=[{"source": "Y", "english": "Y"}],
                ),
            ]
        )
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
            entities=[
                _make_entity(
                    entity_id="weapon_000001",
                    category="weapon",
                    canonical_english="Foo",
                ),
                _make_entity(
                    entity_id="magic_000001",
                    category="magic",
                    canonical_english="Bar",
                ),
            ]
        )
        with pytest.raises(ValueError, match="weapon"):
            validate_glossary_categories(g, ["character", "place"])

    def test_error_lists_affected_entities(self):
        """Error message lists which entities use the invalid category."""
        g = Glossary(
            entities=[
                _make_entity(
                    entity_id="weapon_000001",
                    category="weapon",
                    canonical_english="Foo",
                ),
            ]
        )
        with pytest.raises(ValueError, match="Foo"):
            validate_glossary_categories(g, ["character"])

    def test_empty_glossary_passes(self):
        """Empty glossary always passes."""
        validate_glossary_categories(Glossary(), ["character"])


# ---------------------------------------------------------------------------
# TestEntityLinking
# ---------------------------------------------------------------------------


class TestEntityLinking:
    """Tests for build-time entity linking (find_entity_for_mention)."""

    def test_exact_surface_form_match(self):
        """Exact source match returns the entity."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="スバル", english="Subaru")
        result = find_entity_for_mention(glossary, mention)
        assert result is entity

    def test_same_reading_and_english_match(self):
        """Same non-null reading AND English returns the entity."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "reading": "すばる", "english": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="ナツキ・スバル", reading="すばる", english="Subaru")
        result = find_entity_for_mention(glossary, mention)
        assert result is entity

    def test_no_match_returns_none(self):
        """When no entity matches, returns None (create new entity)."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="エミリア", english="Emilia")
        result = find_entity_for_mention(glossary, mention)
        assert result is None

    def test_null_reading_does_not_match(self):
        """Null reading should not match other null readings."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "reading": None, "english": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="ナツキ", reading=None, english="Subaru")
        result = find_entity_for_mention(glossary, mention)
        # Exact source doesn't match, null reading can't match — should be None.
        assert result is None

    def test_different_category_blocks_jaro_winkler(self):
        """High-similarity source with different category does not match."""
        entity = _make_entity(
            entity_id="character_000001",
            category="character",
            surface_forms=[{"source": "アベル", "english": "Abel"}],
        )
        glossary = Glossary(entities=[entity])
        # Very similar source but different category.
        mention = _make_mention(source="アベル座", english="Abelza", category="place")
        # Exact source doesn't match, reading doesn't match, category differs.
        result = find_entity_for_mention(glossary, mention)
        assert result is None

    def test_ambiguous_jaro_winkler_match_returns_none(self):
        """High-similarity fallback only auto-attaches when unique."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Abel",
                    surface_forms=[{"source": "アベル", "english": "Abel"}],
                ),
                _make_entity(
                    entity_id="character_000002",
                    category="character",
                    canonical_english="Abe",
                    surface_forms=[{"source": "アベル", "english": "Abe"}],
                ),
            ]
        )
        mention = _make_mention(source="アベルー", english="Abel", category="character")
        assert find_entity_for_mention(glossary, mention) is None


# ---------------------------------------------------------------------------
# TestEntityIdGeneration
# ---------------------------------------------------------------------------


class TestEntityIdGeneration:
    """Tests for next_entity_id."""

    def test_first_entity_for_category(self):
        """First entity gets 000001."""
        glossary = Glossary()
        eid = next_entity_id("character", glossary)
        assert eid == "character_000001"

    def test_increments_from_existing(self):
        """ID increments from highest existing for the category."""
        glossary = Glossary(
            entities=[
                _make_entity(entity_id="character_000003", category="character"),
                _make_entity(entity_id="character_000001", category="character"),
            ]
        )
        eid = next_entity_id("character", glossary)
        assert eid == "character_000004"

    def test_different_categories_independent(self):
        """Different categories have independent counters."""
        glossary = Glossary(
            entities=[
                _make_entity(entity_id="character_000005", category="character"),
                _make_entity(entity_id="place_000002", category="place"),
            ]
        )
        assert next_entity_id("character", glossary) == "character_000006"
        assert next_entity_id("place", glossary) == "place_000003"
        assert next_entity_id("item", glossary) == "item_000001"


# ---------------------------------------------------------------------------
# TestCorrectionRouting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestSurfaceFormMerge
# ---------------------------------------------------------------------------


class TestSurfaceFormMerge:
    """Tests for add_or_update_surface_form."""

    def test_new_surface_form_added(self):
        """New source creates a new surface form on the entity."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru"}],
        )
        mention = _make_mention(source="ナツキ・スバル", english="Natsuki Subaru")
        add_or_update_surface_form(entity, mention, "0000.001")
        assert len(entity.surface_forms) == 2
        new_sf = entity.surface_forms[1]
        assert new_sf.source == "ナツキ・スバル"
        assert new_sf.english == "Natsuki Subaru"
        assert new_sf.first_seen_chunk == "0000.001"

    def test_existing_source_increments_count(self):
        """Repeat source increments occurrence_count."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru", "occurrence_count": 1}],
        )
        mention = _make_mention(source="スバル", english="Subaru")
        add_or_update_surface_form(entity, mention, "0000.002")
        assert len(entity.surface_forms) == 1
        assert entity.surface_forms[0].occurrence_count == 2

    def test_context_hint_appended(self):
        """Context hint is appended to existing surface form."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru"}],
        )
        mention = _make_mention(
            source="スバル", english="Subaru", context_hint="possibly same as ナツキ"
        )
        add_or_update_surface_form(entity, mention, "0000.001")
        assert "possibly same as ナツキ" in entity.surface_forms[0].context_hints

    def test_reading_backfilled(self):
        """Missing reading on existing form is backfilled from mention."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "english": "Subaru", "reading": None}],
        )
        mention = _make_mention(source="スバル", english="Subaru", reading="すばる")
        add_or_update_surface_form(entity, mention, "0000.001")
        assert entity.surface_forms[0].reading == "すばる"


# ---------------------------------------------------------------------------
# TestSummaryMerge
# ---------------------------------------------------------------------------


class TestSummaryMerge:
    """Tests for merge_entity_summary."""

    def test_first_summary_stored(self):
        """First summary is stored directly."""
        entity = _make_entity(summary=None)
        merge_entity_summary(entity, "A young man from another world.", "0000.001")
        assert entity.summary == "A young man from another world."
        assert entity.latest_evidence_chunk == "0000.001"

    def test_new_observation_appended(self):
        """New observation is appended."""
        entity = _make_entity(summary="A young man from another world.")
        merge_entity_summary(entity, "He has Return by Death.", "0000.005")
        assert "He has Return by Death." in entity.summary
        assert entity.latest_evidence_chunk == "0000.005"

    def test_duplicate_observation_not_appended(self):
        """Identical observation is not duplicated."""
        entity = _make_entity(summary="A young man from another world.")
        merge_entity_summary(entity, "A young man from another world.", "0000.005")
        assert entity.summary == "A young man from another world."

    def test_null_update_ignored(self):
        """None summary_update is ignored."""
        entity = _make_entity(summary="Existing.")
        merge_entity_summary(entity, None, "0000.001")
        assert entity.summary == "Existing."

    def test_summary_truncated_at_max_length(self):
        """Summary is truncated if it exceeds max length."""
        entity = _make_entity(summary="A" * 400)
        merge_entity_summary(entity, "B" * 200, "0000.001")
        assert len(entity.summary) <= 503  # 500 + "..."


# ---------------------------------------------------------------------------
# TestGlossaryBuild
# ---------------------------------------------------------------------------


class TestGlossaryBuild:
    """Tests for the glossary_build function."""

    def test_entities_created_correctly(self, tmp_path):
        """New entities are created with correct fields."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "ナツキ・スバル",
                    "reading": "ナツキ・スバル",
                    "english": "Natsuki Subaru",
                    "category": "character",
                    "aliases": ["スバル"],
                    "nicknames": {},
                    "speech_style": "Casual modern speech.",
                    "notes": "Protagonist.",
                    "summary_update": "A young man transported to another world.",
                },
                {
                    "source": "エミリア",
                    "english": "Emilia",
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

        assert len(glossary.entities) == 2
        subaru = next(e for e in glossary.entities if e.canonical_english == "Natsuki Subaru")
        assert subaru.entity_id == "character_000001"
        assert subaru.source == "extracted"
        assert "スバル" in subaru.aliases
        assert subaru.speech_style == "Casual modern speech."
        assert subaru.summary == "A young man transported to another world."
        assert subaru.first_seen_chunk is not None
        assert len(subaru.surface_forms) == 1
        assert subaru.surface_forms[0].source == "ナツキ・スバル"
        assert subaru.surface_forms[0].english == "Natsuki Subaru"

        emilia = next(e for e in glossary.entities if e.canonical_english == "Emilia")
        assert emilia.nicknames["Subaru"] == "Emilia-tan"

        # book_id populated from manifest.
        assert glossary.book_id == "test-book"

    def test_aliases_unioned_across_batches(self, tmp_path):
        """Aliases from later batches are merged with existing."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
                    "category": "character",
                    "aliases": ["バルス"],
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
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

        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru")
        assert "バルス" in subaru.aliases
        assert "ナツキ" in subaru.aliases
        assert subaru.aliases.count("バルス") == 1

    def test_speech_styles_accumulated(self, tmp_path):
        """Speech style observations are accumulated with newline delimiter."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
                    "category": "character",
                    "speech_style": "Casual speech.",
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
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

        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru")
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
            mentions=[
                {
                    "source": "プリシラ",
                    "english": "Priscilla",
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
        priscilla = next(e for e in glossary.entities if e.canonical_english == "Priscilla")
        assert priscilla.canonical_english == "Priscilla"

        # But it should be in the build meta conflicts.
        meta = _load_build_meta(work)
        assert len(meta.corrections) == 1
        assert meta.corrections[0]["corrected_english"] == "Priscilla Barielle"
        assert len(meta.conflicts) >= 1

    def test_correction_prefers_unique_source_form_when_english_is_ambiguous(self, tmp_path):
        """Correction routing falls back to a unique source-form match."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        # Pre-seed two entities sharing the same canonical_english.
        _save_glossary(
            work,
            Glossary(
                entities=[
                    GlossaryEntity(
                        entity_id="character_000001",
                        category="character",
                        canonical_english="Priscilla",
                        surface_forms=[SurfaceForm(source="プリシラ", english="Priscilla")],
                        source="extracted",
                    ),
                    GlossaryEntity(
                        entity_id="character_000002",
                        category="character",
                        canonical_english="Priscilla",
                        surface_forms=[
                            SurfaceForm(
                                source="プリシラ・バーリエル",
                                english="Priscilla Barielle",
                            )
                        ],
                        source="extracted",
                    ),
                ]
            ),
        )

        mock_response = _mock_extraction_response(
            mentions=[],
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
            glossary_build(work, config, state, force=False)

        # canonical_english is ambiguous (two entities share "Priscilla"),
        # so the correction should fall back to the source_form signal
        # and attach to entity_000002 (which owns "プリシラ・バーリエル").
        meta = _load_build_meta(work)
        conflict = next(c for c in meta.conflicts if c.source_form == "プリシラ・バーリエル")
        assert conflict.entity_id == "character_000002"

    def test_user_sourced_entities_never_modified(self, tmp_path):
        """Entities with source='user' are never modified by build."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        # Pre-seed a user entity.
        pre_glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru (custom)",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru (custom)")],
                    source="user",
                    aliases=["original_alias"],
                )
            ]
        )
        _save_glossary(work, pre_glossary)

        mock_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Natsuki Subaru",
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

        # User entity should be unchanged.
        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru (custom)")
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

        mock_response = _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, force=True)

        batch_items = [k for k in state.items.keys() if k.startswith("glossary_build:")]
        assert len(batch_items) >= 1
        for item_key in batch_items:
            assert state.items[item_key].status == "completed"

    def test_surface_form_occurrence_count_increments(self, tmp_path):
        """Repeated mentions of the same source increment occurrence_count."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            mentions=[{"source": "スバル", "english": "Subaru", "category": "character"}]
        )
        batch2_response = _mock_extraction_response(
            mentions=[{"source": "スバル", "english": "Subaru", "category": "character"}]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru")
        assert subaru.surface_forms[0].occurrence_count == 2

    def test_new_surface_form_added_to_existing_entity(self, tmp_path):
        """A variant form attaches as a new surface form on an existing entity."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary_phase.target_tokens_per_call = 1200
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        # Batch 1: mention スバル. Batch 2: mention スバル again with same reading+english
        # (so it attaches) but also a mention with different source that shares
        # reading+english (attaches as new surface form).
        batch1_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
                    "reading": "すばる",
                    "category": "character",
                }
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "ナツキ・スバル",
                    "english": "Subaru",
                    "reading": "すばる",
                    "category": "character",
                }
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        # Should be one entity with two surface forms.
        assert len(glossary.entities) == 1
        subaru = glossary.entities[0]
        assert len(subaru.surface_forms) == 2
        sources = {sf.source for sf in subaru.surface_forms}
        assert sources == {"スバル", "ナツキ・スバル"}


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
        _make_manifest(work, n_spines=1, chunks_per_spine=4)
        _write_chunks(work, n_spines=1, chunks_per_spine=4, tokens_per_chunk=500)

        batch1_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "エミリア",
                    "english": "Emilia",
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
        assert len(glossary.entities) == 1
        assert glossary.entities[0].canonical_english == "Emilia"

        # Second run: resume from batch 2.
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "レム",
                    "english": "Rem",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = batch2_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=False)

        assert len(glossary.entities) == 2
        assert any(e.canonical_english == "Emilia" for e in glossary.entities)
        assert any(e.canonical_english == "Rem" for e in glossary.entities)


# ---------------------------------------------------------------------------
# TestBuildTargeted
# ---------------------------------------------------------------------------


class TestBuildTargeted:
    """Tests for --spine and --batch targeted redo."""

    def _setup_completed_build(self, tmp_path):
        """Set up a work dir with a completed glossary build (2 spines, 2 chunks each)."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=2, chunks_per_spine=2)
        _write_chunks(work, n_spines=2, chunks_per_spine=2, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "english": "Subaru",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        assert state.stages["glossary_build"].status == "completed"
        return work, config, state

    def test_target_spine_redoes_only_that_spine(self, tmp_path):
        """--spine N resets and re-runs only that spine's batches."""
        work, config, state = self._setup_completed_build(tmp_path)

        spine0_items = [k for k in state.items if k.startswith("glossary_build:0000.")]
        spine1_items = [k for k in state.items if k.startswith("glossary_build:0001.")]
        assert all(state.items[k].status == "completed" for k in spine0_items)
        assert all(state.items[k].status == "completed" for k in spine1_items)

        new_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "エミリア",
                    "english": "Emilia",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = new_response
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, target_spine=0)

            call_count = mock_client.complete_json.call_count
            assert call_count >= 1

        refreshed_state = load_state(work)
        for k in spine1_items:
            assert refreshed_state.items[k].status == "completed"

    def test_target_batch_redoes_only_that_batch(self, tmp_path):
        """--batch ID resets and re-runs only that specific batch."""
        work, config, state = self._setup_completed_build(tmp_path)

        new_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "レム",
                    "english": "Rem",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = new_response
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, target_batch="0000.b1")

            assert mock_client.complete_json.call_count == 1

    def test_targeted_run_does_not_mark_stage_completed(self, tmp_path):
        """A targeted run should not mark the stage as completed."""
        work, config, state = self._setup_completed_build(tmp_path)

        new_response = _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = new_response
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, target_batch="0000.b1")

        assert state.stages["glossary_build"].status == "running"

    def test_invalid_batch_id_raises(self, tmp_path):
        """An invalid batch ID raises a clear error."""
        work, config, state = self._setup_completed_build(tmp_path)

        with pytest.raises(RuntimeError, match="not found"):
            glossary_build(work, config, state, target_batch="9999.b1")

    def test_invalid_spine_raises(self, tmp_path):
        """A spine with no batches raises a clear error."""
        work, config, state = self._setup_completed_build(tmp_path)

        with pytest.raises(RuntimeError, match="no glossary batches"):
            glossary_build(work, config, state, target_spine=99)

    def test_batch_takes_precedence_over_spine(self, tmp_path):
        """When both --batch and --spine are set, --batch wins."""
        work, config, state = self._setup_completed_build(tmp_path)

        new_response = _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = new_response
            mock_llm_cls.return_value = mock_client

            glossary_build(
                work,
                config,
                state,
                target_spine=1,
                target_batch="0000.b1",
            )

            assert mock_client.complete_json.call_count == 1


# ---------------------------------------------------------------------------
# TestGlossaryReconcile
# ---------------------------------------------------------------------------


class TestGlossaryReconcile:
    """Tests for the glossary_reconcile function."""

    def test_raises_when_cluster_not_completed(self, tmp_path):
        """Reconcile raises RuntimeError if glossary_cluster stage is not completed."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        # Build is done but cluster is not.
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")

        with pytest.raises(RuntimeError, match="Glossary cluster stage not completed"):
            glossary_reconcile(work, config, state, force=False)

    def _setup_with_conflicts(self, tmp_path):
        """Set up a work dir with a glossary and conflicts for reconcile."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        # Create glossary with an entity.
        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="place_000001",
                    category="place",
                    canonical_english="Lugnica",
                    surface_forms=[
                        SurfaceForm(source="ルグニカ", reading="るぐにか", english="Lugnica")
                    ],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary)

        # Create build meta with a conflict.
        meta = _BuildMeta(
            conflicts=[
                {
                    "entity_id": "place_000001",
                    "source_form": "ルグニカ",
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

        entity = next(e for e in glossary.entities if e.entity_id == "place_000001")
        assert entity.canonical_english == "Lugunica"

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
                entity = next(e for e in disk_glossary.entities if e.entity_id == "place_000001")
                assert entity.canonical_english == "Lugunica"
            return mark_item_completed(work_dir, pipeline_state, stage, item_id)

        with (
            patch("dao_bridge.glossary.LLMClient") as mock_llm_cls,
            patch(
                "dao_bridge.glossary.mark_item_completed",
                side_effect=checking_mark_item_completed,
            ),
        ):
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entity = next(e for e in glossary.entities if e.entity_id == "place_000001")
        assert entity.canonical_english == "Lugunica"

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
        assert "place_000001" in report
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
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
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

        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru")
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
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
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
                entity = next(e for e in disk_glossary.entities if e.canonical_english == "Subaru")
                assert entity.speech_style == "Speaks casually with modern slang."
            return mark_item_completed(work_dir, pipeline_state, stage, item_id)

        with (
            patch("dao_bridge.glossary.LLMClient") as mock_llm_cls,
            patch(
                "dao_bridge.glossary.mark_item_completed",
                side_effect=checking_mark_item_completed,
            ),
        ):
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_speech
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_english == "Subaru")
        assert subaru.speech_style == "Speaks casually with modern slang."

    def test_no_conflicts_completes_immediately(self, tmp_path):
        """Stage completes as no-op when no conflicts exist."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="X",
                    surface_forms=[SurfaceForm(source="X", english="X")],
                    source="extracted",
                    speech_style="Single observation only.",
                ),
            ]
        )
        _save_glossary(work, glossary)
        _save_build_meta(work, _BuildMeta())

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
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="item_000001",
                    category="item",
                    canonical_english="Holy Sword",
                    surface_forms=[SurfaceForm(source="聖剣", english="Holy Sword")],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary)

        meta = _BuildMeta(
            conflicts=[
                {
                    "entity_id": "item_000001",
                    "source_form": "聖剣",
                    "reading": None,
                    "current_english": "Holy Sword",
                    "alternatives": [],
                    "category_variants": ["ability"],
                }
            ],
        )
        _save_build_meta(work, meta)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)
            mock_llm_cls.assert_not_called()

        report = (work / "glossary_reconcile_report.md").read_text(encoding="utf-8")
        assert "item_000001" in report
        assert "Category" in report or "category" in report

    def test_build_validates_preexisting_categories(self, tmp_path):
        """Build stage validates categories on pre-existing glossary entities."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        bad_glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="weapon_000001",
                    category="weapon",
                    canonical_english="X",
                    surface_forms=[SurfaceForm(source="X", english="X")],
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
        """Entities are grouped by category and sorted by canonical_english."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entities=[
                _make_entity(
                    entity_id="character_000002",
                    canonical_english="Zorro",
                    surface_forms=[{"source": "B-char", "english": "Zorro"}],
                ),
                _make_entity(
                    entity_id="character_000001",
                    canonical_english="Alice",
                    surface_forms=[{"source": "A-char", "english": "Alice"}],
                ),
                _make_entity(
                    entity_id="place_000001",
                    category="place",
                    canonical_english="Kingdom",
                    surface_forms=[{"source": "Place1", "english": "Kingdom"}],
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        char_pos = md.index("## Character")
        place_pos = md.index("## Place")
        assert char_pos < place_pos

        alice_pos = md.index("Alice")
        zorro_pos = md.index("Zorro")
        assert alice_pos < zorro_pos

    def test_optional_fields_rendered_when_present(self, tmp_path):
        """All optional fields are rendered when present."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[
                        SurfaceForm(source="スバル", reading="すばる", english="Subaru")
                    ],
                    aliases=["バルス"],
                    nicknames={"Rem": "Subaru-kun"},
                    speech_style="Casual.",
                    notes="Main character.",
                    source="extracted",
                    summary="A young man from another world.",
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        assert "Summary: A young man from another world." in md
        assert "`スバル`" in md
        assert "-> Subaru" in md
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
            entities=[
                _make_entity(
                    entity_id="term_000001",
                    category="term",
                    canonical_english="Xterm",
                    surface_forms=[{"source": "X", "english": "Xterm"}],
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)

        assert "Aliases:" not in md
        assert "Nicknames:" not in md
        assert "Speech style:" not in md
        assert "Notes:" not in md

    def test_stdout_mode(self, tmp_path):
        """In stdout mode, no file is written."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entities=[
                _make_entity(
                    surface_forms=[{"source": "X", "english": "X"}],
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
            entities=[
                _make_entity(
                    surface_forms=[{"source": "X", "english": "X"}],
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
            entities=[
                _make_entity(
                    surface_forms=[{"source": "X", "english": "X"}],
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
        assert "No entities" in md

    def test_entity_id_in_export(self, tmp_path):
        """Entity ID appears in the exported markdown."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entities=[
                _make_entity(
                    entity_id="character_000042",
                    canonical_english="Test",
                    surface_forms=[{"source": "テスト", "english": "Test"}],
                ),
            ]
        )
        _save_glossary(work, glossary)

        md = glossary_export(work, config, stdout=True)
        assert "character_000042" in md


# ---------------------------------------------------------------------------
# TestLanguageAgnostic
# ---------------------------------------------------------------------------


class TestLanguageAgnostic:
    """Tests that prompts use resolved language names, not hardcoded values."""

    def test_prompt_uses_source_language(self, tmp_path):
        """The extraction prompt contains the resolved language name."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.languages.source = "zh"
        config.languages.target = "en"
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        captured_messages = []

        def capture_complete_json(messages, response_model=None, **kwargs):
            captured_messages.append(messages)
            return _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = capture_complete_json
            mock_llm_cls.return_value = mock_client

            glossary_build(work, config, state, force=True)

        assert len(captured_messages) == 1
        prompt_content = captured_messages[0][0]["content"]
        assert "Chinese" in prompt_content
        assert "English" in prompt_content
        assert "Japanese" not in prompt_content

    def test_build_does_not_inject_accumulated_glossary(self, tmp_path):
        """Phase 1 extraction prompt omits the full accumulated glossary."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)
        _save_glossary(
            work,
            Glossary(
                entities=[
                    GlossaryEntity(
                        entity_id="character_000001",
                        category="character",
                        canonical_english="Hidden Subaru",
                        surface_forms=[
                            SurfaceForm(source="ナツキ・スバル", english="Hidden Subaru")
                        ],
                        source="extracted",
                    )
                ]
            ),
        )

        captured_messages = []

        def capture_complete_json(messages, response_model=None, **kwargs):
            captured_messages.append(messages)
            return _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = capture_complete_json
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=False)

        prompt_content = captured_messages[0][0]["content"]
        assert "Hidden Subaru" not in prompt_content
        assert "ナツキ・スバル -> Hidden Subaru" not in prompt_content
        assert "(not provided in phase 1)" in prompt_content


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
            mentions=[
                {
                    "source": "スバル",
                    "reading": "すばる",
                    "english": "Subaru",
                    "category": "character",
                    "aliases": ["ナツキ・スバル"],
                    "speech_style": "Casual speech.",
                    "summary_update": "A young man from another world.",
                },
                {
                    "source": "エミリア",
                    "english": "Emilia",
                    "category": "character",
                },
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_build_response
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        assert len(glossary.entities) >= 2
        assert any(e.canonical_english == "Subaru" for e in glossary.entities)
        assert any(e.canonical_english == "Emilia" for e in glossary.entities)

        # Verify glossary.json exists on disk.
        gp = glossary_path(work)
        assert gp.exists()
        disk_glossary = Glossary(**json.loads(gp.read_text(encoding="utf-8")))
        assert len(disk_glossary.entities) >= 2

        # --- glossary-cluster (no duplicates, so no-op) ---
        from dao_bridge.glossary import glossary_cluster

        glossary = glossary_cluster(work, config, state, force=True)

        assert state.stages.get("glossary_cluster") is not None
        assert state.stages["glossary_cluster"].status == "completed"

        cluster_report_path = work / "glossary_cluster_report.md"
        assert cluster_report_path.exists()

        # --- glossary-reconcile (no conflicts, so no-op) ---
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)

        assert state.stages.get("glossary_build") is not None
        assert state.stages["glossary_build"].status == "completed"
        assert state.stages.get("glossary_reconcile") is not None
        assert state.stages["glossary_reconcile"].status == "completed"

        report_path = work / "glossary_reconcile_report.md"
        assert report_path.exists()
