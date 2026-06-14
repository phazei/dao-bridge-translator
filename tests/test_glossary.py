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
    _pack_spine_batches,
    _rebalance_final_two_batches,
    _save_build_meta,
    _save_glossary,
    add_or_update_surface_form,
    compress_entity_summaries,
    compress_entity_summary,
    find_entity_for_mention,
    glossary_build,
    glossary_export,
    glossary_reconcile,
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
    GlossarySummaryCompressResponse,
    Manifest,
    ManifestItem,
    SummaryObservation,
    SurfaceForm,
)
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    load_state,
    mark_item_completed,
    mark_stage_completed,
    mark_stage_started,
    save_state,
)
from dao_bridge.workdir import (
    atomic_write,
    chunk_dir,
    glossary_build_path,
    glossary_cluster_path,
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
    canonical_name: str = "Subaru",
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
        canonical_name=canonical_name,
        surface_forms=sfs,
        source="extracted",
        **kwargs,
    )


def _make_mention(
    source: str = "スバル",
    translation: str = "Subaru",
    category: str = "character",
    **kwargs,
) -> ExtractedMention:
    """Helper to create an ExtractedMention with sensible defaults."""
    return ExtractedMention(source=source, translation=translation, category=category, **kwargs)


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
                    canonical_name="X",
                    surface_forms=[{"source": "X", "translation": "X"}],
                ),
                _make_entity(
                    entity_id="place_000001",
                    category="place",
                    canonical_name="Y",
                    surface_forms=[{"source": "Y", "translation": "Y"}],
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
                    canonical_name="Foo",
                ),
                _make_entity(
                    entity_id="magic_000001",
                    category="magic",
                    canonical_name="Bar",
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
                    canonical_name="Foo",
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
            surface_forms=[{"source": "スバル", "translation": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="スバル", translation="Subaru")
        result = find_entity_for_mention(glossary, mention)
        assert result is entity

    def test_same_reading_and_translation_match(self):
        """Same non-null reading AND translation returns the entity."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "reading": "すばる", "translation": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="ナツキ・スバル", reading="すばる", translation="Subaru")
        result = find_entity_for_mention(glossary, mention)
        assert result is entity

    def test_no_match_returns_none(self):
        """When no entity matches, returns None (create new entity)."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "translation": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="エミリア", translation="Emilia")
        result = find_entity_for_mention(glossary, mention)
        assert result is None

    def test_null_reading_does_not_match(self):
        """Null reading should not match other null readings."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "reading": None, "translation": "Subaru"}],
        )
        glossary = Glossary(entities=[entity])
        mention = _make_mention(source="ナツキ", reading=None, translation="Subaru")
        result = find_entity_for_mention(glossary, mention)
        # Exact source doesn't match, null reading can't match — should be None.
        assert result is None

    def test_different_category_blocks_jaro_winkler(self):
        """High-similarity source with different category does not match."""
        entity = _make_entity(
            entity_id="character_000001",
            category="character",
            surface_forms=[{"source": "アベル", "translation": "Abel"}],
        )
        glossary = Glossary(entities=[entity])
        # Very similar source but different category.
        mention = _make_mention(source="アベル座", translation="Abelza", category="place")
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
                    canonical_name="Abel",
                    surface_forms=[{"source": "アベル", "translation": "Abel"}],
                ),
                _make_entity(
                    entity_id="character_000002",
                    category="character",
                    canonical_name="Abe",
                    surface_forms=[{"source": "アベル", "translation": "Abe"}],
                ),
            ]
        )
        mention = _make_mention(source="アベルー", translation="Abel", category="character")
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
            surface_forms=[{"source": "スバル", "translation": "Subaru"}],
        )
        mention = _make_mention(source="ナツキ・スバル", translation="Natsuki Subaru")
        add_or_update_surface_form(entity, mention, "0000.001")
        assert len(entity.surface_forms) == 2
        new_sf = entity.surface_forms[1]
        assert new_sf.source == "ナツキ・スバル"
        assert new_sf.translation == "Natsuki Subaru"
        assert new_sf.first_seen_chunk == "0000.001"

    def test_existing_source_increments_count(self):
        """Repeat source increments occurrence_count."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "translation": "Subaru", "occurrence_count": 1}],
        )
        mention = _make_mention(source="スバル", translation="Subaru")
        add_or_update_surface_form(entity, mention, "0000.002")
        assert len(entity.surface_forms) == 1
        assert entity.surface_forms[0].occurrence_count == 2

    def test_context_hint_appended(self):
        """Context hint is appended to existing surface form."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "translation": "Subaru"}],
        )
        mention = _make_mention(
            source="スバル", translation="Subaru", context_hint="possibly same as ナツキ"
        )
        add_or_update_surface_form(entity, mention, "0000.001")
        assert "possibly same as ナツキ" in entity.surface_forms[0].context_hints

    def test_reading_backfilled(self):
        """Missing reading on existing form is backfilled from mention."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "translation": "Subaru", "reading": None}],
        )
        mention = _make_mention(source="スバル", translation="Subaru", reading="すばる")
        add_or_update_surface_form(entity, mention, "0000.001")
        assert entity.surface_forms[0].reading == "すばる"

    def test_same_source_different_translation_adds_variant(self):
        """Same source with different translation is preserved in translation_variants."""
        entity = _make_entity(
            surface_forms=[{"source": "スバル", "translation": "Subaru"}],
        )
        mention = _make_mention(source="スバル", translation="Subaru Natsuki")
        add_or_update_surface_form(entity, mention, "0000.001")
        sf = entity.surface_forms[0]
        assert sf.translation == "Subaru"
        assert sf.translation_variants == ["Subaru Natsuki"]

    def test_same_source_different_translation_deduped(self):
        """Repeat same-source variant is not duplicated."""
        entity = _make_entity(
            surface_forms=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "translation_variants": ["Subaru Natsuki"],
                }
            ],
        )
        mention = _make_mention(source="スバル", translation="Subaru Natsuki")
        add_or_update_surface_form(entity, mention, "0000.001")
        sf = entity.surface_forms[0]
        assert sf.translation_variants == ["Subaru Natsuki"]


# ---------------------------------------------------------------------------
# TestMentionConflictRouting
# ---------------------------------------------------------------------------


class TestMentionConflictRouting:
    """Same-source translation disagreements should NOT create entity-level ConflictRecord."""

    def test_same_source_different_translation_no_entity_conflict(self):
        """Same source with different translation creates translation_variants only."""
        from dao_bridge.glossary import _BuildMeta, _merge_mention_into_glossary

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru")],
                    source="extracted",
                ),
            ]
        )
        meta = _BuildMeta()

        mention = ExtractedMention(
            source="スバル",
            translation="Subaru Natsuki",
            category="character",
        )
        _merge_mention_into_glossary(glossary, mention, "0001.b1", "0001.001", meta)

        # Should NOT have created an entity-level conflict.
        assert len(meta.conflicts) == 0
        # Should be captured in translation_variants instead.
        sf = glossary.entities[0].surface_forms[0]
        assert "Subaru Natsuki" in sf.translation_variants

    def test_different_source_different_translation_adds_surface_form(self):
        """New source form with different translation adds a surface form, no conflict."""
        from dao_bridge.glossary import _BuildMeta, _merge_mention_into_glossary

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru")],
                    source="extracted",
                ),
            ]
        )
        meta = _BuildMeta()

        # Force entity linking to return the existing entity even though
        # the source form is different.  In practice this happens via
        # Jaro-Winkler >= 0.95 or reading+translation match.
        mention = ExtractedMention(
            source="ナツキ・スバル",
            translation="Subaru Natsuki",
            category="character",
        )
        with patch(
            "dao_bridge.glossary.find_entity_for_mention",
            return_value=glossary.entities[0],
        ):
            _merge_mention_into_glossary(glossary, mention, "0001.b1", "0001.001", meta)

        # Verify it attached to the existing entity.
        assert len(glossary.entities) == 1
        # New source form with a legitimately different translation should
        # NOT create an entity-level conflict — it is simply another valid
        # surface form (e.g. full name vs short name).
        assert len(meta.conflicts) == 0
        # The new surface form should be present on the entity.
        sources = {sf.source for sf in glossary.entities[0].surface_forms}
        assert "ナツキ・スバル" in sources
        translations = {sf.translation for sf in glossary.entities[0].surface_forms}
        assert "Subaru Natsuki" in translations

    def test_same_source_same_translation_no_conflict(self):
        """Same source with same translation creates no conflicts at all."""
        from dao_bridge.glossary import _BuildMeta, _merge_mention_into_glossary

        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru")],
                    source="extracted",
                ),
            ]
        )
        meta = _BuildMeta()

        mention = ExtractedMention(
            source="スバル",
            translation="Subaru",
            category="character",
        )
        _merge_mention_into_glossary(glossary, mention, "0001.b1", "0001.001", meta)

        assert len(meta.conflicts) == 0
        assert glossary.entities[0].surface_forms[0].translation_variants == []


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

    def test_off_path_does_not_accumulate_observations(self):
        """With compress disabled, no observations accumulate (Phase 1 path)."""
        entity = _make_entity(summary=None)
        merge_entity_summary(entity, "First.", "0000.001", compress_enabled=False)
        merge_entity_summary(entity, "Second.", "0000.002", compress_enabled=False)
        assert entity.summary_observations == []
        assert "First." in entity.summary
        assert "Second." in entity.summary


# ---------------------------------------------------------------------------
# TestSummaryCompress (Phase 2B)
# ---------------------------------------------------------------------------


def _compress_response(summary: str):
    """Build a GlossarySummaryCompressResponse."""
    return GlossarySummaryCompressResponse(summary=summary)


class TestSummaryObservationAccumulation:
    """merge_entity_summary in deferred (compress) mode."""

    def test_observation_accumulated_not_written_to_summary(self):
        entity = _make_entity(summary=None)
        merge_entity_summary(entity, "An observation.", "0003.005", compress_enabled=True)
        assert entity.summary is None
        assert len(entity.summary_observations) == 1
        assert entity.summary_observations[0].chunk_id == "0003.005"
        assert entity.summary_observations[0].text == "An observation."
        # latest_evidence_chunk is still set during build (cheap, no LLM).
        assert entity.latest_evidence_chunk == "0003.005"

    def test_multiple_observations_accumulate(self):
        entity = _make_entity(summary=None)
        merge_entity_summary(entity, "First.", "0001.001", compress_enabled=True)
        merge_entity_summary(entity, "Second.", "0002.001", compress_enabled=True)
        assert entity.summary is None
        assert [o.text for o in entity.summary_observations] == ["First.", "Second."]

    def test_null_update_ignored_in_compress_mode(self):
        entity = _make_entity(summary=None)
        merge_entity_summary(entity, None, "0001.001", compress_enabled=True)
        assert entity.summary_observations == []


class TestCompressEntitySummary:
    """compress_entity_summary (single entity)."""

    def _ctx(self):
        """Common args: config, template, langs."""
        config = AppConfig(source_epub="x.epub")
        template = (
            "{source_language} {target_language} {category} "
            "{canonical_name}\n{observations}\n{max_length}"
        )
        return config, template

    def test_bootstrap_single_observation_no_llm_call(self):
        config, template = self._ctx()
        entity = _make_entity(summary=None)
        entity.summary_observations = [SummaryObservation(chunk_id="0001.001", text="Only one.")]
        client = MagicMock()

        made_call = compress_entity_summary(
            entity,
            config,
            lambda: client,
            source_lang="Japanese",
            target_lang="English",
            template=template,
        )

        assert made_call is False
        client.complete_json.assert_not_called()
        assert entity.summary == "Only one."
        # Observations retained.
        assert len(entity.summary_observations) == 1

    def test_multiple_observations_one_llm_call(self):
        config, template = self._ctx()
        entity = _make_entity(summary=None)
        entity.summary_observations = [
            SummaryObservation(chunk_id="0002.001", text="Later fact."),
            SummaryObservation(chunk_id="0001.001", text="Earlier fact."),
        ]
        client = MagicMock()
        client.complete_json.return_value = _compress_response("Compressed summary.")

        made_call = compress_entity_summary(
            entity,
            config,
            lambda: client,
            source_lang="Japanese",
            target_lang="English",
            template=template,
        )

        assert made_call is True
        assert client.complete_json.call_count == 1
        assert entity.summary == "Compressed summary."
        # Observations retained and original order untouched on the entity.
        assert len(entity.summary_observations) == 2

    def test_observations_compressed_in_chunk_order(self):
        config, template = self._ctx()
        entity = _make_entity(summary=None)
        entity.summary_observations = [
            SummaryObservation(chunk_id="0005.001", text="Z-last."),
            SummaryObservation(chunk_id="0001.001", text="A-first."),
        ]
        client = MagicMock()
        client.complete_json.return_value = _compress_response("ok")

        compress_entity_summary(
            entity,
            config,
            lambda: client,
            source_lang="ja",
            target_lang="en",
            template=template,
        )

        # The rendered prompt must list A-first before Z-last.
        sent_messages = client.complete_json.call_args.args[0]
        prompt_text = sent_messages[0]["content"]
        assert prompt_text.index("A-first.") < prompt_text.index("Z-last.")

    def test_empty_summary_falls_back(self):
        config, template = self._ctx()
        entity = _make_entity(summary=None)
        entity.summary_observations = [
            SummaryObservation(chunk_id="0001.001", text="Fact one."),
            SummaryObservation(chunk_id="0002.001", text="Fact two."),
        ]
        client = MagicMock()
        client.complete_json.return_value = _compress_response("   ")

        compress_entity_summary(
            entity,
            config,
            lambda: client,
            source_lang="ja",
            target_lang="en",
            template=template,
        )

        # Fallback joins the observations rather than leaving summary empty.
        assert entity.summary is not None
        assert "Fact one." in entity.summary
        assert "Fact two." in entity.summary

    def test_summary_truncated_to_max_length(self):
        config, template = self._ctx()
        config.glossary.summary_max_length = 50
        entity = _make_entity(summary=None)
        entity.summary_observations = [
            SummaryObservation(chunk_id="0001.001", text="a"),
            SummaryObservation(chunk_id="0002.001", text="b"),
        ]
        client = MagicMock()
        client.complete_json.return_value = _compress_response("X " * 100)

        compress_entity_summary(
            entity,
            config,
            lambda: client,
            source_lang="ja",
            target_lang="en",
            template=template,
        )

        assert len(entity.summary) <= 53  # 50 + "..."


class TestCompressEntitySummariesPass:
    """compress_entity_summaries (full deferred pass)."""

    def test_o_entities_not_o_mentions(self, tmp_path):
        """LLM calls scale with entities (>=2 obs), not total observations."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True

        # Entity A: 3 observations -> 1 LLM call.
        ent_a = _make_entity(entity_id="character_000001", canonical_name="A", summary=None)
        ent_a.summary_observations = [
            SummaryObservation(chunk_id=f"0001.{i:03d}", text=f"obs {i}") for i in range(3)
        ]
        # Entity B: 1 observation -> bootstrap, no LLM call.
        ent_b = _make_entity(entity_id="character_000002", canonical_name="B", summary=None)
        ent_b.summary_observations = [SummaryObservation(chunk_id="0002.001", text="solo")]
        # Entity C: 2 observations -> 1 LLM call.
        ent_c = _make_entity(entity_id="character_000003", canonical_name="C", summary=None)
        ent_c.summary_observations = [
            SummaryObservation(chunk_id="0003.001", text="c1"),
            SummaryObservation(chunk_id="0003.002", text="c2"),
        ]
        glossary = Glossary(entities=[ent_a, ent_b, ent_c], book_id="test-book")
        bp = glossary_build_path(work)
        _save_glossary(work, glossary, bp)
        meta = _BuildMeta()

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _compress_response("compressed")
            mock_llm_cls.return_value = mock_client

            calls = compress_entity_summaries(work, config, glossary, meta, save_path=bp)

        # 6 total observations, but only 2 entities need an LLM call.
        assert calls == 2
        assert mock_client.complete_json.call_count == 2
        assert ent_b.summary == "solo"  # bootstrap
        assert meta.summary_compress_done is True

    def test_skips_user_entities(self, tmp_path):
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        user_ent = _make_entity(entity_id="character_000001", summary=None)
        user_ent.source = "user"
        user_ent.summary_observations = [
            SummaryObservation(chunk_id="0001.001", text="x"),
            SummaryObservation(chunk_id="0002.001", text="y"),
        ]
        glossary = Glossary(entities=[user_ent], book_id="test-book")
        bp = glossary_build_path(work)
        _save_glossary(work, glossary, bp)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_llm_cls.return_value = mock_client
            calls = compress_entity_summaries(work, config, glossary, _BuildMeta(), save_path=bp)

        assert calls == 0
        mock_client.complete_json.assert_not_called()
        assert user_ent.summary is None

    def test_resume_skips_already_compressed(self, tmp_path):
        """An entity that already has a summary is skipped on resume."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        done = _make_entity(entity_id="character_000001", summary="already done")
        done.summary_observations = [
            SummaryObservation(chunk_id="0001.001", text="a"),
            SummaryObservation(chunk_id="0002.001", text="b"),
        ]
        pending = _make_entity(entity_id="character_000002", summary=None)
        pending.summary_observations = [
            SummaryObservation(chunk_id="0003.001", text="c"),
            SummaryObservation(chunk_id="0003.002", text="d"),
        ]
        glossary = Glossary(entities=[done, pending], book_id="test-book")
        bp = glossary_build_path(work)
        _save_glossary(work, glossary, bp)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _compress_response("new summary")
            mock_llm_cls.return_value = mock_client
            calls = compress_entity_summaries(work, config, glossary, _BuildMeta(), save_path=bp)

        assert calls == 1  # only the pending entity
        assert done.summary == "already done"
        assert pending.summary == "new summary"

    def test_on_progress_fires_per_entity(self, tmp_path):
        """on_progress fires for every entity that needs compression, including
        bootstrap and resume-skipped ones."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        # LLM-compressed (2 obs), bootstrap (1 obs), already-done (resume-skip).
        ent_llm = _make_entity(entity_id="character_000001", summary=None)
        ent_llm.summary_observations = [
            SummaryObservation(chunk_id="0001.001", text="a"),
            SummaryObservation(chunk_id="0001.002", text="b"),
        ]
        ent_boot = _make_entity(entity_id="character_000002", summary=None)
        ent_boot.summary_observations = [SummaryObservation(chunk_id="0002.001", text="solo")]
        ent_done = _make_entity(entity_id="character_000003", summary="already")
        ent_done.summary_observations = [
            SummaryObservation(chunk_id="0003.001", text="c"),
            SummaryObservation(chunk_id="0003.002", text="d"),
        ]
        glossary = Glossary(entities=[ent_llm, ent_boot, ent_done], book_id="test-book")
        bp = glossary_build_path(work)
        _save_glossary(work, glossary, bp)

        seen: list[str] = []
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = _compress_response("x")
            mock_llm_cls.return_value = mock_client
            compress_entity_summaries(
                work,
                config,
                glossary,
                _BuildMeta(),
                save_path=bp,
                on_progress=lambda eid: seen.append(eid),
            )

        assert seen == [
            "character_000001",
            "character_000002",
            "character_000003",
        ]


def _dispatching_client(extraction_responses, compress_summary="Compressed."):
    """A MagicMock LLM client whose complete_json dispatches by response_model.

    Extraction calls (GlossaryExtractionResponse) are served from
    *extraction_responses* in order; compression calls
    (GlossarySummaryCompressResponse) return a fixed compressed summary.
    """
    extraction_iter = iter(extraction_responses)

    def _complete_json(messages, response_model=None, **kwargs):
        if response_model is GlossarySummaryCompressResponse:
            return GlossarySummaryCompressResponse(summary=compress_summary)
        return next(extraction_iter)

    client = MagicMock()
    client.complete_json.side_effect = _complete_json
    return client


class TestGlossaryBuildSummaryCompression:
    """glossary_build with summary_compress_enabled (Phase 2B integration)."""

    def test_build_accumulates_then_compresses(self, tmp_path):
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        config.glossary_phase.target_tokens_per_call = 600  # force 2 batches
        config.glossary_phase.min_batch_tokens = 100  # don't absorb the runt
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        # Two batches, each contributing a summary_update for the same entity.
        batch1 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "A boy from another world.",
                }
            ]
        )
        batch2 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "He can return after death.",
                }
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_llm_cls.return_value = _dispatching_client(
                [batch1, batch2], compress_summary="A boy from another world who can revive."
            )
            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        # Two observations accumulated and retained, with chunk provenance.
        assert len(subaru.summary_observations) == 2
        assert {o.text for o in subaru.summary_observations} == {
            "A boy from another world.",
            "He can return after death.",
        }
        # Compressed summary written.
        assert subaru.summary == "A boy from another world who can revive."
        # Meta flag set; persisted glossary matches.
        meta = _load_build_meta(work)
        assert meta.summary_compress_done is True
        reloaded = _load_glossary(work, glossary_build_path(work))
        rs = next(e for e in reloaded.entities if e.canonical_name == "Subaru")
        assert rs.summary == "A boy from another world who can revive."
        assert len(rs.summary_observations) == 2

    def test_disabled_uses_concatenation_path(self, tmp_path):
        """summary_compress_enabled=False reproduces Phase 1 behaviour."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        assert config.glossary.summary_compress_enabled is False
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        resp = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "A boy from another world.",
                }
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = resp
            mock_llm_cls.return_value = mock_client
            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        # Phase 1: summary written directly, no observations accumulated.
        assert subaru.summary == "A boy from another world."
        assert subaru.summary_observations == []
        meta = _load_build_meta(work)
        assert meta.summary_compress_done is False

    def test_force_summaries_recompresses_without_extraction(self, tmp_path):
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)
        config.glossary_phase.target_tokens_per_call = 600
        config.glossary_phase.min_batch_tokens = 100

        batch1 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "First.",
                }
            ]
        )
        batch2 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "Second.",
                }
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_llm_cls.return_value = _dispatching_client(
                [batch1, batch2], compress_summary="Initial compressed."
            )
            glossary_build(work, config, state, force=True)

        # Now recompress with a different compressor output, no extraction calls.
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()

            def _only_compress(messages, response_model=None, **kwargs):
                assert response_model is GlossarySummaryCompressResponse
                return GlossarySummaryCompressResponse(summary="Recompressed.")

            client.complete_json.side_effect = _only_compress
            mock_llm_cls.return_value = client
            glossary = glossary_build(work, config, state, force_summaries=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        assert subaru.summary == "Recompressed."
        # Observations preserved (they are the input).
        assert len(subaru.summary_observations) == 2

    def test_force_summaries_requires_enabled(self, tmp_path):
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)  # compression disabled
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        resp = _mock_extraction_response(
            mentions=[{"source": "ス", "translation": "S", "category": "character"}]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = resp
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        with pytest.raises(RuntimeError, match="summary_compress_enabled"):
            glossary_build(work, config, state, force_summaries=True)

    def test_force_summaries_no_build_output_errors(self, tmp_path):
        """Without a build output there is nothing to recompress."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)
        # No glossary_build.json on disk (extraction never ran).
        assert not glossary_build_path(work).exists()

        with pytest.raises(RuntimeError, match="has not produced output"):
            glossary_build(work, config, state, force_summaries=True)

    def test_force_summaries_restarts_interrupted_running_stage(self, tmp_path):
        """--force-summaries restarts compression even when the stage is
        ``running`` from an interrupted prior pass (the regression: it used to
        reject any non-``completed`` stage)."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)

        # Simulate the interrupted-mid-pass state: stage "running", done flag
        # already cleared by _build_then_gut_compression.
        state.stages["glossary_build"].status = "running"
        save_state(work, state)
        assert not is_stage_completed(state, "glossary_build")
        meta = _load_build_meta(work)
        assert meta.summary_compress_done is False

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()

            def _only_compress(messages, response_model=None, **kwargs):
                assert response_model is GlossarySummaryCompressResponse
                return GlossarySummaryCompressResponse(summary="Restarted.")

            client.complete_json.side_effect = _only_compress
            mock_llm_cls.return_value = client
            glossary = glossary_build(work, config, state, force_summaries=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        assert subaru.summary == "Restarted."
        # The pass finished, so the stage is honestly completed again.
        assert is_stage_completed(state, "glossary_build")
        assert _load_build_meta(work).summary_compress_done is True

    def test_force_summaries_renulls_already_done_summaries(self, tmp_path):
        """--force-summaries is a full restart: an entity that already has a
        summary is re-nulled and recomputed (not resume-skipped)."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)

        # Give the entity a pre-existing summary so we can prove it is redone.
        bp = glossary_build_path(work)
        glossary = _load_glossary(work, bp)
        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        subaru.summary = "Stale summary that must be replaced."
        _save_glossary(work, glossary, bp)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()
            client.complete_json.side_effect = lambda *a, **k: GlossarySummaryCompressResponse(
                summary="Freshly recomputed."
            )
            mock_llm_cls.return_value = client
            glossary = glossary_build(work, config, state, force_summaries=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        assert subaru.summary == "Freshly recomputed."
        # An LLM call was made for the entity (it was not resume-skipped).
        assert client.complete_json.call_count == 1

    def _build_then_gut_compression(self, work, config, state):
        """Build a glossary, then simulate an interrupted compression pass.

        Leaves the project in the inconsistent state an aborted
        ``--force-summaries`` produces: stage flag still ``completed``,
        ``summary_compress_done=False``, and the entity summary nulled while its
        observations remain.
        """
        config.glossary.summary_compress_enabled = True
        config.glossary_phase.target_tokens_per_call = 600
        config.glossary_phase.min_batch_tokens = 100
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        batch1 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "First.",
                }
            ]
        )
        batch2 = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "Second.",
                }
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_llm_cls.return_value = _dispatching_client(
                [batch1, batch2], compress_summary="Initial compressed."
            )
            glossary_build(work, config, state, force=True)

        # Simulate the aborted --force-summaries: gut the summary, clear the
        # done flag, but leave the coarse stage flag at "completed".
        bp = glossary_build_path(work)
        glossary = _load_glossary(work, bp)
        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        subaru.summary = None
        _save_glossary(work, glossary, bp)
        meta = _load_build_meta(work)
        meta.summary_compress_done = False
        _save_build_meta(work, meta)
        assert is_stage_completed(state, "glossary_build")  # stale "completed"

    def test_run_resumes_unfinished_compression_instead_of_skipping(self, tmp_path):
        """A plain build over an interrupted compression resumes it, not skip."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)

        # A plain run (no flags) must NOT early-return; it must recompress.
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()

            def _only_compress(messages, response_model=None, **kwargs):
                assert response_model is GlossarySummaryCompressResponse
                return GlossarySummaryCompressResponse(summary="Resumed compressed.")

            client.complete_json.side_effect = _only_compress
            mock_llm_cls.return_value = client
            glossary = glossary_build(work, config, state)
            assert client.complete_json.call_count == 1

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
        assert subaru.summary == "Resumed compressed."
        meta = _load_build_meta(work)
        assert meta.summary_compress_done is True
        assert is_stage_completed(state, "glossary_build")

    def test_run_invalidates_downstream_when_resuming_compression(self, tmp_path):
        """Resuming compression marks cluster/reconcile stale (they may have run
        over null summaries)."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)

        # Pretend downstream stages had run over the gutted glossary.
        mark_stage_completed(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_reconcile")

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()
            client.complete_json.return_value = GlossarySummaryCompressResponse(
                summary="Resumed."
            )
            mock_llm_cls.return_value = client
            glossary_build(work, config, state)

        assert not is_stage_completed(state, "glossary_cluster")
        assert not is_stage_completed(state, "glossary_reconcile")

    def test_completed_compression_still_skips(self, tmp_path):
        """A fully-complete build (compression done) early-returns without LLM."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        config.glossary.summary_compress_enabled = True
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        resp = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "category": "character",
                    "summary_update": "A boy.",
                }
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()
            client.complete_json.return_value = resp
            mock_llm_cls.return_value = client
            glossary_build(work, config, state, force=True)

        assert _load_build_meta(work).summary_compress_done is True

        # Second plain run: no LLM client should ever be constructed.
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary_build(work, config, state)
            mock_llm_cls.assert_not_called()

    def test_disabled_completed_build_skips_without_resume(self, tmp_path):
        """Compression disabled: a completed build always skips immediately."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)  # compression disabled
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        resp = _mock_extraction_response(
            mentions=[{"source": "ス", "translation": "S", "category": "character"}]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()
            client.complete_json.return_value = resp
            mock_llm_cls.return_value = client
            glossary_build(work, config, state, force=True)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary_build(work, config, state)
            mock_llm_cls.assert_not_called()

    def test_force_summaries_marks_stage_completed(self, tmp_path):
        """--force-summaries leaves the stage honestly completed afterwards."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            client = MagicMock()
            client.complete_json.return_value = GlossarySummaryCompressResponse(
                summary="Forced."
            )
            mock_llm_cls.return_value = client
            glossary_build(work, config, state, force_summaries=True)

        assert is_stage_completed(state, "glossary_build")
        assert _load_build_meta(work).summary_compress_done is True

    def test_force_summaries_interrupt_leaves_resumable_state(self, tmp_path):
        """If --force-summaries fails mid-pass, the stage is not 'completed' so a
        later run resumes it."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        self._build_then_gut_compression(work, config, state)
        # Re-give the entity a summary so reopen+gut path runs cleanly; the
        # helper already nulled it, which is fine — force-summaries nulls anyway.

        with patch(
            "dao_bridge.glossary.compress_entity_summaries",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                glossary_build(work, config, state, force_summaries=True)

        # Stage must NOT be 'completed' (it was reopened to 'running' before the
        # failing pass) and compression is still flagged unfinished.
        assert not is_stage_completed(state, "glossary_build")
        assert _load_build_meta(work).summary_compress_done is False


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
                    "translation": "Natsuki Subaru",
                    "category": "character",
                    "aliases": ["スバル"],
                    "nicknames": {},
                    "speech_style": "Casual modern speech.",
                    "notes": "Protagonist.",
                    "summary_update": "A young man transported to another world.",
                },
                {
                    "source": "エミリア",
                    "translation": "Emilia",
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
        subaru = next(e for e in glossary.entities if e.canonical_name == "Natsuki Subaru")
        assert subaru.entity_id == "character_000001"
        assert subaru.source == "extracted"
        assert "スバル" in subaru.aliases
        assert subaru.speech_style == "Casual modern speech."
        assert subaru.summary == "A young man transported to another world."
        assert subaru.first_seen_chunk is not None
        assert len(subaru.surface_forms) == 1
        assert subaru.surface_forms[0].source == "ナツキ・スバル"
        assert subaru.surface_forms[0].translation == "Natsuki Subaru"

        emilia = next(e for e in glossary.entities if e.canonical_name == "Emilia")
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
                    "translation": "Subaru",
                    "category": "character",
                    "aliases": ["バルス"],
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
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

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
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
                    "translation": "Subaru",
                    "category": "character",
                    "speech_style": "Casual speech.",
                },
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
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

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
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
                    "translation": "Priscilla",
                    "category": "character",
                },
            ],
            corrections=[
                {
                    "existing_translation": "Priscilla",
                    "source_term": "プリシラ・バーリエル",
                    "corrected_translation": "Priscilla Barielle",
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
        priscilla = next(e for e in glossary.entities if e.canonical_name == "Priscilla")
        assert priscilla.canonical_name == "Priscilla"

        # But it should be in the build meta conflicts.
        meta = _load_build_meta(work)
        assert len(meta.corrections) == 1
        assert meta.corrections[0]["corrected_translation"] == "Priscilla Barielle"
        assert len(meta.conflicts) >= 1

    def test_correction_prefers_unique_source_form_when_translation_is_ambiguous(self, tmp_path):
        """Correction routing falls back to a unique source-form match."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=2)
        _write_chunks(work, n_spines=1, chunks_per_spine=2, tokens_per_chunk=500)

        # Pre-seed two entities sharing the same canonical_name.
        _save_glossary(
            work,
            Glossary(
                entities=[
                    GlossaryEntity(
                        entity_id="character_000001",
                        category="character",
                        canonical_name="Priscilla",
                        surface_forms=[SurfaceForm(source="プリシラ", translation="Priscilla")],
                        source="extracted",
                    ),
                    GlossaryEntity(
                        entity_id="character_000002",
                        category="character",
                        canonical_name="Priscilla",
                        surface_forms=[
                            SurfaceForm(
                                source="プリシラ・バーリエル",
                                translation="Priscilla Barielle",
                            )
                        ],
                        source="extracted",
                    ),
                ]
            ),
            glossary_build_path(work),
        )

        mock_response = _mock_extraction_response(
            mentions=[],
            corrections=[
                {
                    "existing_translation": "Priscilla",
                    "source_term": "プリシラ・バーリエル",
                    "corrected_translation": "Priscilla Barielle",
                    "reason": "Full name appears.",
                }
            ],
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=False)

        # canonical_name is ambiguous (two entities share "Priscilla"),
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

        # Pre-seed a user entity in build output.
        pre_glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Subaru (custom)",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru (custom)")],
                    source="user",
                    aliases=["original_alias"],
                )
            ]
        )
        _save_glossary(work, pre_glossary, glossary_build_path(work))

        mock_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Natsuki Subaru",
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
        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru (custom)")
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
            mentions=[{"source": "スバル", "translation": "Subaru", "category": "character"}]
        )
        batch2_response = _mock_extraction_response(
            mentions=[{"source": "スバル", "translation": "Subaru", "category": "character"}]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = [batch1_response, batch2_response]
            mock_llm_cls.return_value = mock_client

            glossary = glossary_build(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
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

        # Batch 1: mention スバル. Batch 2: mention スバル again with same reading+translation
        # (so it attaches) but also a mention with different source that shares
        # reading+translation (attaches as new surface form).
        batch1_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "スバル",
                    "translation": "Subaru",
                    "reading": "すばる",
                    "category": "character",
                }
            ]
        )
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "ナツキ・スバル",
                    "translation": "Subaru",
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
# TestBuildFileFlow
# ---------------------------------------------------------------------------


class TestBuildFileFlow:
    """Tests for the build stage's per-stage output file convention."""

    def test_build_writes_to_glossary_build_json(self, tmp_path):
        """Build output goes to glossary_build.json, not glossary.json."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        mock_response = _mock_extraction_response(
            mentions=[
                {"source": "スバル", "translation": "Subaru", "category": "character"},
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        # Build output exists at glossary_build.json.
        assert glossary_build_path(work).exists()
        build_glossary = _load_glossary(work, glossary_build_path(work))
        assert len(build_glossary.entities) == 1
        assert build_glossary.entities[0].canonical_name == "Subaru"

        # glossary.json should NOT exist (reconcile hasn't run).
        assert not glossary_path(work).exists()

    def test_build_force_deletes_downstream(self, tmp_path):
        """--force on build deletes glossary_cluster.json and glossary.json."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        # Create fake downstream files.
        glossary_cluster_path(work).write_text("{}", encoding="utf-8")
        glossary_path(work).write_text("{}", encoding="utf-8")

        mock_response = _mock_extraction_response(mentions=[])

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        # Downstream files should have been deleted.
        assert not glossary_cluster_path(work).exists()
        assert not glossary_path(work).exists()


# ---------------------------------------------------------------------------
# TestBuildResume
# ---------------------------------------------------------------------------


class TestBuildResume:
    """Tests for build-stage resumability."""

    def test_resume_after_partial(self, tmp_path):
        """Re-running from a partial glossary_build.json picks up at next batch."""
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
                    "translation": "Emilia",
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

        # Verify first batch was saved to build output.
        glossary = _load_glossary(work, glossary_build_path(work))
        assert len(glossary.entities) == 1
        assert glossary.entities[0].canonical_name == "Emilia"

        # Second run: resume from batch 2.
        batch2_response = _mock_extraction_response(
            mentions=[
                {
                    "source": "レム",
                    "translation": "Rem",
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
        assert any(e.canonical_name == "Emilia" for e in glossary.entities)
        assert any(e.canonical_name == "Rem" for e in glossary.entities)


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
                    "translation": "Subaru",
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
                    "translation": "Emilia",
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
                    "translation": "Rem",
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
# Issue 1+2: Build force/targeted downstream state invalidation
# ---------------------------------------------------------------------------


class TestBuildDownstreamInvalidation:
    """Build --force and --spine/--batch should reset downstream stage state."""

    def _setup_full_pipeline(self, tmp_path):
        """Build -> cluster -> reconcile all completed."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        _make_manifest(work, n_spines=1, chunks_per_spine=1)
        _write_chunks(work, n_spines=1, chunks_per_spine=1, tokens_per_chunk=500)

        # Complete build.
        mock_response = _mock_extraction_response(
            mentions=[
                {"source": "スバル", "translation": "Subaru", "category": "character"},
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        # Mark cluster and reconcile as completed with fake output files.
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")
        mark_stage_started(work, state, "glossary_reconcile")
        mark_stage_completed(work, state, "glossary_reconcile")
        glossary_cluster_path(work).write_text('{"entities":[], "version": 2}', encoding="utf-8")
        glossary_path(work).write_text('{"entities":[], "version": 2}', encoding="utf-8")

        return work, config, state

    def test_build_force_resets_downstream_stage_state(self, tmp_path):
        """--force on build resets cluster and reconcile to pending."""
        work, config, state = self._setup_full_pipeline(tmp_path)

        assert state.stages["glossary_cluster"].status == "completed"
        assert state.stages["glossary_reconcile"].status == "completed"

        mock_response = _mock_extraction_response(mentions=[])
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        # Both downstream stages should be reset.
        assert state.stages["glossary_cluster"].status == "pending"
        assert state.stages["glossary_reconcile"].status == "pending"
        # Downstream output files should be deleted.
        assert not glossary_cluster_path(work).exists()
        assert not glossary_path(work).exists()

    def test_targeted_spine_resets_downstream_stage_state(self, tmp_path):
        """--spine on build resets cluster and reconcile to pending."""
        work, config, state = self._setup_full_pipeline(tmp_path)

        assert state.stages["glossary_cluster"].status == "completed"
        assert state.stages["glossary_reconcile"].status == "completed"

        mock_response = _mock_extraction_response(
            mentions=[
                {"source": "エミリア", "translation": "Emilia", "category": "character"},
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, target_spine=0)

        # Both downstream stages should be reset.
        assert state.stages["glossary_cluster"].status == "pending"
        assert state.stages["glossary_reconcile"].status == "pending"

    def test_targeted_batch_resets_downstream_stage_state(self, tmp_path):
        """--batch on build resets cluster and reconcile to pending."""
        work, config, state = self._setup_full_pipeline(tmp_path)

        assert state.stages["glossary_cluster"].status == "completed"
        assert state.stages["glossary_reconcile"].status == "completed"

        mock_response = _mock_extraction_response(
            mentions=[
                {"source": "レム", "translation": "Rem", "category": "character"},
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, target_batch="0000.b1")

        # Both downstream stages should be reset.
        assert state.stages["glossary_cluster"].status == "pending"
        assert state.stages["glossary_reconcile"].status == "pending"

    def test_build_force_prevents_stale_cluster_skip(self, tmp_path):
        """After build --force, cluster does not skip on is_stage_completed."""
        work, config, state = self._setup_full_pipeline(tmp_path)

        # Cluster was "completed" before the build --force.
        assert state.stages["glossary_cluster"].status == "completed"

        mock_response = _mock_extraction_response(mentions=[])
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_build(work, config, state, force=True)

        # Now cluster should NOT skip -- its state has been reset.
        from dao_bridge.state import is_stage_completed

        assert not is_stage_completed(state, "glossary_cluster")
        assert not is_stage_completed(state, "glossary_reconcile")


# ---------------------------------------------------------------------------
# Surface-form Reconcile
# ---------------------------------------------------------------------------


class TestTranslationVariantsReconcile:
    """Surface-form translation_variants are reconciled directly from the glossary."""

    def test_surface_form_conflict_resolved_from_translation_variants(self, tmp_path):
        """Reconcile updates the surface form and clears translation_variants."""
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
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル",
                            translation="Abel",
                            translation_variants=["Aberu", "Abell"],
                        )
                    ],
                    source="extracted",
                )
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Abell",
            reasoning="Best rendering for this exact form.",
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entity = next(e for e in glossary.entities if e.entity_id == "character_000001")
        sf = entity.surface_forms[0]
        assert sf.translation == "Abell"
        assert sf.translation_variants == []
        assert entity.canonical_name == "Abel"

    def test_multi_surface_forms_resolved_independently(self, tmp_path):
        """Each surface form with variants becomes its own reconcile item."""
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
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル", translation="Abel", translation_variants=["Aberu"]
                        ),
                        SurfaceForm(
                            source="ヴィンセント",
                            translation="Vincent",
                            translation_variants=["Vincento"],
                        ),
                    ],
                    source="extracted",
                )
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        responses = [
            GlossaryReconcileResponse(
                chosen_translation="Aberu",
                reasoning="Preferred for this form.",
            ),
            GlossaryReconcileResponse(
                chosen_translation="Vincento",
                reasoning="Preferred for this form.",
            ),
        ]

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = responses
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entity = next(e for e in glossary.entities if e.entity_id == "character_000001")
        by_source = {sf.source: sf for sf in entity.surface_forms}
        assert by_source["アベル"].translation == "Aberu"
        assert by_source["ヴィンセント"].translation == "Vincento"
        assert by_source["アベル"].translation_variants == []
        assert by_source["ヴィンセント"].translation_variants == []
        assert mock_client.complete_json.call_count == 2

    def test_surface_form_change_and_variant_clear_persisted_together(self, tmp_path):
        """Disk glossary persists chosen translation and cleared variants in one save."""
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
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル", translation="Abel", translation_variants=["Aberu"]
                        )
                    ],
                    source="extracted",
                )
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Aberu",
            reasoning="Best rendering for this form.",
        )

        saved_states: list[tuple[str, list[str]]] = []

        real_save_glossary = _save_glossary

        def checking_save_glossary(work_dir, glossary, path=None):
            real_save_glossary(work_dir, glossary, path)
            disk_glossary = _load_glossary(work_dir, path)
            entity = next(e for e in disk_glossary.entities if e.entity_id == "character_000001")
            sf = entity.surface_forms[0]
            saved_states.append((sf.translation, list(sf.translation_variants)))

        with (
            patch("dao_bridge.glossary.LLMClient") as mock_llm_cls,
            patch(
                "dao_bridge.glossary._save_glossary",
                side_effect=checking_save_glossary,
            ),
        ):
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client
            glossary = glossary_reconcile(work, config, state, force=True)

        entity = next(e for e in glossary.entities if e.entity_id == "character_000001")
        assert entity.surface_forms[0].translation == "Aberu"
        assert entity.surface_forms[0].translation_variants == []
        assert ("Aberu", []) in saved_states

    def test_surface_form_conflicts_do_not_create_item_state(self, tmp_path):
        """Surface-form conflicts are resumed from glossary data, not state items."""
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
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル", translation="Abel", translation_variants=["Aberu"]
                        )
                    ],
                    source="extracted",
                )
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Aberu",
            reasoning="Best rendering for this form.",
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client
            glossary_reconcile(work, config, state, force=True)

        assert not any(
            key.startswith("glossary_reconcile:glossary_reconcile.sf.") for key in state.items
        )


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

        # Create glossary with an entity — written to cluster output
        # (reconcile's input).
        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="place_000001",
                    category="place",
                    canonical_name="Lugnica",
                    surface_forms=[
                        SurfaceForm(source="ルグニカ", reading="るぐにか", translation="Lugnica")
                    ],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))

        # Create build meta with a conflict.
        meta = _BuildMeta(
            conflicts=[
                {
                    "entity_id": "place_000001",
                    "source_form": "ルグニカ",
                    "reading": "るぐにか",
                    "current_translation": "Lugnica",
                    "alternatives": [
                        {
                            "translation": "Lugunica",
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
        """Reconcile applies the LLM's chosen translation form."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Lugunica",
            reasoning="More common romanization.",
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_reconcile
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        entity = next(e for e in glossary.entities if e.entity_id == "place_000001")
        assert entity.canonical_name == "Lugunica"

    def test_term_change_persisted_before_item_completion(self, tmp_path):
        """Disk glossary is updated before a term item is marked complete."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Lugunica",
            reasoning="More common romanization.",
        )

        def checking_mark_item_completed(work_dir, pipeline_state, stage, item_id):
            if item_id.startswith("glossary_reconcile.term."):
                disk_glossary = _load_glossary(work_dir)
                entity = next(e for e in disk_glossary.entities if e.entity_id == "place_000001")
                assert entity.canonical_name == "Lugunica"
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
        assert entity.canonical_name == "Lugunica"

    def test_report_generated(self, tmp_path):
        """Reconcile report markdown is written."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_reconcile = GlossaryReconcileResponse(
            chosen_translation="Lugunica",
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
                    canonical_name="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru")],
                    source="extracted",
                    speech_style="Casual speech.\nUses modern slang.\nFrequent sarcasm.",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        mock_speech = GlossarySpeechMergeResponse(
            consolidated_speech_style="Speaks casually with modern slang and frequent sarcasm."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_speech
            mock_llm_cls.return_value = mock_client

            glossary = glossary_reconcile(work, config, state, force=True)

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
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
                    canonical_name="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", translation="Subaru")],
                    source="extracted",
                    speech_style="Casual speech.\nUses modern slang.",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        mock_speech = GlossarySpeechMergeResponse(
            consolidated_speech_style="Speaks casually with modern slang."
        )

        def checking_mark_item_completed(work_dir, pipeline_state, stage, item_id):
            if item_id.startswith("glossary_reconcile.speech."):
                disk_glossary = _load_glossary(work_dir)
                entity = next(e for e in disk_glossary.entities if e.canonical_name == "Subaru")
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

        subaru = next(e for e in glossary.entities if e.canonical_name == "Subaru")
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
                    canonical_name="X",
                    surface_forms=[SurfaceForm(source="X", translation="X")],
                    source="extracted",
                    speech_style="Single observation only.",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
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
                    canonical_name="Holy Sword",
                    surface_forms=[SurfaceForm(source="聖剣", translation="Holy Sword")],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))

        meta = _BuildMeta(
            conflicts=[
                {
                    "entity_id": "item_000001",
                    "source_form": "聖剣",
                    "reading": None,
                    "current_translation": "Holy Sword",
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
                    canonical_name="X",
                    surface_forms=[SurfaceForm(source="X", translation="X")],
                    source="user",
                ),
            ]
        )
        _save_glossary(work, bad_glossary, glossary_build_path(work))

        with pytest.raises(ValueError, match="weapon"):
            glossary_build(work, config, state, force=False)

    def test_reconcile_reads_from_cluster_output(self, tmp_path):
        """Reconcile loads from glossary_cluster.json."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        # glossary_cluster.json should exist (written by setup), glossary.json should not.
        assert glossary_cluster_path(work).exists()
        assert not glossary_path(work).exists()

        mock_response = GlossaryReconcileResponse(
            chosen_translation="Lugunica", reasoning="Fan convention."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary = glossary_reconcile(work, config, state, force=True)

        # Should have loaded the entity from cluster output.
        assert len(glossary.entities) == 1
        assert glossary.entities[0].canonical_name == "Lugunica"

    def test_reconcile_writes_to_glossary_json(self, tmp_path):
        """Reconcile output goes to glossary.json (final output)."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        mock_response = GlossaryReconcileResponse(
            chosen_translation="Lugunica", reasoning="Fan convention."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_reconcile(work, config, state, force=True)

        # glossary.json should exist as the final output.
        assert glossary_path(work).exists()
        final = _load_glossary(work)
        assert final.entities[0].canonical_name == "Lugunica"

    def test_reconcile_force_rereads_cluster_output(self, tmp_path):
        """--force on reconcile deletes glossary.json and re-reads from glossary_cluster.json."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        # Record original entity data from cluster output.
        original = _load_glossary(work, glossary_cluster_path(work))
        original_translation = original.entities[0].canonical_name
        assert original_translation == "Lugnica"

        mock_response = GlossaryReconcileResponse(
            chosen_translation="Lugunica", reasoning="Fan convention."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary = glossary_reconcile(work, config, state, force=True)

        # Reconcile changed the entity.
        assert glossary.entities[0].canonical_name == "Lugunica"

        # Force re-run: should delete glossary.json and re-read from
        # glossary_cluster.json (which still has "Lugnica").
        mock_response2 = GlossaryReconcileResponse(
            chosen_translation="Lugnica Kingdom", reasoning="Different choice."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response2
            mock_llm_cls.return_value = mock_client
            glossary = glossary_reconcile(work, config, state, force=True)

        # Should have the new LLM choice, applied from pristine cluster output.
        assert glossary.entities[0].canonical_name == "Lugnica Kingdom"

    def test_cluster_output_not_mutated_by_reconcile(self, tmp_path):
        """glossary_cluster.json is byte-identical before and after reconcile."""
        work, config, state = self._setup_with_conflicts(tmp_path)

        # Record cluster output bytes.
        cluster_bytes_before = glossary_cluster_path(work).read_bytes()

        mock_response = GlossaryReconcileResponse(
            chosen_translation="Lugunica", reasoning="Fan convention."
        )

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.return_value = mock_response
            mock_llm_cls.return_value = mock_client
            glossary_reconcile(work, config, state, force=True)

        # Cluster output must not have been mutated.
        cluster_bytes_after = glossary_cluster_path(work).read_bytes()
        assert cluster_bytes_before == cluster_bytes_after


# ---------------------------------------------------------------------------
# TestReconcileProgress
# ---------------------------------------------------------------------------


class TestReconcileProgress:
    """Structured progress is emitted with correct phases, totals, and labels."""

    def test_progress_phases_and_totals(self, tmp_path):
        """All three phases emit correct phase, total, completed, and item_label."""
        from dao_bridge.glossary import GlossaryReconcileProgress

        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        # Entity with a surface-form conflict, an entity-level conflict,
        # and a speech-style consolidation item.
        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル",
                            translation="Abel",
                            translation_variants=["Aberu"],
                        ),
                    ],
                    speech_style="polite\nrude",
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))

        # Entity-level conflict (from build).
        from dao_bridge.glossary import _ConflictRecord

        meta = _BuildMeta(
            conflicts=[
                _ConflictRecord(
                    entity_id="character_000001",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[
                        {
                            "translation": "Abell",
                            "context_snippet": "batch 1",
                            "batch_id": "0000.b1",
                        }
                    ],
                ),
            ]
        )
        _save_build_meta(work, meta)

        # Mock LLM: surface-form -> "Aberu", entity -> "Abell", speech -> consolidated.
        responses = [
            GlossaryReconcileResponse(chosen_translation="Aberu", reasoning="surface form"),
            GlossaryReconcileResponse(chosen_translation="Abell", reasoning="entity"),
            GlossarySpeechMergeResponse(consolidated_speech_style="polite but sometimes rude"),
        ]

        progress_calls: list[GlossaryReconcileProgress] = []

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = responses
            mock_llm_cls.return_value = mock_client

            glossary_reconcile(
                work,
                config,
                state,
                force=True,
                on_progress=lambda p: progress_calls.append(p),
            )

        assert len(progress_calls) == 3

        # Phase 1: surface_form
        p0 = progress_calls[0]
        assert p0.phase == "surface_form"
        assert p0.phase_label == "Surface-form conflicts"
        assert p0.completed == 1
        assert p0.total == 1
        assert "アベル" in p0.item_label
        assert "character_000001" in p0.item_label

        # Phase 2: entity_conflict
        p1 = progress_calls[1]
        assert p1.phase == "entity_conflict"
        assert p1.phase_label == "Entity conflicts"
        assert p1.completed == 1
        assert p1.total == 1
        assert "character_000001" in p1.item_label

        # Phase 3: speech_style
        p2 = progress_calls[2]
        assert p2.phase == "speech_style"
        assert p2.phase_label == "Speech styles"
        assert p2.completed == 1
        assert p2.total == 1
        assert p2.item_label == "Abell"  # canonical_name after entity reconcile

    def test_empty_phases_emit_no_progress(self, tmp_path):
        """Phases with zero items are skipped entirely."""
        from dao_bridge.glossary import GlossaryReconcileProgress

        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)
        mark_stage_started(work, state, "glossary_build")
        mark_stage_completed(work, state, "glossary_build")
        mark_stage_started(work, state, "glossary_cluster")
        mark_stage_completed(work, state, "glossary_cluster")

        # Entity with no conflicts, no variants, no speech delimiter.
        glossary = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(source="アベル", translation="Abel"),
                    ],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        progress_calls: list[GlossaryReconcileProgress] = []

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary_reconcile(
                work,
                config,
                state,
                force=True,
                on_progress=lambda p: progress_calls.append(p),
            )

        assert len(progress_calls) == 0

    def test_surface_form_only_progress(self, tmp_path):
        """When only surface-form conflicts exist, only that phase emits progress."""
        from dao_bridge.glossary import GlossaryReconcileProgress

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
                    canonical_name="Abel",
                    surface_forms=[
                        SurfaceForm(
                            source="アベル", translation="Abel", translation_variants=["Aberu"]
                        ),
                        SurfaceForm(
                            source="アベルちゃん",
                            translation="Abel-chan",
                            translation_variants=["Aberu-chan"],
                        ),
                    ],
                    source="extracted",
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_cluster_path(work))
        _save_build_meta(work, _BuildMeta())

        responses = [
            GlossaryReconcileResponse(chosen_translation="Aberu", reasoning="r1"),
            GlossaryReconcileResponse(chosen_translation="Aberu-chan", reasoning="r2"),
        ]

        progress_calls: list[GlossaryReconcileProgress] = []

        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.complete_json.side_effect = responses
            mock_llm_cls.return_value = mock_client

            glossary_reconcile(
                work,
                config,
                state,
                force=True,
                on_progress=lambda p: progress_calls.append(p),
            )

        assert len(progress_calls) == 2
        assert all(p.phase == "surface_form" for p in progress_calls)
        assert progress_calls[0].completed == 1
        assert progress_calls[0].total == 2
        assert progress_calls[1].completed == 2
        assert progress_calls[1].total == 2


# ---------------------------------------------------------------------------
# TestGlossaryExport
# ---------------------------------------------------------------------------


class TestGlossaryExport:
    """Tests for the glossary_export function."""

    def test_grouped_by_category_sorted_alphabetically(self, tmp_path):
        """Entities are grouped by category and sorted by canonical_name."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)

        glossary = Glossary(
            entities=[
                _make_entity(
                    entity_id="character_000002",
                    canonical_name="Zorro",
                    surface_forms=[{"source": "B-char", "translation": "Zorro"}],
                ),
                _make_entity(
                    entity_id="character_000001",
                    canonical_name="Alice",
                    surface_forms=[{"source": "A-char", "translation": "Alice"}],
                ),
                _make_entity(
                    entity_id="place_000001",
                    category="place",
                    canonical_name="Kingdom",
                    surface_forms=[{"source": "Place1", "translation": "Kingdom"}],
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
                    canonical_name="Subaru",
                    surface_forms=[
                        SurfaceForm(source="スバル", reading="すばる", translation="Subaru")
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
                    canonical_name="Xterm",
                    surface_forms=[{"source": "X", "translation": "Xterm"}],
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
                    surface_forms=[{"source": "X", "translation": "X"}],
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
                    surface_forms=[{"source": "X", "translation": "X"}],
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
                    surface_forms=[{"source": "X", "translation": "X"}],
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
                    canonical_name="Test",
                    surface_forms=[{"source": "テスト", "translation": "Test"}],
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
                        canonical_name="Hidden Subaru",
                        surface_forms=[
                            SurfaceForm(source="ナツキ・スバル", translation="Hidden Subaru")
                        ],
                        source="extracted",
                    )
                ]
            ),
            glossary_build_path(work),
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
                    "translation": "Subaru",
                    "category": "character",
                    "aliases": ["ナツキ・スバル"],
                    "speech_style": "Casual speech.",
                    "summary_update": "A young man from another world.",
                },
                {
                    "source": "エミリア",
                    "translation": "Emilia",
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
        assert any(e.canonical_name == "Subaru" for e in glossary.entities)
        assert any(e.canonical_name == "Emilia" for e in glossary.entities)

        # Verify glossary_build.json exists on disk (build output).
        bp = glossary_build_path(work)
        assert bp.exists()
        disk_glossary = Glossary(**json.loads(bp.read_text(encoding="utf-8")))
        assert len(disk_glossary.entities) >= 2

        # glossary.json should NOT exist yet (reconcile hasn't run).
        assert not glossary_path(work).exists()

        # Record build output bytes for immutability check.
        build_bytes = bp.read_bytes()

        # --- glossary-cluster (no duplicates, so no-op) ---
        from dao_bridge.glossary import glossary_cluster

        glossary = glossary_cluster(work, config, state, force=True)

        assert state.stages.get("glossary_cluster") is not None
        assert state.stages["glossary_cluster"].status == "completed"

        cluster_report_path = work / "glossary_cluster_report.md"
        assert cluster_report_path.exists()

        # glossary_cluster.json should exist.
        assert glossary_cluster_path(work).exists()
        # Build output must not have been mutated by clustering.
        assert bp.read_bytes() == build_bytes

        # --- glossary-reconcile (no conflicts, so no-op) ---
        with patch("dao_bridge.glossary.LLMClient") as mock_llm_cls:
            glossary = glossary_reconcile(work, config, state, force=True)

        assert state.stages.get("glossary_build") is not None
        assert state.stages["glossary_build"].status == "completed"
        assert state.stages.get("glossary_reconcile") is not None
        assert state.stages["glossary_reconcile"].status == "completed"

        report_path = work / "glossary_reconcile_report.md"
        assert report_path.exists()

        # glossary.json should now exist (reconcile output).
        assert glossary_path(work).exists()
        # All three stage output files should coexist.
        assert glossary_build_path(work).exists()
        assert glossary_cluster_path(work).exists()


class TestGlossaryBuildProgressWrapper:
    """The CLI progress wrapper follows both build sub-phases on one task."""

    def test_phase_switch_resets_without_keyerror(self, tmp_path):
        """Switching extract -> compress must reset the task while re-supplying
        every custom field (item, of_batch); a missing field raises
        ``KeyError: 'item'`` on the next render."""
        from dao_bridge.cli import _run_glossary_build_with_progress
        from dao_bridge.glossary import GlossaryBuildProgress

        captured: dict[str, object] = {}

        def fake_build(work, config, state, *, on_progress=None, **kwargs):
            # Extraction batches.
            on_progress(
                GlossaryBuildProgress(
                    item_id="0001.b1", spine_batch_count=2, items_total=2
                )
            )
            on_progress(
                GlossaryBuildProgress(
                    item_id="0001.b2", spine_batch_count=2, items_total=2
                )
            )
            # Deferred compression pass (phase switch).
            on_progress(
                GlossaryBuildProgress(
                    item_id="character_000001",
                    spine_batch_count=0,
                    items_total=3,
                    phase="compress",
                    phase_label="Compressing summaries",
                )
            )
            on_progress(
                GlossaryBuildProgress(
                    item_id="character_000002",
                    spine_batch_count=0,
                    items_total=3,
                    phase="compress",
                    phase_label="Compressing summaries",
                )
            )
            captured["glossary"] = MagicMock(entities=[])
            return captured["glossary"]

        with patch("dao_bridge.glossary.glossary_build", fake_build):
            result = _run_glossary_build_with_progress(
                work=tmp_path,
                config=MagicMock(),
                state=MagicMock(),
            )

        # No KeyError raised during rendering means the reset re-supplied fields.
        assert result is captured["glossary"]
