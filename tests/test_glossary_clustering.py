"""Tests for dao_bridge.glossary_clustering — candidate generation, merging, and integration.

Covers:
- Individual candidate generation heuristics
- Entity merge behaviour
- ID remapping across batched decisions
- Clustering report generation
- Full glossary_cluster integration (mocked LLM)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.config import AppConfig, GlossaryClusterConfig
from dao_bridge.glossary import (
    _cluster_meta_path,
    _load_cluster_meta,
    _load_glossary,
    _save_glossary,
    glossary_cluster,
)
from dao_bridge.glossary_clustering import (
    HEURISTIC_ALIAS_OVERLAP,
    HEURISTIC_EMBEDDING,
    HEURISTIC_JW,
    HEURISTIC_SHARED_READING,
    HEURISTIC_SOURCE_SUBSTRING,
    HEURISTIC_TRANSLATION_CONTAINMENT,
    ClusterConfidence,
    Evidence,
    _alias_overlap_candidates,
    _jw_similarity_candidates,
    _shared_reading_candidates,
    _source_substring_candidates,
    _translation_containment_candidates,
    generate_cluster_candidates,
    merge_entities,
    pick_canonical_for_auto_merge,
    remap_entity_id,
    render_entity_for_cluster_prompt,
    score_candidate_confidence,
    write_cluster_report,
)
from dao_bridge.schemas import (
    Glossary,
    GlossaryClusterDecision,
    GlossaryClusterResponse,
    GlossaryEntity,
    SurfaceForm,
)
from dao_bridge.state import (
    PipelineState,
    load_state,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
)
from dao_bridge.workdir import (
    atomic_write,
    glossary_build_path,
    glossary_cluster_path,
    glossary_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(work_dir: Path, **overrides) -> AppConfig:
    """Create a minimal AppConfig pointing at work_dir.

    Auto-merge is disabled by default so the LLM-path integration tests below
    continue to route every candidate through the (mocked) LLM. Tests that
    exercise the auto-merge path opt in with ``auto_merge_enabled=True`` via a
    nested ``glossary`` override.
    """
    defaults = {
        "source_epub": str(work_dir / "test.epub"),
        "work_dir": str(work_dir),
        "glossary": {"cluster": {"auto_merge_enabled": False}},
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _setup_work_dir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    (work / "chunks").mkdir()
    return work


def _mark_prior_stages_complete(work_dir: Path, state: PipelineState) -> None:
    """Mark all stages before glossary_cluster as completed."""
    for stage in ("extract", "clean", "classify", "chunk", "glossary_build"):
        mark_stage_started(work_dir, state, stage)
        mark_stage_completed(work_dir, state, stage)


def _make_entity(
    entity_id: str = "character_000001",
    category: str = "character",
    canonical_name: str = "Subaru",
    surface_forms: list[dict] | None = None,
    **kwargs,
) -> GlossaryEntity:
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


# ---------------------------------------------------------------------------
# Candidate generation — JP substring
# ---------------------------------------------------------------------------


class TestJpSubstringCandidates:
    """Japanese surface-form substring containment."""

    def test_substring_match(self):
        """アベル is substring of アベルちゃん."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        pairs = _source_substring_candidates(glossary)
        assert ("c001", "c002") in pairs or ("c002", "c001") in pairs

    def test_no_match_different_strings(self):
        """Unrelated strings produce no candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002", "character", "Rem", [{"source": "レム", "translation": "Rem"}]
                ),
            ]
        )
        pairs = _source_substring_candidates(glossary)
        assert len(pairs) == 0

    def test_reverse_containment(self):
        """Longer string containing shorter is detected from either direction."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Emperor Vincent Volakia",
                    [
                        {
                            "source": "ヴィンセント・ヴォラキア皇帝",
                            "translation": "Emperor Vincent Volakia",
                        },
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Vincent Volakia",
                    [
                        {"source": "ヴィンセント・ヴォラキア", "translation": "Vincent Volakia"},
                    ],
                ),
            ]
        )
        pairs = _source_substring_candidates(glossary)
        assert len(pairs) == 1

    def test_single_char_ignored(self):
        """Single-character sources do not create false positives."""
        glossary = Glossary(
            entities=[
                _make_entity("c001", "character", "A", [{"source": "あ", "translation": "A"}]),
                _make_entity("c002", "character", "Aa", [{"source": "ああ", "translation": "Aa"}]),
            ]
        )
        # Single char "あ" is len 1, should not match "ああ" via substring
        pairs = _source_substring_candidates(glossary)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Candidate generation — Translation containment
# ---------------------------------------------------------------------------


class TestTranslationContainmentCandidates:
    """Translation name containment heuristic."""

    def test_canonical_containment(self):
        """'Vincent Volakia' contained in 'Emperor Vincent Volakia'."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Vincent Volakia",
                    [
                        {"source": "ヴィンセント", "translation": "Vincent Volakia"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Emperor Vincent Volakia",
                    [
                        {"source": "ヴォラキア皇帝", "translation": "Emperor Vincent Volakia"},
                    ],
                ),
            ]
        )
        pairs = _translation_containment_candidates(glossary)
        assert len(pairs) == 1

    def test_surface_form_translation_containment(self):
        """Surface form translation is also checked, not just canonical."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "translation": "Abel-chan"},
                    ],
                ),
            ]
        )
        pairs = _translation_containment_candidates(glossary)
        assert len(pairs) == 1

    def test_no_containment(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002", "character", "Rem", [{"source": "レム", "translation": "Rem"}]
                ),
            ]
        )
        pairs = _translation_containment_candidates(glossary)
        assert len(pairs) == 0

    def test_single_char_translation_ignored(self):
        """Single-character translation strings are not matched."""
        glossary = Glossary(
            entities=[
                _make_entity("c001", "character", "A", [{"source": "ア", "translation": "A"}]),
                _make_entity("c002", "character", "AB", [{"source": "アブ", "translation": "AB"}]),
            ]
        )
        pairs = _translation_containment_candidates(glossary)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Candidate generation — shared reading
# ---------------------------------------------------------------------------


class TestSharedReadingCandidates:
    """Shared non-null reading across entities."""

    def test_shared_reading_creates_candidate(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "reading": "あべる", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Aberu",
                    [
                        {"source": "亜辺流", "reading": "あべる", "translation": "Aberu"},
                    ],
                ),
            ]
        )
        pairs = _shared_reading_candidates(glossary)
        assert len(pairs) == 1

    def test_no_shared_reading(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "reading": "あべる", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "reading": "れむ", "translation": "Rem"},
                    ],
                ),
            ]
        )
        pairs = _shared_reading_candidates(glossary)
        assert len(pairs) == 0

    def test_null_reading_not_matched(self):
        """Null readings do not create false positives."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "reading": None, "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "reading": None, "translation": "Rem"},
                    ],
                ),
            ]
        )
        pairs = _shared_reading_candidates(glossary)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Candidate generation — alias overlap
# ---------------------------------------------------------------------------


class TestAliasOverlapCandidates:
    """Alias of entity A matches surface form or alias of entity B."""

    def test_alias_matches_surface_source(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Vincent Volakia",
                    [
                        {"source": "ヴィンセント・ヴォラキア", "translation": "Vincent Volakia"},
                    ],
                    aliases=["アベル"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
            ]
        )
        pairs = _alias_overlap_candidates(glossary)
        assert len(pairs) == 1

    def test_alias_matches_alias(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Entity A",
                    [
                        {"source": "Aaa", "translation": "Entity A"},
                    ],
                    aliases=["shared_alias"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Entity B",
                    [
                        {"source": "Bbb", "translation": "Entity B"},
                    ],
                    aliases=["shared_alias"],
                ),
            ]
        )
        pairs = _alias_overlap_candidates(glossary)
        assert len(pairs) == 1

    def test_no_overlap(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                    aliases=["Masked Man"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "translation": "Rem"},
                    ],
                    aliases=["Blue Oni"],
                ),
            ]
        )
        pairs = _alias_overlap_candidates(glossary)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Candidate generation — Jaro-Winkler similarity
# ---------------------------------------------------------------------------


class TestJwSimilarityCandidates:
    """Jaro-Winkler similarity on source forms and translation names."""

    def test_similar_romanisation(self):
        """Petelgeuse vs Petelgeous should be a candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Petelgeuse",
                    [
                        {"source": "ペテルギウス", "translation": "Petelgeuse"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Petelgeous",
                    [
                        {"source": "ペテルジュース", "translation": "Petelgeous"},
                    ],
                ),
            ]
        )
        pairs = _jw_similarity_candidates(glossary, threshold=0.75)
        assert len(pairs) >= 1

    def test_below_threshold_no_candidate(self):
        """Very different strings are not candidates."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "translation": "Rem"},
                    ],
                ),
            ]
        )
        pairs = _jw_similarity_candidates(glossary, threshold=0.75)
        assert len(pairs) == 0

    def test_japanese_source_similarity(self):
        """Similar Japanese strings (honorific suffix) should be candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Subaru",
                    [
                        {"source": "スバル", "translation": "Subaru"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Subaru-kun",
                    [
                        {"source": "スバルくん", "translation": "Subaru-kun"},
                    ],
                ),
            ]
        )
        pairs = _jw_similarity_candidates(glossary, threshold=0.75)
        assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# Cross-category candidates
# ---------------------------------------------------------------------------


class TestCrossCategoryCandidates:
    """Category is a soft signal — cross-category pairs are allowed."""

    def test_cross_category_pair_generated(self):
        """character vs title with shared translation creates a candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "translation": "Abel-chan"},
                    ],
                ),
            ]
        )
        # JP substring should still fire across categories.
        pairs = _source_substring_candidates(glossary)
        assert len(pairs) == 1

    def test_generate_cluster_candidates_includes_cross_category(self):
        """Top-level candidate generation allows cross-category pairs."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel",
                    [
                        {"source": "アベル様", "translation": "Lord Abel"},
                    ],
                ),
            ]
        )
        config = GlossaryClusterConfig()
        pairs = generate_cluster_candidates(glossary, config)
        assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# Entity merge
# ---------------------------------------------------------------------------


class TestMergeEntities:
    """Tests for the merge_entities function."""

    def test_surface_forms_union(self):
        """Surface forms from loser are added to winner."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Vincent Volakia",
            [
                {"source": "ヴィンセント・ヴォラキア", "translation": "Vincent Volakia"},
            ],
        )
        merge_entities(winner, loser, "Abel")
        sources = [sf.source for sf in winner.surface_forms]
        assert "アベル" in sources
        assert "ヴィンセント・ヴォラキア" in sources

    def test_dedup_by_source(self):
        """Same source in both entities does not create duplicate surface forms."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "occurrence_count": 3},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "occurrence_count": 2},
            ],
        )
        merge_entities(winner, loser)
        sources = [sf.source for sf in winner.surface_forms]
        assert sources.count("アベル") == 1
        # Occurrence counts should be summed.
        assert winner.surface_forms[0].occurrence_count == 5

    def test_dedup_preserves_alternate_translation_as_variant(self):
        """Same source, different translation: alternate is preserved in translation_variants."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Aberu",
            [
                {"source": "アベル", "translation": "Aberu"},
            ],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        assert sf.translation == "Abel"  # Winner's form kept.
        assert "Aberu" in sf.translation_variants

    def test_preferred_canonical_name(self):
        """preferred_canonical_name overrides winner's canonical."""
        winner = _make_entity("c001", "character", "Abel")
        loser = _make_entity("c002", "character", "Vincent Volakia")
        merge_entities(winner, loser, preferred_canonical_name="Vincent Volakia")
        assert winner.canonical_name == "Vincent Volakia"

    def test_aliases_union(self):
        winner = _make_entity("c001", "character", "Abel", aliases=["Masked Man"])
        loser = _make_entity("c002", "character", "Vincent", aliases=["Emperor", "Masked Man"])
        merge_entities(winner, loser)
        assert "Masked Man" in winner.aliases
        assert "Emperor" in winner.aliases
        # No duplicates.
        assert winner.aliases.count("Masked Man") == 1

    def test_nicknames_merge_winner_wins(self):
        winner = _make_entity("c001", "character", "Abel", nicknames={"Subaru": "Abel"})
        loser = _make_entity(
            "c002", "character", "Vincent", nicknames={"Subaru": "Vincent", "Rem": "Emperor"}
        )
        merge_entities(winner, loser)
        assert winner.nicknames["Subaru"] == "Abel"  # Winner wins on conflict.
        assert winner.nicknames["Rem"] == "Emperor"

    def test_speech_style_accumulated(self):
        winner = _make_entity("c001", "character", "Abel", speech_style="calm and composed")
        loser = _make_entity("c002", "character", "Vincent", speech_style="authoritative tone")
        merge_entities(winner, loser)
        assert "calm and composed" in winner.speech_style
        assert "authoritative tone" in winner.speech_style

    def test_speech_style_no_duplicates(self):
        winner = _make_entity("c001", "character", "Abel", speech_style="calm and composed")
        loser = _make_entity("c002", "character", "Vincent", speech_style="calm and composed")
        merge_entities(winner, loser)
        assert winner.speech_style.count("calm and composed") == 1

    def test_summary_merge(self):
        winner = _make_entity("c001", "character", "Abel", summary="A masked traveler.")
        loser = _make_entity("c002", "character", "Vincent", summary="Emperor of Volakia.")
        merge_entities(winner, loser)
        assert "A masked traveler." in winner.summary
        assert "Emperor of Volakia." in winner.summary

    def test_summary_dedup(self):
        """Duplicate summary text is not appended."""
        winner = _make_entity("c001", "character", "Abel", summary="A masked traveler.")
        loser = _make_entity("c002", "character", "Vincent", summary="A masked traveler.")
        merge_entities(winner, loser)
        assert winner.summary.count("A masked traveler.") == 1

    def test_notes_merge(self):
        winner = _make_entity("c001", "character", "Abel", notes="Wears a mask.")
        loser = _make_entity("c002", "character", "Vincent", notes="True identity hidden.")
        merge_entities(winner, loser)
        assert "Wears a mask." in winner.notes
        assert "True identity hidden." in winner.notes

    def test_earliest_first_seen_chunk_kept(self):
        winner = _make_entity("c001", first_seen_chunk="0002.001")
        loser = _make_entity("c002", first_seen_chunk="0001.003")
        merge_entities(winner, loser)
        assert winner.first_seen_chunk == "0001.003"

    def test_latest_evidence_chunk_kept(self):
        winner = _make_entity("c001", latest_evidence_chunk="0005.010")
        loser = _make_entity("c002", latest_evidence_chunk="0008.002")
        merge_entities(winner, loser)
        assert winner.latest_evidence_chunk == "0008.002"

    def test_context_hints_union_across_surface_forms(self):
        """Context hints from loser's surface forms are preserved."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "context_hints": ["hint A"]},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Vincent",
            [
                {"source": "ヴィンセント", "translation": "Vincent", "context_hints": ["hint B"]},
            ],
        )
        merge_entities(winner, loser)
        # Winner's form keeps its hint.
        assert "hint A" in winner.surface_forms[0].context_hints
        # Loser's form (now on winner) keeps its hint.
        vincent_sf = next(sf for sf in winner.surface_forms if sf.source == "ヴィンセント")
        assert "hint B" in vincent_sf.context_hints

    def test_source_books_union(self):
        winner = _make_entity("c001", source_books=["vol5"])
        loser = _make_entity("c002", source_books=["vol5", "vol6"])
        merge_entities(winner, loser)
        assert "vol5" in winner.source_books
        assert "vol6" in winner.source_books
        assert winner.source_books.count("vol5") == 1

    def test_surface_form_reading_backfill(self):
        """If winner's surface form has no reading but loser's does, backfill it."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "reading": None},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "reading": "あべる"},
            ],
        )
        merge_entities(winner, loser)
        assert winner.surface_forms[0].reading == "あべる"

    def test_surface_form_first_seen_earliest(self):
        """Merged surface form keeps the earliest first_seen_chunk."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "first_seen_chunk": "0003.001"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "first_seen_chunk": "0001.005"},
            ],
        )
        merge_entities(winner, loser)
        assert winner.surface_forms[0].first_seen_chunk == "0001.005"


# ---------------------------------------------------------------------------
# ID remapping
# ---------------------------------------------------------------------------


class TestRemapEntityId:
    """Remap entity IDs after merges within a batch."""

    def test_no_remap_needed(self):
        decisions = [("c001", "c002", "Abel")]
        result = remap_entity_id(decisions, {})
        assert result == [("c001", "c002", "Abel")]

    def test_remap_absorbed_id(self):
        """If c002 was merged into c001, a decision referencing c002 resolves to c001."""
        id_map = {"c002": "c001"}
        decisions = [("c002", "c003", "Abel")]
        result = remap_entity_id(decisions, id_map)
        assert result == [("c001", "c003", "Abel")]

    def test_self_merge_dropped(self):
        """If both IDs resolve to the same entity, the decision is dropped."""
        id_map = {"c002": "c001"}
        decisions = [("c001", "c002", "Abel")]
        result = remap_entity_id(decisions, id_map)
        assert len(result) == 0

    def test_chain_remap(self):
        """c003 -> c002 -> c001 resolves correctly."""
        id_map = {"c003": "c002", "c002": "c001"}
        decisions = [("c003", "c004", "Abel")]
        result = remap_entity_id(decisions, id_map)
        assert result == [("c001", "c004", "Abel")]


# ---------------------------------------------------------------------------
# Entity rendering for prompt
# ---------------------------------------------------------------------------


class TestRenderEntityForClusterPrompt:
    """render_entity_for_cluster_prompt produces compact text."""

    def test_basic_rendering(self):
        entity = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel", "reading": "あべる"},
            ],
            summary="A masked traveler.",
        )
        text = render_entity_for_cluster_prompt(entity)
        assert "c001" in text
        assert "character" in text
        assert "Abel" in text
        assert "アベル" in text
        assert "あべる" in text
        assert "A masked traveler." in text

    def test_includes_context_hints(self):
        entity = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {
                    "source": "アベル",
                    "translation": "Abel",
                    "context_hints": ["same person as ヴィンセント"],
                },
            ],
        )
        text = render_entity_for_cluster_prompt(entity)
        assert "same person as ヴィンセント" in text

    def test_includes_aliases_and_speech(self):
        entity = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel"}],
            aliases=["Masked Man"],
            speech_style="calm and composed",
        )
        text = render_entity_for_cluster_prompt(entity)
        assert "Masked Man" in text
        assert "calm and composed" in text


# ---------------------------------------------------------------------------
# Clustering report
# ---------------------------------------------------------------------------


class TestClusterReport:
    """write_cluster_report creates a markdown report."""

    def test_no_merges(self, tmp_path):
        report_path = tmp_path / "report.md"
        write_cluster_report(report_path, [], 1, 5)
        text = report_path.read_text(encoding="utf-8")
        assert "No duplicate entities found" in text
        assert "Iterations: 1" in text
        assert "candidate pairs evaluated: 5" in text

    def test_with_merges(self, tmp_path):
        report_path = tmp_path / "report.md"
        merge_log = [
            {
                "winner_id": "c001",
                "loser_id": "c002",
                "winner_name": "Abel",
                "loser_name": "Vincent Volakia",
                "result_name": "Abel",
                "reasoning": "Same character.",
                "surface_forms_added": ["`ヴィンセント` -> Vincent Volakia"],
            }
        ]
        write_cluster_report(report_path, merge_log, 2, 10)
        text = report_path.read_text(encoding="utf-8")
        assert "Abel <- Vincent Volakia" in text
        assert "c001" in text
        assert "c002" in text
        assert "Same character." in text
        assert "Merges performed: 1" in text


# ---------------------------------------------------------------------------
# Top-level generate_cluster_candidates
# ---------------------------------------------------------------------------


class TestGenerateClusterCandidates:
    """generate_cluster_candidates runs all heuristics and deduplicates."""

    def test_deduplication(self):
        """A pair found by multiple heuristics appears only once."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "translation": "Abel-chan"},
                    ],
                ),
            ]
        )
        # Both JP substring and translation containment should fire,
        # but the result should be deduplicated.
        config = GlossaryClusterConfig()
        pairs = generate_cluster_candidates(glossary, config)
        assert len(pairs) == 1

    def test_empty_glossary(self):
        glossary = Glossary(entities=[])
        config = GlossaryClusterConfig()
        pairs = generate_cluster_candidates(glossary, config)
        assert len(pairs) == 0

    def test_single_entity(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
            ]
        )
        config = GlossaryClusterConfig()
        pairs = generate_cluster_candidates(glossary, config)
        assert len(pairs) == 0

    def test_ordered_pair_consistency(self):
        """Pairs are always ordered (smaller ID first)."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "z999", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "a001",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        config = GlossaryClusterConfig()
        pairs = generate_cluster_candidates(glossary, config)
        for a, b in pairs:
            assert a < b


# ---------------------------------------------------------------------------
# Iterative clustering logic
# ---------------------------------------------------------------------------


class TestIterativeClustering:
    """Transitive merges and iteration capping."""

    def test_transitive_merge(self):
        """Merging A+B exposes match to C in next iteration."""
        # A has surface form "アベル", B has "アベルちゃん", C has "アベル様"
        # A and B match via substring. After merge, A gains "アベルちゃん",
        # which does not directly match C. But C's "アベル様" contains "アベル"
        # from the merged A, so the next iteration should find A-C.
        a = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "translation": "Abel"},
            ],
        )
        b = _make_entity(
            "c002",
            "character",
            "Abel-chan",
            [
                {"source": "アベルちゃん", "translation": "Abel-chan"},
            ],
        )
        c = _make_entity(
            "c003",
            "character",
            "Lord Abel",
            [
                {"source": "アベル様", "translation": "Lord Abel"},
            ],
        )
        glossary = Glossary(entities=[a, b, c])
        config = GlossaryClusterConfig()

        # Iteration 1: A-B match, A-C also matches (アベル in アベル様).
        pairs_iter1 = generate_cluster_candidates(glossary, config)
        assert len(pairs_iter1) >= 2  # A-B and A-C (and possibly B-C)

        # Simulate merging A+B.
        merge_entities(a, b)
        glossary.entities.remove(b)

        # Iteration 2: A now has both forms, C still matches.
        pairs_iter2 = generate_cluster_candidates(glossary, config)
        # Should still find A-C.
        assert any("c001" in pair and "c003" in pair for pair in pairs_iter2)

    def test_iteration_cap(self):
        """Max iterations config is respected even if candidates remain.

        This test verifies the config value is accessible and reasonable.
        The full iteration test is in the integration test below.
        """
        config = GlossaryClusterConfig(max_iterations=2)
        assert config.max_iterations == 2


# ---------------------------------------------------------------------------
# Integration test: glossary_cluster with mocked LLM
# ---------------------------------------------------------------------------


class TestGlossaryClusterIntegration:
    """Full glossary_cluster stage with mocked LLM."""

    def _setup(self, tmp_path):
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Create a glossary with two obvious duplicates.
        # Written to glossary_build.json (cluster's input).
        glossary = Glossary(
            entities=[
                _make_entity(
                    "character_000001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                    summary="A masked traveler.",
                ),
                _make_entity(
                    "character_000002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "translation": "Abel-chan"},
                    ],
                    summary="Abel with honorific.",
                ),
                _make_entity(
                    "place_000001",
                    "place",
                    "Lugnica",
                    [
                        {"source": "ルグニカ", "translation": "Lugnica"},
                    ],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        return work, config, state

    def test_merges_confirmed_pair(self, tmp_path):
        """Confirmed same_entity pair gets merged; unrelated entity untouched."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific suffix.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary = glossary_cluster(work, config, state)

        # Should have merged: 2 characters -> 1, plus the place.
        assert len(glossary.entities) == 2
        abel = next(e for e in glossary.entities if e.entity_id == "character_000001")
        assert "アベル" in [sf.source for sf in abel.surface_forms]
        assert "アベルちゃん" in [sf.source for sf in abel.surface_forms]
        assert abel.canonical_name == "Abel"

        # Place entity untouched.
        lugnica = next(e for e in glossary.entities if e.entity_id == "place_000001")
        assert lugnica.canonical_name == "Lugnica"

    def test_no_merge_when_llm_says_not_same(self, tmp_path):
        """If LLM says not same_entity, no merge happens."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=False,
                    reasoning="Different characters despite similar names.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary = glossary_cluster(work, config, state)

        # All three entities remain.
        assert len(glossary.entities) == 3

    def test_zero_candidates_completes_successfully(self, tmp_path):
        """Stage completes even when no candidates are generated."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Glossary with no possible candidates — written to build output.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "character_000001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "translation": "Abel"},
                    ],
                ),
                _make_entity(
                    "place_000001",
                    "place",
                    "Lugnica",
                    [
                        {"source": "ルグニカ", "translation": "Lugnica"},
                    ],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # No LLM client should be created when there are zero candidates.
        result = glossary_cluster(work, config, state)

        assert len(result.entities) == 2
        # Cluster output should exist.
        assert glossary_cluster_path(work).exists()
        # Report should exist.
        report = (work / "glossary_cluster_report.md").read_text(encoding="utf-8")
        assert "No duplicate entities found" in report

    def test_report_written(self, tmp_path):
        """Clustering report is written to disk."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary_cluster(work, config, state)

        report_path = work / "glossary_cluster_report.md"
        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert "Merges performed: 1" in text

    def test_state_tracking_iteration_level(self, tmp_path):
        """Stage uses iteration-level item IDs (iter1, iter2, ...)."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary_cluster(work, config, state)

        assert state.stages["glossary_cluster"].status == "completed"
        # Iteration-level item should be recorded.
        assert "glossary_cluster:iter1" in state.items

    def test_raises_when_build_not_completed(self, tmp_path):
        """glossary_cluster raises if glossary_build is not completed."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        # Only mark stages before build.
        for stage in ("extract", "clean", "classify", "chunk"):
            mark_stage_started(work, state, stage)
            mark_stage_completed(work, state, stage)

        with pytest.raises(RuntimeError, match="Glossary build stage not completed"):
            glossary_cluster(work, config, state)

    def test_skip_when_already_completed(self, tmp_path):
        """glossary_cluster skips when already completed (no force)."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(decisions=[])

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            # First run.
            glossary_cluster(work, config, state)

        # Second run should skip without calling LLM.
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            glossary = glossary_cluster(work, config, state)
            # LLM should not be called on second run.
            instance.complete_json.assert_not_called()

    # -- Stage file flow tests ---------------------------------------------------

    def test_cluster_reads_from_build_output(self, tmp_path):
        """Clustering loads from glossary_build.json, not glossary.json."""
        work, config, state = self._setup(tmp_path)

        # glossary.json should NOT exist — only glossary_build.json.
        assert not glossary_path(work).exists()
        assert glossary_build_path(work).exists()

        mock_response = GlossaryClusterResponse(decisions=[])

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        # Should have loaded all 3 entities from build output.
        assert len(result.entities) == 3

    def test_cluster_writes_to_glossary_cluster_json(self, tmp_path):
        """Output goes to glossary_cluster.json, not glossary.json."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state)

        # Cluster output exists.
        assert glossary_cluster_path(work).exists()
        cluster_glossary = _load_glossary(work, glossary_cluster_path(work))
        assert len(cluster_glossary.entities) == 2  # Merged 3 -> 2.

    def test_build_output_not_mutated_by_cluster(self, tmp_path):
        """glossary_build.json is byte-identical before and after clustering."""
        work, config, state = self._setup(tmp_path)

        # Record build output bytes before clustering.
        build_bytes_before = glossary_build_path(work).read_bytes()

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state)

        # Build output must not have been mutated.
        build_bytes_after = glossary_build_path(work).read_bytes()
        assert build_bytes_before == build_bytes_after

    def test_cluster_force_rereads_build_output(self, tmp_path):
        """--force deletes cluster output, re-reads from glossary_build.json."""
        work, config, state = self._setup(tmp_path)
        original_count = len(_load_glossary(work, glossary_build_path(work)).entities)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        # First run: merges 2 characters into 1.
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)
        assert len(result.entities) == original_count - 1

        # Force re-run: should re-read from glossary_build.json.
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state, force=True)
            instance.complete_json.assert_called()

        # Same merge result — started fresh from build output.
        assert len(result.entities) == original_count - 1

    def test_cluster_force_deletes_downstream(self, tmp_path):
        """--force on cluster deletes glossary.json (reconcile output)."""
        work, config, state = self._setup(tmp_path)

        # Create a fake reconcile output at glossary.json.
        glossary_path(work).write_text("{}", encoding="utf-8")
        assert glossary_path(work).exists()

        mock_response = GlossaryClusterResponse(decisions=[])

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state, force=True)

        # Downstream reconcile output should have been deleted by --force.
        # (It's not recreated by clustering.)
        assert not glossary_path(work).exists()

    def test_cluster_resume_loads_partial_output(self, tmp_path):
        """On resume, clustering loads from glossary_cluster.json (partial merges)."""
        work, config, state = self._setup(tmp_path)

        # Simulate a partially-merged cluster output: manually merge
        # one pair and write to glossary_cluster.json.
        glossary = _load_glossary(work, glossary_build_path(work))
        merge_entities(glossary.entities[0], glossary.entities[1])
        glossary.entities.pop(1)
        _save_glossary(work, glossary, glossary_cluster_path(work))

        # Set up state as if iter1 failed mid-way.
        mark_stage_started(work, state, "glossary_cluster")
        mark_item_started(work, state, "glossary_cluster", "iter1")
        mark_item_failed(work, state, "glossary_cluster", "iter1", "simulated failure")

        # Resume — should load from glossary_cluster.json (2 entities),
        # not glossary_build.json (3 entities).
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = GlossaryClusterResponse(decisions=[])
            result = glossary_cluster(work, config, state)

        # Should have 2 entities (from the partial cluster output).
        assert len(result.entities) == 2

    def test_cluster_clean_start_loads_build_output(self, tmp_path):
        """When glossary_cluster.json doesn't exist, loads from glossary_build.json."""
        work, config, state = self._setup(tmp_path)

        # Ensure no cluster output exists.
        assert not glossary_cluster_path(work).exists()

        mock_response = GlossaryClusterResponse(decisions=[])

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        # Should have loaded all 3 entities from build output.
        assert len(result.entities) == 3

    def test_force_deletes_cluster_meta(self, tmp_path):
        """--force clears the cluster meta sidecar."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state)

        # Cluster meta should exist after first run.
        assert _cluster_meta_path(work).exists()
        meta = _load_cluster_meta(work)
        assert len(meta.merge_log) == 1

        # Force re-run should clear it.
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state, force=True)

        # Meta should have been rebuilt fresh with just this run's data.
        meta = _load_cluster_meta(work)
        assert len(meta.merge_log) == 1  # From the re-run, not accumulated.
        assert 1 in meta.completed_iterations

    # -- Resume / retry-failed tests ------------------------------------------

    def test_resume_after_iteration_failure(self, tmp_path):
        """After iter1 failure, resume re-runs iter1 from fresh candidates."""
        work, config, state = self._setup(tmp_path)

        # Simulate a failed iter1: mark stage started, iter1 failed.
        # No snapshot needed — build output is never mutated.
        mark_stage_started(work, state, "glossary_cluster")
        mark_item_started(work, state, "glossary_cluster", "iter1")
        mark_item_failed(work, state, "glossary_cluster", "iter1", "simulated error")

        # Now resume — iter1 should re-run, loading from glossary_build.json.
        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        assert len(result.entities) == 2  # Merged 2 chars -> 1 + place.
        assert state.stages["glossary_cluster"].status == "completed"

    def test_retry_failed_reruns_failed_iteration(self, tmp_path):
        """--retry-failed re-runs a failed iteration."""
        work, config, state = self._setup(tmp_path)

        # Complete a first run.
        mock_response_merge = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response_merge
            glossary_cluster(work, config, state)

        assert state.stages["glossary_cluster"].status == "completed"

        # --retry-failed should re-open and process any failed items.
        # Since all items completed successfully, this should be a no-op.
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = GlossaryClusterResponse(decisions=[])
            result = glossary_cluster(work, config, state, retry_failed=True)

        assert state.stages["glossary_cluster"].status == "completed"

    def test_cluster_meta_persists_across_iterations(self, tmp_path):
        """Cluster meta accumulates merge_log across completed iterations."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state)

        meta = _load_cluster_meta(work)
        assert 1 in meta.completed_iterations
        assert len(meta.merge_log) == 1
        assert meta.total_candidates_evaluated > 0

    def test_on_progress_called_with_iteration_ids(self, tmp_path):
        """on_progress receives iteration-level IDs (iter1, iter2, ...)."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        progress_calls = []

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary_cluster(
                work,
                config,
                state,
                on_progress=lambda item_id: progress_calls.append(item_id),
            )

        assert len(progress_calls) > 0
        # All progress IDs should be iteration-level.
        for pid in progress_calls:
            assert pid.startswith("iter")


# ---------------------------------------------------------------------------
# Issue 3: _remap_build_meta_conflicts
# ---------------------------------------------------------------------------


class TestRemapBuildMetaConflicts:
    """Build-meta conflict records should follow surviving entity after merge."""

    def test_remap_loser_to_winner(self):
        """Loser's conflict record is remapped to winner's entity_id."""
        from dao_bridge.glossary import _BuildMeta, _ConflictRecord, _remap_build_meta_conflicts

        meta = _BuildMeta(
            conflicts=[
                _ConflictRecord(
                    entity_id="c002",
                    source_form="アベル",
                    current_translation="Aberu",
                    alternatives=[
                        {"translation": "Abel", "context_snippet": "batch 2", "batch_id": "0001.b1"}
                    ],
                ),
            ]
        )
        _remap_build_meta_conflicts(meta, loser_id="c002", winner_id="c001")

        assert len(meta.conflicts) == 1
        assert meta.conflicts[0].entity_id == "c001"
        assert meta.conflicts[0].alternatives[0]["translation"] == "Abel"

    def test_merge_duplicate_records(self):
        """When both winner and loser have conflict records, they are merged."""
        from dao_bridge.glossary import _BuildMeta, _ConflictRecord, _remap_build_meta_conflicts

        meta = _BuildMeta(
            conflicts=[
                _ConflictRecord(
                    entity_id="c001",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[
                        {
                            "translation": "Aberu",
                            "context_snippet": "batch 1",
                            "batch_id": "0000.b1",
                        }
                    ],
                    category_variants=["character"],
                ),
                _ConflictRecord(
                    entity_id="c002",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[
                        {
                            "translation": "Abeel",
                            "context_snippet": "batch 3",
                            "batch_id": "0002.b1",
                        }
                    ],
                    category_variants=["title"],
                ),
            ]
        )
        _remap_build_meta_conflicts(meta, loser_id="c002", winner_id="c001")

        # Should be merged into one record.
        assert len(meta.conflicts) == 1
        record = meta.conflicts[0]
        assert record.entity_id == "c001"
        # Both alternatives should be present.
        alt_translations = {a["translation"] for a in record.alternatives}
        assert "Aberu" in alt_translations
        assert "Abeel" in alt_translations
        # Both category variants should be present.
        assert "character" in record.category_variants
        assert "title" in record.category_variants

    def test_dedup_alternatives_on_merge(self):
        """Duplicate alternatives (same translation) are not duplicated."""
        from dao_bridge.glossary import _BuildMeta, _ConflictRecord, _remap_build_meta_conflicts

        meta = _BuildMeta(
            conflicts=[
                _ConflictRecord(
                    entity_id="c001",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[
                        {
                            "translation": "Aberu",
                            "context_snippet": "batch 1",
                            "batch_id": "0000.b1",
                        }
                    ],
                ),
                _ConflictRecord(
                    entity_id="c002",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[
                        {
                            "translation": "Aberu",
                            "context_snippet": "batch 2",
                            "batch_id": "0001.b1",
                        }
                    ],
                ),
            ]
        )
        _remap_build_meta_conflicts(meta, loser_id="c002", winner_id="c001")

        assert len(meta.conflicts) == 1
        # "Aberu" should appear only once.
        assert len(meta.conflicts[0].alternatives) == 1

    def test_no_op_when_loser_has_no_conflicts(self):
        """If the loser has no conflict record, nothing changes."""
        from dao_bridge.glossary import _BuildMeta, _ConflictRecord, _remap_build_meta_conflicts

        meta = _BuildMeta(
            conflicts=[
                _ConflictRecord(
                    entity_id="c001",
                    source_form="アベル",
                    current_translation="Abel",
                    alternatives=[],
                ),
            ]
        )
        _remap_build_meta_conflicts(meta, loser_id="c099", winner_id="c001")

        assert len(meta.conflicts) == 1
        assert meta.conflicts[0].entity_id == "c001"


# ---------------------------------------------------------------------------
# Issue 4: Batch-internal merge order dependency
# ---------------------------------------------------------------------------


class TestBatchInternalMergeOrder:
    """Multiple merges in one LLM batch should chain correctly."""

    def test_chained_merge_in_same_batch(self, tmp_path):
        """LLM confirms A+B and B+C: both merges should succeed."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Three entities: A (アベル), B (アベルちゃん), C (アベル様)
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [{"source": "アベル", "translation": "Abel"}],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
                _make_entity(
                    "c003",
                    "character",
                    "Lord Abel",
                    [{"source": "アベル様", "translation": "Lord Abel"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # LLM confirms BOTH A+B and B+C in the same batch.
        # Old code would remap once upfront; after A+B merge absorbs B,
        # B+C would fail. New iterative code should remap B->A after
        # the first merge, making B+C become A+C.
        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific.",
                ),
                GlossaryClusterDecision(
                    entity_id_a="c002",
                    entity_id_b="c003",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        # All three should be merged into one entity.
        assert len(result.entities) == 1
        abel = result.entities[0]
        assert abel.entity_id == "c001"
        sources = {sf.source for sf in abel.surface_forms}
        assert "アベル" in sources
        assert "アベルちゃん" in sources
        assert "アベル様" in sources

    def test_deep_chain_winner_selection(self, tmp_path):
        """A+B, B+C, C+D in same batch: winner selection correct at depth >1.

        LLM says D is preferred winner for the C+D pair.  After chains
        A+B (B absorbed into A) and B+C (resolves to A+C, C absorbed
        into A), the C+D decision resolves to A+D.  The LLM's
        preferred_entity_id=D should resolve correctly and D should win
        that merge despite A having absorbed two entities already.
        """
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [{"source": "アベル", "translation": "Abel"}],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
                _make_entity(
                    "c003",
                    "character",
                    "Abel-sama",
                    [{"source": "アベル様", "translation": "Abel-sama"}],
                ),
                _make_entity(
                    "c004",
                    "character",
                    "Vincent Volakia",
                    [{"source": "ヴィンセント・ヴォラキア", "translation": "Vincent Volakia"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # A+B: winner=A.  B+C: winner=A (B resolves to A).
        # C+D: winner=D (LLM prefers D; C resolves to A, so it becomes A+D).
        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific.",
                ),
                GlossaryClusterDecision(
                    entity_id_a="c002",
                    entity_id_b="c003",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character with honorific.",
                ),
                GlossaryClusterDecision(
                    entity_id_a="c003",
                    entity_id_b="c004",
                    same_entity=True,
                    preferred_entity_id="c004",
                    preferred_canonical_name="Vincent Volakia",
                    reasoning="True identity is Vincent.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        # All four should merge into one entity.
        assert len(result.entities) == 1
        survivor = result.entities[0]
        # D (c004) was preferred for the last merge, so it should be the
        # winner — meaning c004 is the surviving entity_id.
        assert survivor.entity_id == "c004"
        assert survivor.canonical_name == "Vincent Volakia"
        # All surface forms present.
        sources = {sf.source for sf in survivor.surface_forms}
        assert "アベル" in sources
        assert "アベルちゃん" in sources
        assert "アベル様" in sources
        assert "ヴィンセント・ヴォラキア" in sources

    def test_preferred_entity_id_neither_side_falls_back(self, tmp_path):
        """If preferred_entity_id resolves to neither entity, default winner is used."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [{"source": "アベル", "translation": "Abel"}],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # LLM returns a preferred_entity_id that doesn't match either entity.
        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=True,
                    preferred_entity_id="c999",  # bogus ID
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)

        # Should still merge successfully using default winner (ea = c001).
        assert len(result.entities) == 1
        survivor = result.entities[0]
        assert survivor.entity_id == "c001"
        sources = {sf.source for sf in survivor.surface_forms}
        assert "アベル" in sources
        assert "アベルちゃん" in sources


# ---------------------------------------------------------------------------
# Issue 5: translation_variants in merge
# ---------------------------------------------------------------------------


class TestTranslationVariantsMerge:
    """Conflicting translation on same source goes to translation_variants, not context_hints."""

    def test_translation_variant_stored(self):
        """Alternate translation is stored in translation_variants, not context_hints."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel"}],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Aberu",
            [{"source": "アベル", "translation": "Aberu"}],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        assert "Aberu" in sf.translation_variants
        # Should NOT be in context_hints.
        assert not any("alternate translation" in h for h in sf.context_hints)

    def test_translation_variants_union_from_loser(self):
        """Loser's existing translation_variants are also carried over."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel"}],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Aberu",
            [{"source": "アベル", "translation": "Aberu", "translation_variants": ["Abell"]}],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        assert "Aberu" in sf.translation_variants
        assert "Abell" in sf.translation_variants

    def test_no_duplicate_variants(self):
        """Same variant is not added twice."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel", "translation_variants": ["Aberu"]}],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Aberu",
            [{"source": "アベル", "translation": "Aberu"}],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        assert sf.translation_variants.count("Aberu") == 1

    def test_winner_name_not_added_as_variant(self):
        """The winner's own translation is not added to translation_variants."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel"}],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel copy",
            [{"source": "アベル", "translation": "Abel", "translation_variants": ["Aberu"]}],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        # "Abel" should not appear in variants (it IS the translation).
        assert "Abel" not in sf.translation_variants
        # "Aberu" should be carried over from loser.
        assert "Aberu" in sf.translation_variants


# ---------------------------------------------------------------------------
# Issue 1+2: Cluster force resets downstream reconcile state
# ---------------------------------------------------------------------------


class TestClusterForceResetsDownstreamState:
    """glossary_cluster --force should reset glossary_reconcile state."""

    def test_cluster_force_resets_reconcile_stage_state(self, tmp_path):
        """After cluster --force, reconcile stage should be pending."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Set up build output.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [{"source": "アベル", "translation": "Abel"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # Simulate reconcile as completed.
        from dao_bridge.state import mark_stage_started as _mss, mark_stage_completed as _msc

        _mss(work, state, "glossary_cluster")
        _msc(work, state, "glossary_cluster")
        _mss(work, state, "glossary_reconcile")
        _msc(work, state, "glossary_reconcile")
        glossary_path(work).write_text("{}", encoding="utf-8")

        assert state.stages["glossary_reconcile"].status == "completed"

        # Now run cluster with --force.
        mock_response = GlossaryClusterResponse(decisions=[])
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state, force=True)

        # Reconcile state should be reset to pending.
        assert state.stages["glossary_reconcile"].status == "pending"
        # Reconcile output should be deleted.
        assert not glossary_path(work).exists()


# ---------------------------------------------------------------------------
# Candidate evidence (generate_cluster_candidates returns Candidates dict)
# ---------------------------------------------------------------------------


class TestCandidateEvidence:
    """generate_cluster_candidates carries per-pair Evidence."""

    def test_returns_dict_keyed_by_ordered_pairs(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        assert isinstance(cands, dict)
        assert ("c001", "c002") in cands
        assert isinstance(cands[("c001", "c002")], Evidence)
        # Backward-compatible consumer behaviour over dict keys.
        assert len(cands) == 1
        for a, b in cands:
            assert a < b

    def test_multiple_heuristics_recorded(self):
        """A pair flagged by several heuristics records all of them."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        ev = cands[("c001", "c002")]
        # Source substring (アベル ⊂ アベルちゃん) and translation containment
        # (Abel ⊂ Abel-chan) both fire.
        assert HEURISTIC_SOURCE_SUBSTRING in ev.heuristics
        assert HEURISTIC_TRANSLATION_CONTAINMENT in ev.heuristics

    def test_jw_score_populated_only_when_jw_fired(self):
        """jw_score is set (>= threshold) when JW flags, None otherwise."""
        # Romanisation variants — JW fires, containment does not.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Petelgeuse",
                    [{"source": "ペテルギウス", "translation": "Petelgeuse"}],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Petelgeous",
                    [{"source": "ペテルギウス", "translation": "Petelgeous"}],
                ),
            ]
        )
        config = GlossaryClusterConfig()
        cands = generate_cluster_candidates(glossary, config)
        ev = cands[("c001", "c002")]
        assert HEURISTIC_JW in ev.heuristics
        assert ev.jw_score is not None
        assert ev.jw_score >= config.jw_threshold

    def test_jw_score_none_when_jw_did_not_fire(self):
        """A pair found only via containment has no jw_score."""
        ev = Evidence(heuristics={HEURISTIC_TRANSLATION_CONTAINMENT})
        assert ev.jw_score is None

    def test_same_category_reflected(self):
        same = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(same, GlossaryClusterConfig())
        assert cands[("c001", "c002")].same_category is True

        cross = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel-sama",
                    [{"source": "アベル様", "translation": "Abel-sama"}],
                ),
            ]
        )
        cands2 = generate_cluster_candidates(cross, GlossaryClusterConfig())
        assert cands2[("c001", "t001")].same_category is False

    def test_containment_direction_recorded(self):
        """source_contains / translation_contains record the container id."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        ev = cands[("c001", "c002")]
        # c002 is the container in both source (アベルちゃん ⊃ アベル) and
        # translation (Abel-chan ⊃ Abel).
        assert ev.source_contains == "c002"
        assert ev.translation_contains == "c002"

    def test_containment_direction_none_when_bidirectional(self):
        """Identical forms (no strict container) record None direction."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        ev = cands[("c001", "c002")]
        assert ev.source_contains is None
        assert ev.translation_contains is None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestScoreCandidateConfidence:
    """score_candidate_confidence tiers a pair from its evidence only."""

    def test_two_containment_signals_same_category_high(self):
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=True,
        )
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH

    def test_shared_reading_plus_alias_overlap_same_category_high(self):
        ev = Evidence(
            heuristics={HEURISTIC_SHARED_READING, HEURISTIC_ALIAS_OVERLAP},
            same_category=True,
        )
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH

    def test_high_jw_plus_containment_same_category_high(self):
        ev = Evidence(
            heuristics={HEURISTIC_JW, HEURISTIC_TRANSLATION_CONTAINMENT},
            jw_score=0.93,
            same_category=True,
        )
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH

    def test_single_strong_signal_same_category_medium(self):
        ev = Evidence(heuristics={HEURISTIC_SOURCE_SUBSTRING}, same_category=True)
        assert score_candidate_confidence(ev) == ClusterConfidence.MEDIUM

    def test_two_strong_signals_different_category_medium(self):
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=False,
        )
        assert score_candidate_confidence(ev) == ClusterConfidence.MEDIUM

    def test_jw_just_below_strong_boundary_medium(self):
        """JW 0.78 (< 0.90) is not a strong signal; alone -> MEDIUM."""
        ev = Evidence(heuristics={HEURISTIC_JW}, jw_score=0.78, same_category=True)
        assert score_candidate_confidence(ev) == ClusterConfidence.MEDIUM

    def test_jw_at_090_boundary_counts_as_strong(self):
        """JW exactly 0.90 is strong; with containment + same cat -> HIGH."""
        ev = Evidence(
            heuristics={HEURISTIC_JW, HEURISTIC_SOURCE_SUBSTRING},
            jw_score=0.90,
            same_category=True,
        )
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH

    def test_jw_only_high_magnitude_is_single_signal_medium(self):
        """A high JW alone is one strong signal -> MEDIUM (needs 2)."""
        ev = Evidence(heuristics={HEURISTIC_JW}, jw_score=0.97, same_category=True)
        assert score_candidate_confidence(ev) == ClusterConfidence.MEDIUM

    def test_never_returns_low(self):
        """This phase's scorer never returns LOW for any evidence shape."""
        shapes = [
            Evidence(),
            Evidence(heuristics={HEURISTIC_JW}, jw_score=0.5),
            Evidence(
                heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_SHARED_READING},
                same_category=True,
            ),
            Evidence(heuristics={HEURISTIC_ALIAS_OVERLAP}, same_category=False),
        ]
        for ev in shapes:
            assert score_candidate_confidence(ev) != ClusterConfidence.LOW


class TestScorerKnownFalsePositiveBaseline:
    """Regression baseline: the string-only scorer's unsafe HIGH behaviour.

    With embeddings OFF the string-only scorer cannot distinguish "qualifier
    means same entity" from "qualifier means a distinct rank". Both emit
    containment + JW + same category and score HIGH. This is a KNOWN
    false-positive source (see the addendum in
    build_phases/glossary-cluster-evidence-and-auto-merge.md and the note on
    score_candidate_confidence).

    Phase 2A (embeddings) corroborates with cosine: the SAME string evidence,
    with a depressed cosine, now scores MEDIUM. Both behaviours are asserted
    here — the OFF path preserves the (unsafe) Phase 1 baseline; the ON path
    proves the corroboration fix.
    """

    def test_quasi_immortal_emperor_scores_high_embeddings_off(self):
        """准仙帝/Quasi-Immortal Emperor vs 仙帝/Immortal Emperor are DISTINCT
        adjacent realms, yet the string-only scorer says HIGH (the known,
        unsafe Phase 1 baseline — preserved when embeddings are OFF)."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "t001",
                    "title",
                    "Immortal Emperor",
                    [{"source": "仙帝", "translation": "Immortal Emperor"}],
                ),
                _make_entity(
                    "t002",
                    "title",
                    "Quasi-Immortal Emperor",
                    [{"source": "准仙帝", "translation": "Quasi-Immortal Emperor"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        ev = cands[("t001", "t002")]
        # Containment (仙帝 ⊂ 准仙帝, Immortal Emperor ⊂ Quasi-Immortal Emperor)
        # + high JW + same category.
        assert HEURISTIC_SOURCE_SUBSTRING in ev.heuristics
        assert HEURISTIC_TRANSLATION_CONTAINMENT in ev.heuristics
        # Embeddings OFF — Phase 1 baseline preserved: scores HIGH (unsafe).
        # No config argument: exercises the single-arg backward-compat path.
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH
        # Same result when config is supplied with embeddings explicitly off.
        off_config = GlossaryClusterConfig(embedding_enabled=False)
        assert score_candidate_confidence(ev, off_config) == ClusterConfidence.HIGH

    def test_quasi_immortal_emperor_scores_medium_embeddings_on(self):
        """The 2A fix: same string evidence (containment + JW + same category),
        but a depressed cosine demotes it from HIGH to MEDIUM so the LLM
        adjudicates the adjacent-realm pair."""
        ev = Evidence(
            heuristics={
                HEURISTIC_SOURCE_SUBSTRING,
                HEURISTIC_TRANSLATION_CONTAINMENT,
            },
            same_category=True,
            cosine=0.70,  # below embedding_auto_merge_min_cosine (0.86)
        )
        on_config = GlossaryClusterConfig(embedding_enabled=True)
        assert score_candidate_confidence(ev, on_config) == ClusterConfidence.MEDIUM


# ---------------------------------------------------------------------------
# Auto-merge canonical picker
# ---------------------------------------------------------------------------


class TestPickCanonicalForAutoMerge:
    """pick_canonical_for_auto_merge selects winner/loser/name deterministically."""

    def test_containment_direction_wins(self):
        ea = _make_entity("c001", "character", "Abel")
        eb = _make_entity("c002", "character", "Abel-chan")
        ev = Evidence(translation_contains="c002")
        winner, loser, name = pick_canonical_for_auto_merge(ea, eb, ev)
        assert winner.entity_id == "c002"
        assert loser.entity_id == "c001"
        assert name == "Abel-chan"

    def test_source_containment_direction_used_when_no_translation(self):
        ea = _make_entity("c001", "character", "Abel")
        eb = _make_entity("c002", "character", "Abel-chan")
        ev = Evidence(source_contains="c001")
        winner, _loser, name = pick_canonical_for_auto_merge(ea, eb, ev)
        assert winner.entity_id == "c001"
        assert name == "Abel"

    def test_longer_canonical_name_wins_without_direction(self):
        ea = _make_entity("c001", "character", "Vincent")
        eb = _make_entity("c002", "character", "Vincent Volakia")
        winner, loser, name = pick_canonical_for_auto_merge(ea, eb, Evidence())
        assert winner.entity_id == "c002"
        assert name == "Vincent Volakia"

    def test_earlier_first_seen_chunk_wins_on_equal_length(self):
        ea = _make_entity("c001", "character", "Aaa", first_seen_chunk="0005.001")
        eb = _make_entity("c002", "character", "Bbb", first_seen_chunk="0001.001")
        winner, _loser, _name = pick_canonical_for_auto_merge(ea, eb, Evidence())
        assert winner.entity_id == "c002"

    def test_stable_fallback_to_ea_when_no_signal(self):
        ea = _make_entity("c001", "character", "Aaa")
        eb = _make_entity("c002", "character", "Bbb")
        winner, _loser, name = pick_canonical_for_auto_merge(ea, eb, Evidence())
        assert winner.entity_id == "c001"
        assert name == "Aaa"


# ---------------------------------------------------------------------------
# Auto-merge integration (auto_merge_enabled=True, opt-in)
# ---------------------------------------------------------------------------


def _make_config_auto_merge(work_dir: Path) -> AppConfig:
    """Config with auto-merge explicitly enabled (opt-in)."""
    return _make_config(work_dir, glossary={"cluster": {"auto_merge_enabled": True}})


class TestAutoMergeIntegration:
    """End-to-end glossary_cluster with auto_merge_enabled=True."""

    def test_high_pairs_merge_without_llm_call(self, tmp_path):
        """A HIGH-confidence pair auto-merges; the LLM is never called."""
        work = _setup_work_dir(tmp_path)
        config = _make_config_auto_merge(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Abel / Abel-chan: source substring + translation containment + same
        # category => HIGH.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            result = glossary_cluster(work, config, state)
            # No LLM call for a purely-HIGH candidate set.
            instance.complete_json.assert_not_called()

        assert len(result.entities) == 1
        # Report tags it as auto-merge with the fixed reasoning + counts.
        report = (work / "glossary_cluster_report.md").read_text(encoding="utf-8")
        assert "Auto-merges (high confidence): 1" in report
        assert "LLM-confirmed merges: 0" in report
        assert "**Type:** auto-merge" in report
        assert "HIGH CONFIDENCE AUTO-MERGE" in report

    def test_medium_pair_routed_to_llm(self, tmp_path):
        """A MEDIUM (cross-category) pair still goes to the LLM."""
        work = _setup_work_dir(tmp_path)
        config = _make_config_auto_merge(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # Cross-category containment => same_category False => MEDIUM.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="t001",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)
            instance.complete_json.assert_called()

        assert len(result.entities) == 1
        report = (work / "glossary_cluster_report.md").read_text(encoding="utf-8")
        assert "Auto-merges (high confidence): 0" in report
        assert "LLM-confirmed merges: 1" in report
        assert "**Type:** LLM-confirmed" in report

    def test_mixed_high_and_medium(self, tmp_path):
        """HIGH pair auto-merges (no LLM); MEDIUM pair handled by LLM."""
        work = _setup_work_dir(tmp_path)
        config = _make_config_auto_merge(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        glossary = Glossary(
            entities=[
                # HIGH pair (same-category containment).
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
                # MEDIUM pair (cross-category containment).
                _make_entity(
                    "p001", "place", "Rome", [{"source": "ローマ", "translation": "Rome"}]
                ),
                _make_entity(
                    "t002",
                    "title",
                    "Rome City",
                    [{"source": "ローマシティ", "translation": "Rome City"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # The LLM should only ever see the MEDIUM pair; it declines to merge.
        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="p001",
                    entity_id_b="t002",
                    same_entity=False,
                    reasoning="Different kinds.",
                ),
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)
            # LLM called for the MEDIUM pair only.
            instance.complete_json.assert_called()
            # The prompt must NOT contain the auto-merged HIGH entities.
            sent_prompt = instance.complete_json.call_args_list[0].args[0][0]["content"]
            assert "ローマ" in sent_prompt  # MEDIUM pair present
            assert "アベル" not in sent_prompt  # HIGH pair never sent to LLM

        # HIGH pair merged (Abel/Abel-chan -> 1), MEDIUM pair NOT merged (2).
        names = sorted(e.canonical_name for e in result.entities)
        assert "Rome" in names
        assert "Rome City" in names
        # Abel collapsed to a single entity.
        abel_like = [e for e in result.entities if "Abel" in e.canonical_name]
        assert len(abel_like) == 1
        report = (work / "glossary_cluster_report.md").read_text(encoding="utf-8")
        assert "Auto-merges (high confidence): 1" in report

    def test_disabled_routes_everything_to_llm(self, tmp_path):
        """auto_merge_enabled=False: a HIGH pair still goes to the LLM."""
        work = _setup_work_dir(tmp_path)
        config = _make_config(work)  # default fixture: auto_merge_enabled False
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_name="Abel",
                    reasoning="Same character.",
                ),
            ]
        )
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state)
            # Even though the pair is HIGH, the flag is off -> LLM is consulted.
            instance.complete_json.assert_called()

        assert len(result.entities) == 1
        report = (work / "glossary_cluster_report.md").read_text(encoding="utf-8")
        assert "Auto-merges (high confidence): 0" in report
        assert "LLM-confirmed merges: 1" in report

    def test_auto_merge_absorbs_entity_also_in_medium_pair(self, tmp_path):
        """Invariant: an entity in a HIGH pair that also appears in a MEDIUM
        pair is remapped through id_map; the MEDIUM pair resolves/drops without
        referencing a removed entity (no KeyError, no stale merge)."""
        work = _setup_work_dir(tmp_path)
        config = _make_config_auto_merge(work)
        state = load_state(work)
        _mark_prior_stages_complete(work, state)

        # c001/c002 are a HIGH same-category containment pair.
        # c002 ALSO forms a MEDIUM cross-category pair with t003 (containment,
        # different category). After auto-merging c001+c002, the c002<->t003
        # candidate must remap c002 to the surviving id before the LLM phase.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
                _make_entity(
                    "t003",
                    "title",
                    "Abel-chan the Great",
                    [{"source": "アベルちゃん大王", "translation": "Abel-chan the Great"}],
                ),
            ]
        )
        _save_glossary(work, glossary, glossary_build_path(work))

        # LLM declines the surviving MEDIUM pair (whatever it resolves to).
        mock_response = GlossaryClusterResponse(decisions=[])
        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            # Must not raise (no KeyError on evidence lookup, no stale entity).
            result = glossary_cluster(work, config, state)

        # c001+c002 auto-merged. t003 declined by LLM -> remains separate.
        ids = {e.entity_id for e in result.entities}
        # Loser of the HIGH auto-merge is gone; t003 survives.
        assert "t003" in ids
        assert len(result.entities) == 2


# ---------------------------------------------------------------------------
# Phase 2A — embedding text assembly (pure-logic, no sentence-transformers)
# ---------------------------------------------------------------------------


class TestEntityEmbeddingText:
    """entity_embedding_text assembles enriched text, dropping empty parts.

    Pure string assembly — must NOT import sentence_transformers.
    """

    def test_includes_category_name_sources_translations_summary_hints(self):
        from dao_bridge.glossary_embeddings import entity_embedding_text

        entity = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {
                    "source": "アベル",
                    "translation": "Abel",
                    "context_hints": ["fugitive", "ties to royalty"],
                }
            ],
            summary="A fugitive traveling under the name Abel.",
        )
        text = entity_embedding_text(entity)
        assert "character" in text
        assert "Abel" in text
        assert "アベル" in text
        assert "A fugitive traveling under the name Abel." in text
        assert "fugitive" in text
        assert "ties to royalty" in text

    def test_drops_empty_parts_no_dangling_separators(self):
        from dao_bridge.glossary_embeddings import entity_embedding_text

        # No summary, no hints — those parts must be dropped entirely.
        entity = _make_entity(
            "c001",
            "character",
            "Abel",
            [{"source": "アベル", "translation": "Abel"}],
        )
        text = entity_embedding_text(entity)
        assert ".." not in text  # no empty ". ." segments
        assert not text.startswith(". ")
        assert not text.endswith(". ")
        assert text  # non-empty


# ---------------------------------------------------------------------------
# Phase 2A — confidence scoring with embeddings
# ---------------------------------------------------------------------------


class TestScoreCandidateConfidenceEmbeddings:
    """Cosine-corroborated scoring rules (embeddings ON)."""

    @staticmethod
    def _on() -> GlossaryClusterConfig:
        # Explicit thresholds so these tests assert the SCORING RULES, not the
        # tuned-per-model config defaults (which can change without breaking the
        # rule contract).
        return GlossaryClusterConfig(
            embedding_enabled=True,
            embedding_candidate_threshold=0.55,
            embedding_auto_merge_min_cosine=0.86,
            embedding_low_confidence_max_cosine=0.55,
        )

    def test_embedding_only_low_cosine_is_low(self):
        """Only the embedding heuristic fired + cosine below the LOW floor."""
        ev = Evidence(
            heuristics={HEURISTIC_EMBEDDING},
            same_category=True,
            cosine=0.50,  # < embedding_low_confidence_max_cosine (0.55 here)
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.LOW

    def test_embedding_only_mid_cosine_is_medium_not_low(self):
        """Embedding-only but cosine at/above the LOW floor -> MEDIUM (to LLM)."""
        ev = Evidence(
            heuristics={HEURISTIC_EMBEDDING},
            same_category=True,
            cosine=0.60,  # >= LOW floor (0.55), < auto floor (0.86)
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.MEDIUM

    def test_containment_jw_same_category_low_cosine_medium(self):
        """The 准仙帝/仙帝 fix: strong string evidence + low cosine -> MEDIUM."""
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=True,
            cosine=0.70,  # < embedding_auto_merge_min_cosine (0.86)
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.MEDIUM

    def test_containment_jw_same_category_high_cosine_high(self):
        """Strong string evidence + corroborating high cosine -> HIGH."""
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=True,
            cosine=0.90,  # >= embedding_auto_merge_min_cosine (0.86)
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.HIGH

    def test_strong_embedding_plus_one_strong_string_same_category_high(self):
        """Strong embedding (high cosine) + one strong string signal -> HIGH."""
        ev = Evidence(
            heuristics={HEURISTIC_EMBEDDING, HEURISTIC_SOURCE_SUBSTRING},
            same_category=True,
            cosine=0.88,
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.HIGH

    def test_high_string_evidence_none_cosine_demoted_to_medium(self):
        """Would be HIGH on strings, but cosine is None (no embedding row)
        -> demoted to MEDIUM, never LOW (string evidence still merits LLM)."""
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=True,
            cosine=None,
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.MEDIUM

    def test_strong_string_different_category_high_cosine_still_medium(self):
        """same_category is still required for HIGH even with high cosine."""
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=False,
            cosine=0.95,
        )
        assert score_candidate_confidence(ev, self._on()) == ClusterConfidence.MEDIUM


class TestScoreCandidateConfidenceOffPathInvariant:
    """Backward-compat / None-safety contract: embeddings OFF == Phase 1."""

    def test_off_path_never_returns_low_and_ignores_cosine(self):
        """With embeddings off (config None or disabled), the scorer never
        returns LOW and never consults cosine — even if cosine is set."""
        off = GlossaryClusterConfig(embedding_enabled=False)
        shapes = [
            Evidence(),
            Evidence(heuristics={HEURISTIC_JW}, jw_score=0.5),
            # Cosine set to a LOW-tier value, but OFF path must ignore it.
            Evidence(heuristics={HEURISTIC_EMBEDDING}, cosine=0.10),
            Evidence(
                heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_SHARED_READING},
                same_category=True,
                cosine=0.10,
            ),
        ]
        for ev in shapes:
            # None config -> Phase 1 path.
            assert score_candidate_confidence(ev) != ClusterConfidence.LOW
            # Explicit disabled config -> identical.
            assert score_candidate_confidence(ev, off) != ClusterConfidence.LOW

    def test_off_path_high_unchanged_with_cosine_present(self):
        """A 2+ strong string signal + same-category pair stays HIGH on the OFF
        path regardless of any cosine value (cosine must not be consulted)."""
        off = GlossaryClusterConfig(embedding_enabled=False)
        ev = Evidence(
            heuristics={HEURISTIC_SOURCE_SUBSTRING, HEURISTIC_TRANSLATION_CONTAINMENT},
            same_category=True,
            cosine=0.01,  # would force MEDIUM on the ON path; ignored when off
        )
        assert score_candidate_confidence(ev, off) == ClusterConfidence.HIGH
        assert score_candidate_confidence(ev) == ClusterConfidence.HIGH


# ---------------------------------------------------------------------------
# Phase 2A — embedding candidate generation (requires sentence-transformers)
# ---------------------------------------------------------------------------


class TestEmbeddingCandidateGeneration:
    """Embedding-sourced candidate generation + cosine population.

    These load a real model, so skip when sentence-transformers is absent.
    """

    def test_adds_candidates_above_threshold_and_populates_cosine(self):
        pytest.importorskip("sentence_transformers")
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [{"source": "アベル", "translation": "Abel"}],
                    summary="A fugitive emperor traveling under the name Abel.",
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Vincent Volakia",
                    [{"source": "ヴィンセント", "translation": "Vincent Volakia"}],
                    summary="The emperor of Vollachia, ruling under the name Vincent.",
                ),
            ]
        )
        config = GlossaryClusterConfig(
            embedding_enabled=True,
            # Force every pair to clear the candidate threshold so the test does
            # not depend on a specific model's exact cosine.
            embedding_candidate_threshold=-1.0,
        )
        cands = generate_cluster_candidates(glossary, config)
        pair = ("c001", "c002")
        assert pair in cands
        assert HEURISTIC_EMBEDDING in cands[pair].heuristics
        # Cosine populated on the pair.
        assert cands[pair].cosine is not None
        assert -1.0 <= cands[pair].cosine <= 1.0

    def test_embeddings_off_leaves_cosine_none(self):
        """Embeddings disabled -> no embedding heuristic, cosine stays None."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "translation": "Abel-chan"}],
                ),
            ]
        )
        cands = generate_cluster_candidates(glossary, GlossaryClusterConfig())
        for ev in cands.values():
            assert ev.cosine is None
            assert HEURISTIC_EMBEDDING not in ev.heuristics

    def test_compute_entity_embeddings_aligned_and_normalized(self):
        pytest.importorskip("sentence_transformers")
        import numpy as np

        from dao_bridge.glossary_embeddings import compute_entity_embeddings

        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "translation": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "place",
                    "Vollachia",
                    [{"source": "ヴォラキア", "translation": "Vollachia"}],
                ),
            ]
        )
        entity_ids, emb = compute_entity_embeddings(
            glossary, "paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert entity_ids == ["c001", "c002"]
        assert emb.shape[0] == 2
        # Rows are L2-normalized (cosine == dot product).
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_graceful_error_when_package_missing(self):
        """If sentence-transformers is missing, _load_model raises a clear
        RuntimeError pointing at the install extra."""
        import dao_bridge.glossary_embeddings as ge

        # Simulate the package being unavailable regardless of environment.
        ge._model_cache.clear()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match=r"\[embeddings\]"):
                ge._load_model("does-not-matter")
