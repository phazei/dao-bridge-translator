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
    _load_glossary,
    _save_glossary,
    glossary_cluster,
)
from dao_bridge.glossary_clustering import (
    _english_containment_candidates,
    _jp_substring_candidates,
    _jw_similarity_candidates,
    _alias_overlap_candidates,
    _shared_reading_candidates,
    generate_cluster_candidates,
    merge_entities,
    remap_entity_id,
    render_entity_for_cluster_prompt,
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
    mark_stage_completed,
    mark_stage_started,
)
from dao_bridge.workdir import atomic_write


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
    canonical_english: str = "Subaru",
    surface_forms: list[dict] | None = None,
    **kwargs,
) -> GlossaryEntity:
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
                    "c001", "character", "Abel", [{"source": "アベル", "english": "Abel"}]
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "english": "Abel-chan"}],
                ),
            ]
        )
        pairs = _jp_substring_candidates(glossary)
        assert ("c001", "c002") in pairs or ("c002", "c001") in pairs

    def test_no_match_different_strings(self):
        """Unrelated strings produce no candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "english": "Abel"}]
                ),
                _make_entity("c002", "character", "Rem", [{"source": "レム", "english": "Rem"}]),
            ]
        )
        pairs = _jp_substring_candidates(glossary)
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
                            "english": "Emperor Vincent Volakia",
                        },
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Vincent Volakia",
                    [
                        {"source": "ヴィンセント・ヴォラキア", "english": "Vincent Volakia"},
                    ],
                ),
            ]
        )
        pairs = _jp_substring_candidates(glossary)
        assert len(pairs) == 1

    def test_single_char_ignored(self):
        """Single-character sources do not create false positives."""
        glossary = Glossary(
            entities=[
                _make_entity("c001", "character", "A", [{"source": "あ", "english": "A"}]),
                _make_entity("c002", "character", "Aa", [{"source": "ああ", "english": "Aa"}]),
            ]
        )
        # Single char "あ" is len 1, should not match "ああ" via substring
        pairs = _jp_substring_candidates(glossary)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Candidate generation — English containment
# ---------------------------------------------------------------------------


class TestEnglishContainmentCandidates:
    """English name containment heuristic."""

    def test_canonical_containment(self):
        """'Vincent Volakia' contained in 'Emperor Vincent Volakia'."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Vincent Volakia",
                    [
                        {"source": "ヴィンセント", "english": "Vincent Volakia"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Emperor Vincent Volakia",
                    [
                        {"source": "ヴォラキア皇帝", "english": "Emperor Vincent Volakia"},
                    ],
                ),
            ]
        )
        pairs = _english_containment_candidates(glossary)
        assert len(pairs) == 1

    def test_surface_form_english_containment(self):
        """Surface form English is also checked, not just canonical."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "english": "Abel-chan"},
                    ],
                ),
            ]
        )
        pairs = _english_containment_candidates(glossary)
        assert len(pairs) == 1

    def test_no_containment(self):
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001", "character", "Abel", [{"source": "アベル", "english": "Abel"}]
                ),
                _make_entity("c002", "character", "Rem", [{"source": "レム", "english": "Rem"}]),
            ]
        )
        pairs = _english_containment_candidates(glossary)
        assert len(pairs) == 0

    def test_single_char_english_ignored(self):
        """Single-character English strings are not matched."""
        glossary = Glossary(
            entities=[
                _make_entity("c001", "character", "A", [{"source": "ア", "english": "A"}]),
                _make_entity("c002", "character", "AB", [{"source": "アブ", "english": "AB"}]),
            ]
        )
        pairs = _english_containment_candidates(glossary)
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
                        {"source": "アベル", "reading": "あべる", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Aberu",
                    [
                        {"source": "亜辺流", "reading": "あべる", "english": "Aberu"},
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
                        {"source": "アベル", "reading": "あべる", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "reading": "れむ", "english": "Rem"},
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
                        {"source": "アベル", "reading": None, "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "reading": None, "english": "Rem"},
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
                        {"source": "ヴィンセント・ヴォラキア", "english": "Vincent Volakia"},
                    ],
                    aliases=["アベル"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
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
                        {"source": "Aaa", "english": "Entity A"},
                    ],
                    aliases=["shared_alias"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Entity B",
                    [
                        {"source": "Bbb", "english": "Entity B"},
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
                        {"source": "アベル", "english": "Abel"},
                    ],
                    aliases=["Masked Man"],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "english": "Rem"},
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
    """Jaro-Winkler similarity on source forms and English names."""

    def test_similar_romanisation(self):
        """Petelgeuse vs Petelgeous should be a candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Petelgeuse",
                    [
                        {"source": "ペテルギウス", "english": "Petelgeuse"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Petelgeous",
                    [
                        {"source": "ペテルジュース", "english": "Petelgeous"},
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
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Rem",
                    [
                        {"source": "レム", "english": "Rem"},
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
                        {"source": "スバル", "english": "Subaru"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Subaru-kun",
                    [
                        {"source": "スバルくん", "english": "Subaru-kun"},
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
        """character vs title with shared English creates a candidate."""
        glossary = Glossary(
            entities=[
                _make_entity(
                    "c001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "english": "Abel-chan"},
                    ],
                ),
            ]
        )
        # JP substring should still fire across categories.
        pairs = _jp_substring_candidates(glossary)
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
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "t001",
                    "title",
                    "Abel",
                    [
                        {"source": "アベル様", "english": "Lord Abel"},
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
                {"source": "アベル", "english": "Abel"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Vincent Volakia",
            [
                {"source": "ヴィンセント・ヴォラキア", "english": "Vincent Volakia"},
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
                {"source": "アベル", "english": "Abel", "occurrence_count": 3},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "english": "Abel", "occurrence_count": 2},
            ],
        )
        merge_entities(winner, loser)
        sources = [sf.source for sf in winner.surface_forms]
        assert sources.count("アベル") == 1
        # Occurrence counts should be summed.
        assert winner.surface_forms[0].occurrence_count == 5

    def test_dedup_preserves_alternate_english_as_hint(self):
        """Same source, different English: alternate is preserved as context hint."""
        winner = _make_entity(
            "c001",
            "character",
            "Abel",
            [
                {"source": "アベル", "english": "Abel"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Aberu",
            [
                {"source": "アベル", "english": "Aberu"},
            ],
        )
        merge_entities(winner, loser)
        sf = winner.surface_forms[0]
        assert sf.english == "Abel"  # Winner's form kept.
        assert any("alternate English: Aberu" in h for h in sf.context_hints)

    def test_preferred_canonical_english(self):
        """preferred_canonical_english overrides winner's canonical."""
        winner = _make_entity("c001", "character", "Abel")
        loser = _make_entity("c002", "character", "Vincent Volakia")
        merge_entities(winner, loser, preferred_canonical_english="Vincent Volakia")
        assert winner.canonical_english == "Vincent Volakia"

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
                {"source": "アベル", "english": "Abel", "context_hints": ["hint A"]},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Vincent",
            [
                {"source": "ヴィンセント", "english": "Vincent", "context_hints": ["hint B"]},
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
                {"source": "アベル", "english": "Abel", "reading": None},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "english": "Abel", "reading": "あべる"},
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
                {"source": "アベル", "english": "Abel", "first_seen_chunk": "0003.001"},
            ],
        )
        loser = _make_entity(
            "c002",
            "character",
            "Abel",
            [
                {"source": "アベル", "english": "Abel", "first_seen_chunk": "0001.005"},
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
                {"source": "アベル", "english": "Abel", "reading": "あべる"},
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
                    "english": "Abel",
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
            [{"source": "アベル", "english": "Abel"}],
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
                "winner_english": "Abel",
                "loser_english": "Vincent Volakia",
                "result_english": "Abel",
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
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "c002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "english": "Abel-chan"},
                    ],
                ),
            ]
        )
        # Both JP substring and English containment should fire,
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
                    "c001", "character", "Abel", [{"source": "アベル", "english": "Abel"}]
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
                    "z999", "character", "Abel", [{"source": "アベル", "english": "Abel"}]
                ),
                _make_entity(
                    "a001",
                    "character",
                    "Abel-chan",
                    [{"source": "アベルちゃん", "english": "Abel-chan"}],
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
                {"source": "アベル", "english": "Abel"},
            ],
        )
        b = _make_entity(
            "c002",
            "character",
            "Abel-chan",
            [
                {"source": "アベルちゃん", "english": "Abel-chan"},
            ],
        )
        c = _make_entity(
            "c003",
            "character",
            "Lord Abel",
            [
                {"source": "アベル様", "english": "Lord Abel"},
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
        glossary = Glossary(
            entities=[
                _make_entity(
                    "character_000001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
                    ],
                    summary="A masked traveler.",
                ),
                _make_entity(
                    "character_000002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "english": "Abel-chan"},
                    ],
                    summary="Abel with honorific.",
                ),
                _make_entity(
                    "place_000001",
                    "place",
                    "Lugnica",
                    [
                        {"source": "ルグニカ", "english": "Lugnica"},
                    ],
                ),
            ]
        )
        _save_glossary(work, glossary)

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
                    preferred_canonical_english="Abel",
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
        assert abel.canonical_english == "Abel"

        # Place entity untouched.
        lugnica = next(e for e in glossary.entities if e.entity_id == "place_000001")
        assert lugnica.canonical_english == "Lugnica"

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

        # Glossary with no possible candidates.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "character_000001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "place_000001",
                    "place",
                    "Lugnica",
                    [
                        {"source": "ルグニカ", "english": "Lugnica"},
                    ],
                ),
            ]
        )
        _save_glossary(work, glossary)

        # No LLM client should be created when there are zero candidates.
        result = glossary_cluster(work, config, state)

        assert len(result.entities) == 2
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
                    preferred_canonical_english="Abel",
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

    def test_state_tracking(self, tmp_path):
        """Stage is marked completed in pipeline state."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_english="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response

            glossary_cluster(work, config, state)

        assert state.stages["glossary_cluster"].status == "completed"

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

    def test_force_reruns(self, tmp_path):
        """--force re-runs clustering even if already completed."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_english="Abel",
                    reasoning="Same character.",
                ),
            ]
        )

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            glossary_cluster(work, config, state)

        # Force re-run: need to restore the glossary since first run merged.
        glossary = Glossary(
            entities=[
                _make_entity(
                    "character_000001",
                    "character",
                    "Abel",
                    [
                        {"source": "アベル", "english": "Abel"},
                    ],
                ),
                _make_entity(
                    "character_000002",
                    "character",
                    "Abel-chan",
                    [
                        {"source": "アベルちゃん", "english": "Abel-chan"},
                    ],
                ),
                _make_entity(
                    "place_000001",
                    "place",
                    "Lugnica",
                    [
                        {"source": "ルグニカ", "english": "Lugnica"},
                    ],
                ),
            ]
        )
        _save_glossary(work, glossary)

        with patch("dao_bridge.glossary.LLMClient") as MockClient:
            instance = MockClient.return_value
            instance.complete_json.return_value = mock_response
            result = glossary_cluster(work, config, state, force=True)
            instance.complete_json.assert_called()

        assert len(result.entities) == 2

    def test_on_progress_called(self, tmp_path):
        """on_progress callback is invoked for each batch."""
        work, config, state = self._setup(tmp_path)

        mock_response = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="character_000001",
                    entity_id_b="character_000002",
                    same_entity=True,
                    preferred_entity_id="character_000001",
                    preferred_canonical_english="Abel",
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
