"""Glossary entity clustering: candidate generation and entity merging.

This module implements the clustering stage of the glossary pipeline.
After the build stage creates entities (possibly with duplicates that
build-time linking could not safely resolve), clustering finds and merges
those duplicates through deterministic heuristics followed by LLM
confirmation.

The main entry-point for the stage function is :func:`glossary_cluster`
in ``glossary.py``; this module provides the lower-level primitives it
calls.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from dao_bridge.config import GlossaryClusterConfig
from dao_bridge.schemas import Glossary, GlossaryEntity, SurfaceForm
from dao_bridge.similarity import string_similarity

logger = logging.getLogger("dao_bridge")

# Re-use the same constants as the build stage.
_SPEECH_STYLE_DELIMITER = "\n"
_MAX_SUMMARY_LENGTH = 500

# ---------------------------------------------------------------------------
# Candidate generation — individual heuristics
# ---------------------------------------------------------------------------

CandidatePair = tuple[str, str]
"""An ordered ``(entity_id_a, entity_id_b)`` pair where ``a < b``."""


# Heuristic name constants — single source of truth for tagging + scoring.
HEURISTIC_SOURCE_SUBSTRING = "source_substring"
HEURISTIC_TRANSLATION_CONTAINMENT = "translation_containment"
HEURISTIC_SHARED_READING = "shared_reading"
HEURISTIC_ALIAS_OVERLAP = "alias_overlap"
HEURISTIC_JW = "jaro_winkler"
HEURISTIC_EMBEDDING = "embedding"

# JW magnitude that promotes the Jaro-Winkler signal to "strong" for auto-merge
# scoring. Deliberately higher than the candidate-generation threshold (0.75):
# candidate generation casts a wide net, auto-merge skips the LLM and needs a
# higher bar (mirrors build-stage auto-attach at JW >= 0.95).
_AUTO_MERGE_JW_STRONG = 0.90


@dataclass
class Evidence:
    """Why a candidate pair was generated, and how strong the signals were.

    Carried per pair from candidate generation through to confidence scoring.
    Never persisted; lives only for the duration of a clustering iteration.
    """

    heuristics: set[str] = field(default_factory=set)
    """Names of heuristics that flagged this pair (see HEURISTIC_* constants)."""

    jw_score: float | None = None
    """Best bi-directional Jaro-Winkler score observed for this pair, if the
    JW heuristic flagged it. None when JW did not fire."""

    same_category: bool = False
    """Whether the two entities share a category. A supporting signal, never
    sufficient alone."""

    source_contains: str | None = None
    """For source-substring matches: the entity_id whose source form CONTAINS
    the other's (the longer/more specific form). None if not applicable or
    bidirectional."""

    translation_contains: str | None = None
    """For translation-containment matches: the entity_id whose translation
    CONTAINS the other's. None if not applicable or bidirectional."""

    # Populated by the embedding heuristic when GlossaryClusterConfig
    # .embedding_enabled is True (Phase 2A); None when embeddings are off or the
    # pair has no embedding row.
    cosine: float | None = None
    """Cosine similarity from the embedding heuristic. Populated for ALL candidate
    pairs (string- and embedding-sourced) when ``embedding_enabled`` is True;
    ``None`` when embeddings are disabled."""


Candidates = dict[CandidatePair, Evidence]
"""Candidate pairs keyed by canonically ordered ``(a, b)`` with their evidence."""


class ClusterConfidence(Enum):
    """Confidence tier for a candidate pair."""

    HIGH = "high"      # Auto-merge without LLM.
    MEDIUM = "medium"  # Send to LLM for confirmation (today's default path).
    LOW = "low"        # Reserved for embeddings: weak semantic-only pairs to
    # auto-reject. Not produced in this phase.


def _ordered_pair(id_a: str, id_b: str) -> CandidatePair:
    """Return a canonically ordered pair so ``(A, B)`` and ``(B, A)`` hash the same."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


# -- 1. Source-language substring containment --------------------------------


def _source_substring_candidates(glossary: Glossary) -> set[CandidatePair]:
    """Surface form source substring containment (both directions).

    Catches e.g. ``アベル`` <-> ``アベルちゃん``.
    """
    pairs: set[CandidatePair] = set()
    entities = glossary.entities
    for i, ea in enumerate(entities):
        for j in range(i + 1, len(entities)):
            eb = entities[j]
            if _has_source_substring_overlap(ea, eb):
                pairs.add(_ordered_pair(ea.entity_id, eb.entity_id))
    return pairs


def _has_source_substring_overlap(ea: GlossaryEntity, eb: GlossaryEntity) -> bool:
    """Return True if any surface-form source of *ea* is a substring of *eb*'s (or vice versa)."""
    for sf_a in ea.surface_forms:
        if not sf_a.source:
            continue
        for sf_b in eb.surface_forms:
            if not sf_b.source:
                continue
            # Skip trivially short substrings (1 char) to avoid noise.
            if len(sf_a.source) <= 1 and len(sf_b.source) <= 1:
                continue
            if (len(sf_a.source) > 1 and sf_a.source in sf_b.source) or (
                len(sf_b.source) > 1 and sf_b.source in sf_a.source
            ):
                return True
    return False


# -- 2. Translation containment ---------------------------------------------


def _translation_containment_candidates(glossary: Glossary) -> set[CandidatePair]:
    """Canonical name or surface-form translation containment.

    Catches e.g. ``Vincent Volakia`` <-> ``Emperor Vincent Volakia``.
    """
    pairs: set[CandidatePair] = set()
    entities = glossary.entities
    for i, ea in enumerate(entities):
        for j in range(i + 1, len(entities)):
            eb = entities[j]
            if _has_translation_containment(ea, eb):
                pairs.add(_ordered_pair(ea.entity_id, eb.entity_id))
    return pairs


def _all_translation_forms(entity: GlossaryEntity) -> list[str]:
    """Collect all translation strings from canonical name + surface forms."""
    forms = [entity.canonical_name]
    for sf in entity.surface_forms:
        if sf.translation and sf.translation != entity.canonical_name:
            forms.append(sf.translation)
    return forms


def _has_translation_containment(ea: GlossaryEntity, eb: GlossaryEntity) -> bool:
    """Return True if any translation form of *ea* contains/is contained by *eb*'s."""
    forms_a = _all_translation_forms(ea)
    forms_b = _all_translation_forms(eb)
    for eng_a in forms_a:
        la = eng_a.lower()
        if len(la) <= 1:
            continue
        for eng_b in forms_b:
            lb = eng_b.lower()
            if len(lb) <= 1:
                continue
            if la in lb or lb in la:
                return True
    return False


# -- 3. Shared non-null reading ---------------------------------------------


def _shared_reading_candidates(glossary: Glossary) -> set[CandidatePair]:
    """Two entities sharing a non-null reading on any surface form."""
    # Build reading -> entity_id index.
    reading_index: dict[str, list[str]] = defaultdict(list)
    for entity in glossary.entities:
        seen_readings: set[str] = set()
        for sf in entity.surface_forms:
            if sf.reading and sf.reading not in seen_readings:
                reading_index[sf.reading].append(entity.entity_id)
                seen_readings.add(sf.reading)

    pairs: set[CandidatePair] = set()
    for _reading, eids in reading_index.items():
        if len(eids) < 2:
            continue
        for i, eid_a in enumerate(eids):
            for eid_b in eids[i + 1 :]:
                pairs.add(_ordered_pair(eid_a, eid_b))
    return pairs


# -- 4. Alias overlap -------------------------------------------------------


def _alias_overlap_candidates(glossary: Glossary) -> set[CandidatePair]:
    """Alias of entity A matches a surface-form source or alias of entity B."""
    # Build indexes.
    source_index: dict[str, list[str]] = defaultdict(list)
    alias_index: dict[str, list[str]] = defaultdict(list)
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.source:
                source_index[sf.source].append(entity.entity_id)
        for alias in entity.aliases:
            alias_index[alias].append(entity.entity_id)

    pairs: set[CandidatePair] = set()
    for entity in glossary.entities:
        for alias in entity.aliases:
            # Alias matches another entity's surface form source.
            for other_eid in source_index.get(alias, []):
                if other_eid != entity.entity_id:
                    pairs.add(_ordered_pair(entity.entity_id, other_eid))
            # Alias matches another entity's alias.
            for other_eid in alias_index.get(alias, []):
                if other_eid != entity.entity_id:
                    pairs.add(_ordered_pair(entity.entity_id, other_eid))
    return pairs


# -- 5. Jaro-Winkler similarity ---------------------------------------------


def _jw_similarity_candidates(
    glossary: Glossary,
    threshold: float = 0.75,
) -> set[CandidatePair]:
    """Bi-directional Jaro-Winkler on source forms and translation names.

    Catches romanisation inconsistencies (``Petelgeuse`` vs ``Petelgeous``)
    and honorific variants.
    """
    pairs: set[CandidatePair] = set()
    entities = glossary.entities
    for i, ea in enumerate(entities):
        for j in range(i + 1, len(entities)):
            eb = entities[j]
            if _jw_any_match(ea, eb, threshold):
                pairs.add(_ordered_pair(ea.entity_id, eb.entity_id))
    return pairs


def _jw_any_match(ea: GlossaryEntity, eb: GlossaryEntity, threshold: float) -> bool:
    """Return True if any source-form or translation pair exceeds *threshold*."""
    # Source-form pairs.
    for sf_a in ea.surface_forms:
        if not sf_a.source:
            continue
        for sf_b in eb.surface_forms:
            if not sf_b.source:
                continue
            if string_similarity(sf_a.source, sf_b.source) >= threshold:
                return True

    # Translation pairs (canonical name + surface form translations).
    for tl_a in _all_translation_forms(ea):
        if not tl_a:
            continue
        for tl_b in _all_translation_forms(eb):
            if not tl_b:
                continue
            if string_similarity(tl_a, tl_b) >= threshold:
                return True

    return False


# ---------------------------------------------------------------------------
# Evidence annotators (read-only; reuse the heuristics' own predicate rules)
# ---------------------------------------------------------------------------
#
# These are lookup-only annotations of an already-authoritative pair set. They
# never re-decide *whether* a pair qualifies — the heuristic above already did —
# they only read back a magnitude or direction for pairs that already passed.


def _best_jw_score(ea: GlossaryEntity, eb: GlossaryEntity) -> float | None:
    """Return the max bi-directional Jaro-Winkler score over source and
    translation forms, mirroring :func:`_jw_any_match` but returning the
    magnitude instead of a bool. ``None`` if no comparable forms exist.
    """
    best: float | None = None

    def _track(value: float) -> None:
        nonlocal best
        if best is None or value > best:
            best = value

    # Source-form pairs.
    for sf_a in ea.surface_forms:
        if not sf_a.source:
            continue
        for sf_b in eb.surface_forms:
            if not sf_b.source:
                continue
            _track(string_similarity(sf_a.source, sf_b.source))

    # Translation pairs (canonical name + surface form translations).
    for tl_a in _all_translation_forms(ea):
        if not tl_a:
            continue
        for tl_b in _all_translation_forms(eb):
            if not tl_b:
                continue
            _track(string_similarity(tl_a, tl_b))

    return best


def _containment_direction_source(
    ea: GlossaryEntity, eb: GlossaryEntity
) -> str | None:
    """Return the entity_id whose source form CONTAINS the other's.

    ``None`` if neither strictly contains the other, or if both directions hold
    (ambiguous). Reuses the same ``len > 1`` skip rule as
    :func:`_has_source_substring_overlap`.
    """
    a_contains_b = False
    b_contains_a = False
    for sf_a in ea.surface_forms:
        if not sf_a.source:
            continue
        for sf_b in eb.surface_forms:
            if not sf_b.source:
                continue
            if len(sf_a.source) <= 1 and len(sf_b.source) <= 1:
                continue
            if len(sf_a.source) > 1 and sf_b.source != sf_a.source and sf_b.source in sf_a.source:
                a_contains_b = True
            if len(sf_b.source) > 1 and sf_a.source != sf_b.source and sf_a.source in sf_b.source:
                b_contains_a = True
    if a_contains_b and not b_contains_a:
        return ea.entity_id
    if b_contains_a and not a_contains_b:
        return eb.entity_id
    return None


def _containment_direction_translation(
    ea: GlossaryEntity, eb: GlossaryEntity
) -> str | None:
    """Return the entity_id whose translation form CONTAINS the other's.

    ``None`` if neither strictly contains the other, or if both directions hold
    (ambiguous). Mirrors :func:`_has_translation_containment` over
    :func:`_all_translation_forms`.
    """
    forms_a = [f.lower() for f in _all_translation_forms(ea) if f and len(f) > 1]
    forms_b = [f.lower() for f in _all_translation_forms(eb) if f and len(f) > 1]
    a_contains_b = False
    b_contains_a = False
    for la in forms_a:
        for lb in forms_b:
            if la == lb:
                continue
            if lb in la:
                a_contains_b = True
            if la in lb:
                b_contains_a = True
    if a_contains_b and not b_contains_a:
        return ea.entity_id
    if b_contains_a and not a_contains_b:
        return eb.entity_id
    return None


# ---------------------------------------------------------------------------
# Top-level candidate generation
# ---------------------------------------------------------------------------


def _add_embedding_candidates(
    glossary: Glossary,
    config: GlossaryClusterConfig,
    evidence: dict[CandidatePair, Evidence],
) -> None:
    """Add embedding-sourced candidate pairs and populate ``ev.cosine``.

    Mutates *evidence* in place:

    1. Compute embeddings + the full pairwise cosine matrix once.
    2. For every entity pair with cosine >= ``embedding_candidate_threshold``,
       add the pair and tag ``HEURISTIC_EMBEDDING``.
    3. For EVERY pair already in *evidence* (string- or embedding-sourced),
       populate ``ev.cosine`` from the matrix. This is the key step: the cosine
       score corroborates or contradicts string evidence during scoring, even
       for pairs that the string heuristics found.

    Candidate extraction is vectorized — ``cos`` is a numpy array, so the
    upper-triangle threshold is applied with numpy masking rather than a pure
    Python O(n^2) loop (which would be paid 3-4x per clustering invocation).
    """
    import numpy as np

    from dao_bridge.glossary_embeddings import (
        compute_entity_embeddings,
        cosine_matrix,
    )

    if len(glossary.entities) < 2:
        return

    entity_ids, emb = compute_entity_embeddings(glossary, config.embedding_model)
    cos = cosine_matrix(emb)
    index_of = {eid: i for i, eid in enumerate(entity_ids)}

    # New embedding-sourced candidates — vectorized upper-triangle threshold.
    iu, ju = np.triu_indices(len(entity_ids), k=1)
    mask = cos[iu, ju] >= config.embedding_candidate_threshold
    for i, j in zip(iu[mask], ju[mask], strict=False):
        pair = _ordered_pair(entity_ids[i], entity_ids[j])
        if pair[0] == pair[1]:
            continue
        evidence[pair].heuristics.add(HEURISTIC_EMBEDDING)

    # Populate cosine for ALL candidate pairs (string- or embedding-sourced).
    # Only pairs already in `evidence` are looked up, so this loop is over the
    # candidate set (small), not the full O(n^2) matrix.
    for (id_a, id_b), ev in evidence.items():
        ia, ib = index_of.get(id_a), index_of.get(id_b)
        if ia is not None and ib is not None:
            ev.cosine = float(cos[ia][ib])


def generate_cluster_candidates(
    glossary: Glossary,
    config: GlossaryClusterConfig,
) -> Candidates:
    """Run all heuristics and return candidate pairs WITH evidence.

    Each heuristic still returns a set of pairs; this function tags those pairs
    with the heuristic that produced them and accumulates them into per-pair
    :class:`Evidence`. No heuristic logic is duplicated here — we only annotate.

    Category is treated as a soft signal — cross-category pairs are allowed so
    that e.g. ``character`` vs ``title`` merges can be evaluated by the LLM.
    Same-category simply provides supporting evidence.

    When ``config.embedding_enabled`` is True, an additional embedding heuristic
    (Phase 2A) generates semantic candidate pairs the string heuristics cannot
    reach and populates ``Evidence.cosine`` for *every* candidate pair (string-
    or embedding-sourced) so the confidence scorer gains a corroborating signal.
    When embeddings are disabled, behavior is identical to Phase 1 and every
    ``Evidence.cosine`` stays ``None``.

    The return is a ``dict`` keyed by the same canonically ordered ``(a, b)``
    pairs as before, so existing consumers (``if not candidates``,
    ``sorted(candidates)``, iterating ``(a, b)`` pairs) are unchanged.
    """
    evidence: dict[CandidatePair, Evidence] = defaultdict(Evidence)
    entity_by_id = {e.entity_id: e for e in glossary.entities}

    def _tag(pairs: set[CandidatePair], name: str) -> None:
        for pair in pairs:
            if pair[0] == pair[1]:
                continue
            evidence[pair].heuristics.add(name)

    _tag(_source_substring_candidates(glossary), HEURISTIC_SOURCE_SUBSTRING)
    _tag(
        _translation_containment_candidates(glossary),
        HEURISTIC_TRANSLATION_CONTAINMENT,
    )
    _tag(_shared_reading_candidates(glossary), HEURISTIC_SHARED_READING)
    _tag(_alias_overlap_candidates(glossary), HEURISTIC_ALIAS_OVERLAP)
    _tag(
        _jw_similarity_candidates(glossary, threshold=config.jw_threshold),
        HEURISTIC_JW,
    )

    # Embedding heuristic (Phase 2A) — semantic candidate generation + cosine
    # corroboration. Only runs when explicitly enabled; degrades gracefully via
    # the lazy import in glossary_embeddings. Embeddings are recomputed every
    # iteration on purpose: after a merge the surviving entity's embedding text
    # changes, which is what surfaces transitive semantic merges (see the
    # "recompute-per-iteration is intentional" note in
    # build_phases/glossary-refactor-phase2.md). Do NOT hoist this out of the
    # per-iteration candidate generation.
    if config.embedding_enabled:
        _add_embedding_candidates(glossary, config, evidence)

    # Annotate same_category, JW magnitude, and containment direction for the
    # pairs we already have. This is annotation of an authoritative pair set,
    # not re-qualification.
    for (id_a, id_b), ev in evidence.items():
        ea = entity_by_id.get(id_a)
        eb = entity_by_id.get(id_b)
        if ea is None or eb is None:
            continue
        ev.same_category = ea.category == eb.category
        if HEURISTIC_JW in ev.heuristics:
            ev.jw_score = _best_jw_score(ea, eb)
        if HEURISTIC_SOURCE_SUBSTRING in ev.heuristics:
            ev.source_contains = _containment_direction_source(ea, eb)
        if HEURISTIC_TRANSLATION_CONTAINMENT in ev.heuristics:
            ev.translation_contains = _containment_direction_translation(ea, eb)

    logger.debug(
        "Cluster candidate generation produced %d pairs",
        len(evidence),
    )
    return dict(evidence)


# ---------------------------------------------------------------------------
# Confidence scoring + auto-merge canonical picker
# ---------------------------------------------------------------------------


def score_candidate_confidence(
    evidence: Evidence,
    config: GlossaryClusterConfig | None = None,
) -> ClusterConfidence:
    """Tier a candidate pair from its recorded evidence.

    Reads evidence only — performs no entity comparison, no recomputation.

    Two behavior modes, gated on whether embeddings are enabled:

    **Embeddings OFF (Phase 1 path).** When *config* is ``None`` or
    ``config.embedding_enabled`` is False, scoring is byte-for-byte Phase 1: HIGH
    requires "2+ strong string signals AND same category", everything else is
    MEDIUM, and LOW is never returned. ``Evidence.cosine`` is never consulted on
    this path. This is the backward-compatibility contract; every existing
    single-argument caller (and the Phase 1 unit tests) keeps working unchanged.

    **Embeddings ON (Phase 2A path).** ``Evidence.cosine`` corroborates or
    contradicts the string evidence:

    1. *Embedding-only weak pairs -> LOW (auto-reject).* If the only heuristic is
       ``HEURISTIC_EMBEDDING`` (no string signal) and
       ``cosine < embedding_low_confidence_max_cosine``, return LOW so the pair
       is dropped before partitioning without an LLM call.
    2. *HIGH requires embedding corroboration.* The "2+ strong signals AND same
       category" rule additionally requires
       ``cosine >= embedding_auto_merge_min_cosine``. This is the fix for the
       ``准仙帝``/``仙帝`` false merge: it has containment + high JW + same
       category (HIGH under the string-only scorer), but its cosine is depressed
       by the adjacent-realm summaries, so it drops to MEDIUM and goes to the
       LLM. A pair that would have been HIGH on strings but has ``cosine is
       None`` or low cosine is demoted to MEDIUM (not LOW — string evidence still
       merits LLM review).
    3. *Embedding as a strong signal.* A
       ``cosine >= embedding_auto_merge_min_cosine`` counts as one strong signal
       toward the "2+" tally, so a strong embedding + one strong string signal +
       same category can reach HIGH (subject to rule 2, satisfied by
       construction).
    4. Everything else -> MEDIUM.

    .. note::

        The embedding signal exists because the string-only scorer is **known to
        produce false auto-merges on real data** (it defaulted ``auto_merge_enabled``
        to False). The "2+ strong signals AND same category" rule cannot tell
        apart two cases that fire identical string evidence (source/translation
        containment + a high Jaro-Winkler score + shared category):

        - "qualifier means the SAME entity" — e.g. ``Dark Huo Ling'er`` is an
          aspect of ``Huo Ling'er`` (a correct merge), and
        - "qualifier means a DISTINCT rank/thing" — e.g. ``Quasi-Immortal
          Emperor`` (准仙帝) is a cultivation realm *just below* ``Immortal
          Emperor`` (仙帝), a wrong merge.

        A live run on perfect-world-cn produced 4 auto-merges, 2 of them wrong
        (the Quasi-/Immortal Emperor case above, and ``Huang`` vs ``Da Zhuang``
        on a coincidental JW match). See the addendum in
        ``build_phases/glossary-cluster-evidence-and-auto-merge.md``. Embedding
        distance (``Evidence.cosine``) distinguishes the adjacent-realm and
        unrelated-character cases, which is why ``auto_merge_enabled=True`` is
        only production-safe alongside ``embedding_enabled=True``.
    """
    embedding_enabled = config is not None and config.embedding_enabled

    strong = {
        HEURISTIC_SOURCE_SUBSTRING,
        HEURISTIC_TRANSLATION_CONTAINMENT,
        HEURISTIC_SHARED_READING,
        HEURISTIC_ALIAS_OVERLAP,
    }
    strong_count = len(evidence.heuristics & strong)

    # A high-magnitude JW counts as a strong signal; the candidate-gen threshold
    # (0.75) is intentionally looser, so JW alone at the candidate level is NOT
    # strong.
    high_jw = evidence.jw_score is not None and evidence.jw_score >= _AUTO_MERGE_JW_STRONG
    if high_jw:
        strong_count += 1

    # Rule 0 — embeddings disabled: Phase 1 path. Never consults cosine, never
    # returns LOW. Byte-for-byte Phase 1 behaviour.
    if not embedding_enabled:
        if strong_count >= 2 and evidence.same_category:
            return ClusterConfidence.HIGH
        return ClusterConfidence.MEDIUM

    # --- Embeddings ON below. All cosine comparisons are guarded by both
    # `embedding_enabled` (true here) AND `ev.cosine is not None` to avoid a
    # `None >= float` TypeError. ---

    # Rule 1 — embedding-only weak pairs -> LOW (auto-reject).
    only_embedding = evidence.heuristics == {HEURISTIC_EMBEDDING}
    if (
        only_embedding
        and evidence.cosine is not None
        and evidence.cosine < config.embedding_low_confidence_max_cosine
    ):
        return ClusterConfidence.LOW

    # Rule 3 — a strong embedding (cosine at/above the auto-merge floor) counts
    # as one strong signal toward the "2+" tally, alongside the string signals.
    # Cosine is populated for every candidate pair, so this applies whether the
    # pair was embedding-sourced or string-sourced.
    strong_cosine = (
        evidence.cosine is not None
        and evidence.cosine >= config.embedding_auto_merge_min_cosine
    )
    if strong_cosine:
        strong_count += 1

    # Rule 2 — HIGH requires 2+ strong signals, same category, AND embedding
    # corroboration (cosine at/above the auto-merge floor). Otherwise demote to
    # MEDIUM (string evidence still merits LLM review — never LOW here).
    if strong_count >= 2 and evidence.same_category and strong_cosine:
        return ClusterConfidence.HIGH

    # Rule 4 — everything else.
    return ClusterConfidence.MEDIUM


def pick_canonical_for_auto_merge(
    ea: GlossaryEntity,
    eb: GlossaryEntity,
    evidence: Evidence,
) -> tuple[GlossaryEntity, GlossaryEntity, str]:
    """Choose (winner, loser, preferred_canonical_name) for a HIGH-confidence merge.

    Priority:
      1. Containment direction, if known — the container (more specific) name wins.
      2. Longer canonical_name (tends to be more specific).
      3. Earlier first_seen_chunk.
      4. Stable fallback (ea).
    """
    # 1. Prefer the container side when direction is known.
    container_id = evidence.translation_contains or evidence.source_contains
    if container_id == ea.entity_id:
        return ea, eb, ea.canonical_name
    if container_id == eb.entity_id:
        return eb, ea, eb.canonical_name

    # 2. Longer canonical name.
    if len(ea.canonical_name) > len(eb.canonical_name):
        return ea, eb, ea.canonical_name
    if len(eb.canonical_name) > len(ea.canonical_name):
        return eb, ea, eb.canonical_name

    # 3. Earlier first_seen_chunk.
    if ea.first_seen_chunk and eb.first_seen_chunk:
        if ea.first_seen_chunk <= eb.first_seen_chunk:
            return ea, eb, ea.canonical_name
        return eb, ea, eb.canonical_name

    # 4. Stable fallback.
    return ea, eb, ea.canonical_name


# ---------------------------------------------------------------------------
# Entity rendering for cluster prompt
# ---------------------------------------------------------------------------


def render_entity_for_cluster_prompt(entity: GlossaryEntity) -> str:
    """Produce a compact text representation of *entity* for the LLM confirmation prompt."""
    lines: list[str] = []
    lines.append(f"Entity ID: {entity.entity_id}")
    lines.append(f"Category: {entity.category}")
    lines.append(f"Canonical name: {entity.canonical_name}")
    if entity.summary:
        lines.append(f"Summary: {entity.summary}")
    if entity.surface_forms:
        lines.append("Surface forms:")
        for sf in entity.surface_forms:
            parts = f"  - {sf.source} -> {sf.translation}"
            if sf.reading:
                parts += f" (reading: {sf.reading})"
            if sf.context_hints:
                parts += f" [hints: {'; '.join(sf.context_hints)}]"
            lines.append(parts)
    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")
    if entity.nicknames:
        nick_parts = [f"{speaker} -> {nick}" for speaker, nick in entity.nicknames.items()]
        lines.append(f"Nicknames: {'; '.join(nick_parts)}")
    if entity.speech_style:
        lines.append(f"Speech style: {entity.speech_style}")
    if entity.notes:
        lines.append(f"Notes: {entity.notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity merge
# ---------------------------------------------------------------------------


def merge_entities(
    winner: GlossaryEntity,
    loser: GlossaryEntity,
    preferred_canonical_name: str | None = None,
) -> None:
    """Merge *loser* into *winner* in place.

    After this call the caller should remove *loser* from the glossary.

    Parameters
    ----------
    winner:
        The surviving entity (mutated in place).
    loser:
        The entity being absorbed.
    preferred_canonical_name:
        If provided, override *winner*'s ``canonical_name``.  Typically
        comes from the LLM clustering decision.
    """
    # Canonical name.
    if preferred_canonical_name:
        winner.canonical_name = preferred_canonical_name

    # Union surface forms — dedup by normalised source.
    _merge_surface_forms(winner, loser)

    # Union aliases.
    alias_set = set(winner.aliases)
    for alias in loser.aliases:
        if alias not in alias_set:
            winner.aliases.append(alias)
            alias_set.add(alias)

    # Merge nicknames (winner wins on key conflict).
    for speaker, nick in loser.nicknames.items():
        if speaker not in winner.nicknames:
            winner.nicknames[speaker] = nick

    # Accumulate speech_style observations.
    if loser.speech_style:
        if winner.speech_style:
            existing = winner.speech_style.split(_SPEECH_STYLE_DELIMITER)
            for obs in loser.speech_style.split(_SPEECH_STYLE_DELIMITER):
                if obs.strip() and obs.strip() not in [e.strip() for e in existing]:
                    winner.speech_style = (
                        winner.speech_style + _SPEECH_STYLE_DELIMITER + obs.strip()
                    )
        else:
            winner.speech_style = loser.speech_style

    # Merge notes conservatively.
    if loser.notes:
        if winner.notes:
            if loser.notes not in winner.notes:
                winner.notes = winner.notes + " " + loser.notes
        else:
            winner.notes = loser.notes

    # Merge summary conservatively.
    if loser.summary:
        if winner.summary:
            if loser.summary not in winner.summary:
                merged = winner.summary + " " + loser.summary
                if len(merged) > _MAX_SUMMARY_LENGTH:
                    merged = merged[:_MAX_SUMMARY_LENGTH].rsplit(" ", 1)[0] + "..."
                winner.summary = merged
        else:
            winner.summary = loser.summary

    # Merge source_books.
    books_set = set(winner.source_books)
    for book in loser.source_books:
        if book not in books_set:
            winner.source_books.append(book)
            books_set.add(book)

    # Temporal tracking — keep earliest first_seen, latest latest_evidence.
    if loser.first_seen_chunk and (
        not winner.first_seen_chunk or loser.first_seen_chunk < winner.first_seen_chunk
    ):
        winner.first_seen_chunk = loser.first_seen_chunk

    if loser.latest_evidence_chunk and (
        not winner.latest_evidence_chunk
        or loser.latest_evidence_chunk > winner.latest_evidence_chunk
    ):
        winner.latest_evidence_chunk = loser.latest_evidence_chunk


def _merge_surface_forms(winner: GlossaryEntity, loser: GlossaryEntity) -> None:
    """Union surface forms from *loser* into *winner*, deduplicating by source.

    When two forms share the same ``source`` string but differ in translation,
    the winner's form is kept but the alternate translation is preserved as
    a variant so the information is not silently lost.
    """
    existing_sources: dict[str, SurfaceForm] = {}
    for sf in winner.surface_forms:
        existing_sources[sf.source] = sf

    for sf_loser in loser.surface_forms:
        existing = existing_sources.get(sf_loser.source)
        if existing is not None:
            # Same source — merge metadata into the existing form.
            existing.occurrence_count += sf_loser.occurrence_count

            # Backfill reading.
            if not existing.reading and sf_loser.reading:
                existing.reading = sf_loser.reading

            # Union context hints.
            for hint in sf_loser.context_hints:
                if hint and hint not in existing.context_hints:
                    existing.context_hints.append(hint)

            # If translation differs, preserve the alternate as a proper variant
            # so reconcile can inspect and resolve translation conflicts.
            if sf_loser.translation != existing.translation:
                if sf_loser.translation not in existing.translation_variants:
                    existing.translation_variants.append(sf_loser.translation)

            # Union any existing translation_variants from the loser.
            for variant in sf_loser.translation_variants:
                if variant not in existing.translation_variants and variant != existing.translation:
                    existing.translation_variants.append(variant)

            # Merge notes.
            if sf_loser.notes:
                if existing.notes:
                    if sf_loser.notes not in existing.notes:
                        existing.notes = existing.notes + " " + sf_loser.notes
                else:
                    existing.notes = sf_loser.notes

            # Keep earlier first_seen_chunk.
            if sf_loser.first_seen_chunk and (
                not existing.first_seen_chunk
                or sf_loser.first_seen_chunk < existing.first_seen_chunk
            ):
                existing.first_seen_chunk = sf_loser.first_seen_chunk
        else:
            # New surface form — add directly.
            winner.surface_forms.append(sf_loser.model_copy(deep=True))
            existing_sources[sf_loser.source] = winner.surface_forms[-1]


# ---------------------------------------------------------------------------
# ID remapping helper
# ---------------------------------------------------------------------------


def remap_entity_id(
    decisions: list[tuple[str, str, str | None]],
    id_map: dict[str, str],
) -> list[tuple[str, str, str | None]]:
    """Resolve entity IDs through a merge-remapping table.

    After merging B into A, any later decision referencing B should be
    rewritten to reference A.  If both IDs in a decision resolve to the
    same entity, the decision is dropped (self-merge).

    Parameters
    ----------
    decisions:
        List of ``(entity_id_a, entity_id_b, preferred_canonical_name)``
        tuples from LLM decisions.
    id_map:
        Mapping from *absorbed* entity IDs to their surviving counterpart.

    Returns
    -------
    list
        Filtered and remapped decisions.
    """
    result: list[tuple[str, str, str | None]] = []
    for eid_a, eid_b, pref_name in decisions:
        resolved_a = _resolve_id(eid_a, id_map)
        resolved_b = _resolve_id(eid_b, id_map)
        if resolved_a == resolved_b:
            # Both resolve to same entity — skip (already merged).
            continue
        result.append((resolved_a, resolved_b, pref_name))
    return result


def _resolve_id(entity_id: str, id_map: dict[str, str]) -> str:
    """Follow the remap chain to the final surviving entity ID."""
    visited: set[str] = set()
    current = entity_id
    while current in id_map and current not in visited:
        visited.add(current)
        current = id_map[current]
    return current


# ---------------------------------------------------------------------------
# Clustering report
# ---------------------------------------------------------------------------


def write_cluster_report(
    report_path,
    merge_log: list[dict],
    total_iterations: int,
    total_candidates_evaluated: int,
) -> None:
    """Write a human-readable clustering report as markdown.

    Parameters
    ----------
    report_path:
        Destination path (typically ``<work_dir>/glossary_cluster_report.md``).
    merge_log:
        List of dicts recording each merge that was performed.
    total_iterations:
        Number of candidate-generation iterations run.
    total_candidates_evaluated:
        Total number of candidate pairs sent to the LLM.
    """
    from dao_bridge.workdir import atomic_write

    auto_count = sum(1 for e in merge_log if e.get("auto_merged"))
    llm_count = len(merge_log) - auto_count

    lines = ["# Glossary Clustering Report", ""]
    lines.append(f"- Iterations: {total_iterations}")
    lines.append(f"- Total candidate pairs evaluated: {total_candidates_evaluated}")
    lines.append(f"- Auto-merges (high confidence): {auto_count}")
    lines.append(f"- LLM-confirmed merges: {llm_count}")
    lines.append(f"- Merges performed: {len(merge_log)}")
    lines.append("")

    if not merge_log:
        lines.append("No duplicate entities found.")
    else:
        lines.append("## Merges")
        lines.append("")
        for entry in merge_log:
            merge_type = "auto-merge" if entry.get("auto_merged") else "LLM-confirmed"
            lines.append(f"### {entry['winner_name']} <- {entry['loser_name']}")
            lines.append(f"- **Type:** {merge_type}")
            lines.append(f"- **Winner:** `{entry['winner_id']}`")
            lines.append(f"- **Absorbed:** `{entry['loser_id']}`")
            lines.append(f"- **Result canonical name:** {entry['result_name']}")
            lines.append(f"- **Reasoning:** {entry['reasoning']}")
            if entry.get("surface_forms_added"):
                lines.append("- **Surface forms added:**")
                for sf_str in entry["surface_forms_added"]:
                    lines.append(f"  - {sf_str}")
            lines.append("")

    atomic_write(report_path, "\n".join(lines))
