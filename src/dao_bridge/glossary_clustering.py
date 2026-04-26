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
# Top-level candidate generation
# ---------------------------------------------------------------------------


def generate_cluster_candidates(
    glossary: Glossary,
    config: GlossaryClusterConfig,
) -> set[CandidatePair]:
    """Run all heuristics and return deduplicated candidate pairs.

    Category is treated as a soft signal — cross-category pairs are
    allowed so that e.g. ``character`` vs ``title`` merges can be
    evaluated by the LLM.  Same-category simply provides supporting
    evidence.
    """
    candidates: set[CandidatePair] = set()
    candidates |= _source_substring_candidates(glossary)
    candidates |= _translation_containment_candidates(glossary)
    candidates |= _shared_reading_candidates(glossary)
    candidates |= _alias_overlap_candidates(glossary)
    candidates |= _jw_similarity_candidates(glossary, threshold=config.jw_threshold)

    # Remove self-pairs (should not happen, but be safe).
    candidates = {(a, b) for a, b in candidates if a != b}

    logger.debug(
        "Cluster candidate generation produced %d pairs",
        len(candidates),
    )
    return candidates


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

    lines = ["# Glossary Clustering Report", ""]
    lines.append(f"- Iterations: {total_iterations}")
    lines.append(f"- Total candidate pairs evaluated: {total_candidates_evaluated}")
    lines.append(f"- Merges performed: {len(merge_log)}")
    lines.append("")

    if not merge_log:
        lines.append("No duplicate entities found.")
    else:
        lines.append("## Merges")
        lines.append("")
        for entry in merge_log:
            lines.append(f"### {entry['winner_name']} <- {entry['loser_name']}")
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
