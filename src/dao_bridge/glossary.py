"""Glossary extraction, reconciliation, and export (entity-centric v2).

Four-stage glossary pipeline:

1. **Build** — extracts mentions (proper nouns, characters, places, etc.)
   from chunked source text via batched LLM calls.  Links each mention to
   an existing entity or creates a new one.  Accumulates a per-book
   glossary, saving after each batch for resumability.
2. **Cluster** — finds duplicate entities that build-time linking
   missed and merges them with LLM confirmation.
3. **Reconcile** — resolves within-book conflicts (differing English
   proposals, corrections, category mismatches) and consolidates multiple
   speech-style observations per character entity.
4. **Export** — renders the glossary as human-readable markdown for review.

The ``source`` field on each :class:`~dao_bridge.schemas.GlossaryEntity`
distinguishes provenance:

- ``"extracted"`` — found by the LLM during build.
- ``"user"`` — manually added or edited by a human; never modified.
- ``"seed"`` / ``"master"`` — reserved for future master-glossary features.
"""

from __future__ import annotations

import functools
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from dao_bridge.chunk import count_tokens
from dao_bridge.config import AppConfig, resolve_language_name
from dao_bridge.llm_client import LLMClient, LLMStructuredOutputError
from dao_bridge.schemas import (
    Chunk,
    ExtractedMention,
    Glossary,
    GlossaryEntity,
    GlossaryExtractionResponse,
    GlossaryReconcileResponse,
    GlossarySpeechMergeResponse,
    SurfaceForm,
)
from dao_bridge.similarity import string_similarity
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    iter_pending_items,
    mark_item_completed,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
    reopen_stage,
    reset_stage,
    reset_stage_items,
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

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parent / "prompts"
_BUILD_META_FILENAME = "_glossary_build_meta.json"
_CLUSTER_META_FILENAME = "_glossary_cluster_meta.json"

# Delimiter used to accumulate multiple speech-style observations before
# the reconcile stage consolidates them.
_SPEECH_STYLE_DELIMITER = "\n"

# Maximum character length for entity summaries before truncation.
_MAX_SUMMARY_LENGTH = 500

# Auto-attach threshold for Jaro-Winkler string similarity.
_SIMILARITY_AUTO_ATTACH = 0.95

# PR 1 intentionally does not inject the accumulated glossary into
# extraction prompts; build-time linking and later clustering handle
# deduplication instead.
_EXTRACTION_GLOSSARY_PLACEHOLDER = "(not provided in phase 1)"

# ---------------------------------------------------------------------------
# Build metadata sidecar (internal, not part of the public glossary format)
# ---------------------------------------------------------------------------


class _ConflictRecord(BaseModel):
    """A single term conflict detected during the build stage."""

    entity_id: str
    source_form: str
    reading: str | None = None
    current_english: str
    alternatives: list[dict] = Field(default_factory=list)
    # Each alternative: {"english": str, "context_snippet": str, "batch_id": str}
    category_variants: list[str] = Field(default_factory=list)


class _BuildMeta(BaseModel):
    """Internal sidecar persisted alongside ``glossary.json`` during build.

    Stores conflict data and batch progress so that reconcile can consume
    the conflicts and build can resume from a crash.
    """

    conflicts: list[_ConflictRecord] = Field(default_factory=list)
    corrections: list[dict] = Field(default_factory=list)
    processed_batches: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cluster metadata sidecar
# ---------------------------------------------------------------------------


class _ClusterMeta(BaseModel):
    """Internal sidecar persisted during the clustering stage.

    Tracks which iterations completed and accumulates the merge log so
    that resumed runs can skip completed iterations and the final report
    includes all merges across runs.
    """

    completed_iterations: list[int] = Field(default_factory=list)
    merge_log: list[dict] = Field(default_factory=list)
    total_candidates_evaluated: int = 0


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the ``prompts/`` directory.

    Cached — template files are read once per process.

    Parameters
    ----------
    name:
        Template filename (e.g. ``"glossary_extract.txt"``).
    """
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------


def validate_glossary_categories(glossary: Glossary, categories: list[str]) -> None:
    """Validate that every entity's category is in the allowed list.

    Raises
    ------
    ValueError
        With a clear message listing each invalid category and the entities
        that use it.
    """
    valid = set(categories)
    invalid: dict[str, list[str]] = defaultdict(list)
    for entity in glossary.entities:
        if entity.category not in valid:
            label = entity.canonical_english or entity.entity_id
            invalid[entity.category].append(label)
    if invalid:
        lines = []
        for cat, entities in sorted(invalid.items()):
            lines.append(f"  '{cat}': used by {', '.join(entities)}")
        raise ValueError(
            f"Invalid glossary categories found (allowed: {', '.join(categories)}):\n"
            + "\n".join(lines)
        )


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def _load_all_chunks(work_dir: Path, manifest) -> list[Chunk]:
    """Load all chunks from disk in spine + chunk order.

    Iterates all spine items that have ``chunk_count > 0``, loads each
    chunk JSON file, and returns a flat list sorted by spine index then
    chunk index.
    """
    chunks: list[Chunk] = []
    sw = manifest.spine_padding_width

    for item in manifest.spine:
        if not item.chunk_count or item.chunk_count == 0:
            continue
        cd = chunk_dir(work_dir, item.spine_index, sw)
        if not cd.exists():
            continue
        # Collect and sort chunk files for this spine item.
        chunk_files = sorted(cd.glob("*.json"))
        for cf in chunk_files:
            data = json.loads(cf.read_text(encoding="utf-8"))
            chunks.append(Chunk(**data))

    # Ensure ordering by (spine_index, chunk_index).
    chunks.sort(key=lambda c: (c.spine_index, c.chunk_index))
    return chunks


# ---------------------------------------------------------------------------
# Progress callback dataclass
# ---------------------------------------------------------------------------


@dataclass
class GlossaryBuildProgress:
    """Passed to the *on_progress* callback after each sub-batch completes."""

    item_id: str
    """Spine-aligned item ID, e.g. ``"0003.b2"``."""
    spine_batch_count: int
    """Total sub-batches for this spine item (e.g. 5)."""
    items_total: int
    """Total work items across all spines."""


@dataclass(frozen=True)
class _GlossaryBatch:
    """A deterministic glossary extraction batch covering contiguous chunks.

    Each batch maps to a single LLM call during the build stage.  The
    *item_id* is the state-tracking key (e.g. ``"0003.b2"``).
    """

    item_id: str
    """State-tracking key, e.g. ``"0003.b2"``."""
    spine_index: int
    chunks: tuple[Chunk, ...]
    """Immutable tuple of chunks — never split, never reordered."""
    spine_batch_count: int
    """Total sub-batches for this spine item."""

    @property
    def token_count(self) -> int:
        """Total tokens across all chunks in this batch."""
        return sum(c.token_count for c in self.chunks)

    @property
    def chunk_range_label(self) -> str:
        """Human-readable chunk range for logging, e.g. ``'0003.004-0003.006'``."""
        start = self.chunks[0].chunk_id
        end = self.chunks[-1].chunk_id
        return start if start == end else f"{start}-{end}"


# ---------------------------------------------------------------------------
# Spine grouping and batch packing
# ---------------------------------------------------------------------------


def _group_chunks_by_spine(chunks: list[Chunk]) -> list[tuple[int, list[Chunk]]]:
    """Group a flat chunk list by spine index, preserving order.

    Returns a list of ``(spine_index, chunks)`` tuples, ordered by spine
    index.  Each spine's chunks are in chunk-index order.
    """
    by_spine: dict[int, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_spine[chunk.spine_index].append(chunk)

    # Sort by spine index; chunks within each spine are already sorted
    # because the input is sorted by (spine_index, chunk_index).
    return sorted(by_spine.items())


def _rebalance_final_two_batches(
    previous: list[Chunk],
    last: list[Chunk],
) -> tuple[list[Chunk], list[Chunk]]:
    """Re-split the final two batches at the most even chunk boundary.

    Merges *previous* and *last* into a single list, then evaluates
    every possible split point using a prefix-sum to find the one that
    minimises the absolute token difference between the two halves.

    On ties, the earlier split point is preferred so the first batch
    stays smaller.

    Returns
    -------
    tuple[list[Chunk], list[Chunk]]
        The rebalanced (previous, last) pair.
    """
    combined = previous + last
    if len(combined) < 2:
        return previous, last

    # Build prefix sums.
    prefix_tokens: list[int] = []
    running = 0
    for chunk in combined:
        running += chunk.token_count
        prefix_tokens.append(running)

    total_tokens = prefix_tokens[-1]

    # Evaluate every valid split point (1 .. len-1).
    best_split = 1
    best_delta = abs(prefix_tokens[0] - (total_tokens - prefix_tokens[0]))

    for split_idx in range(2, len(combined)):
        left_tokens = prefix_tokens[split_idx - 1]
        right_tokens = total_tokens - left_tokens
        delta = abs(left_tokens - right_tokens)
        if delta < best_delta:
            best_split = split_idx
            best_delta = delta

    return combined[:best_split], combined[best_split:]


def _pack_spine_batches(
    spine_chunks: list[Chunk],
    target_tokens: int,
    min_batch_tokens: int,
    redistribute_threshold: float,
) -> list[list[Chunk]]:
    """Pack chunks for a single spine item into sub-batches.

    Greedy-packs chunks up to *target_tokens*, then applies remainder
    balancing to avoid runt final batches:

    - If the final sub-batch has fewer than *min_batch_tokens* tokens,
      absorb it into the previous sub-batch.
    - If the final sub-batch has fewer than
      ``target_tokens * redistribute_threshold`` tokens (but above
      *min_batch_tokens*), redistribute the last two sub-batches evenly
      via :func:`_rebalance_final_two_batches`.

    Parameters
    ----------
    spine_chunks:
        Chunks for a single spine item, in chunk-index order.
    target_tokens:
        Maximum token count per sub-batch.
    min_batch_tokens:
        Threshold below which the final sub-batch is absorbed into the
        previous one.
    redistribute_threshold:
        Fraction of *target_tokens*.  If the final sub-batch is below
        this threshold (but above *min_batch_tokens*), the last two
        sub-batches are redistributed evenly.

    Returns
    -------
    list[list[Chunk]]
        List of sub-batches, each a list of whole chunks.
    """
    if not spine_chunks:
        return []

    # --- Greedy packing ---
    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_tokens = 0

    for chunk in spine_chunks:
        if current_batch and current_tokens + chunk.token_count > target_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += chunk.token_count

    if current_batch:
        batches.append(current_batch)

    # --- Remainder balancing ---
    if len(batches) >= 2:
        last_tokens = sum(c.token_count for c in batches[-1])
        threshold_tokens = target_tokens * redistribute_threshold

        if last_tokens < min_batch_tokens:
            # Absorb the final sub-batch into the previous one.
            batches[-2].extend(batches[-1])
            batches.pop()
        elif last_tokens < threshold_tokens:
            batches[-2], batches[-1] = _rebalance_final_two_batches(batches[-2], batches[-1])

    return batches


# ---------------------------------------------------------------------------
# Entity ID generation
# ---------------------------------------------------------------------------


def next_entity_id(category: str, glossary: Glossary) -> str:
    """Generate the next sequential entity ID for *category*.

    Format: ``"{category}_{NNNNNN}"`` where ``NNNNNN`` is zero-padded to 6
    digits.  Scans existing entities to find the highest existing number
    for the given category prefix.

    Parameters
    ----------
    category:
        Entity category, e.g. ``"character"``.
    glossary:
        Current glossary (used to find highest existing ID).

    Returns
    -------
    str
        New entity ID, e.g. ``"character_000001"``.
    """
    prefix = f"{category}_"
    max_num = 0
    for entity in glossary.entities:
        if entity.entity_id.startswith(prefix):
            try:
                num = int(entity.entity_id[len(prefix) :])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    return f"{prefix}{max_num + 1:06d}"


# ---------------------------------------------------------------------------
# Build-time entity linking
# ---------------------------------------------------------------------------


def find_entity_for_mention(
    glossary: Glossary,
    mention: ExtractedMention,
) -> GlossaryEntity | None:
    """Find an existing entity that a mention should attach to.

    Implements the candidate retrieval order from the spec:

    1. Exact surface-form source match (highest confidence)
    2. Same non-null reading AND same proposed English
    3. Jaro-Winkler >= 0.95 on source AND same category

    Returns ``None`` if no safe match is found — the caller should
    create a new entity.

    Parameters
    ----------
    glossary:
        Current glossary.
    mention:
        The extracted mention to link.

    Returns
    -------
    GlossaryEntity | None
        The matched entity, or ``None`` if no safe match.
    """
    # 1. Exact surface-form source match.
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.source == mention.source:
                return entity

    # 2. Same non-null reading AND same proposed English.
    if mention.reading:
        for entity in glossary.entities:
            for sf in entity.surface_forms:
                if sf.reading and sf.reading == mention.reading and sf.english == mention.english:
                    return entity

    # 3. High Jaro-Winkler on source + same category — but only if
    #    the match is unambiguous (exactly one candidate).
    candidates: list[GlossaryEntity] = []
    for entity in glossary.entities:
        if entity.category != mention.category:
            continue
        for sf in entity.surface_forms:
            if string_similarity(sf.source, mention.source) >= _SIMILARITY_AUTO_ATTACH:
                candidates.append(entity)
                break  # One matching form per entity is enough

    if len(candidates) == 1:
        return candidates[0]

    # 0 candidates: no match.  2+ candidates: ambiguous — create a new
    # entity and let clustering resolve it later.
    return None


def _find_entity_by_id(glossary: Glossary, entity_id: str) -> GlossaryEntity | None:
    """Find an entity by its entity_id."""
    for entity in glossary.entities:
        if entity.entity_id == entity_id:
            return entity
    return None


def _find_entity_by_canonical_english(
    glossary: Glossary, canonical_english: str
) -> GlossaryEntity | None:
    """Find an entity by canonical English name (case-sensitive)."""
    for entity in glossary.entities:
        if entity.canonical_english == canonical_english:
            return entity
    return None


def _find_entity_for_correction(
    glossary: Glossary,
    existing_english: str,
    source_form: str,
) -> GlossaryEntity | None:
    """Find the most likely correction target entity.

    Prefer canonical-English matches for the common case, but fall back
    to matching the correction's source form against surface forms so we
    do not depend on canonical English being unique or unchanged.

    Both signals require a *unique* match — if multiple entities share
    the same canonical English or the same surface form source, the
    signal is ambiguous and we skip it rather than picking the first.
    """
    english_matches = [
        entity for entity in glossary.entities if entity.canonical_english == existing_english
    ]
    if len(english_matches) == 1:
        return english_matches[0]

    source_matches: list[GlossaryEntity] = []
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.source == source_form:
                source_matches.append(entity)
                break

    if len(source_matches) == 1:
        return source_matches[0]

    return None


# ---------------------------------------------------------------------------
# Surface form and entity merge helpers
# ---------------------------------------------------------------------------


def add_or_update_surface_form(
    entity: GlossaryEntity,
    mention: ExtractedMention,
    chunk_id: str,
) -> None:
    """Add a new surface form or update an existing one on *entity*.

    If a surface form with the same ``source`` already exists, increments
    its ``occurrence_count`` and appends any new context hint.  Otherwise,
    creates a new :class:`SurfaceForm`.

    Parameters
    ----------
    entity:
        The entity to update.
    mention:
        The extracted mention providing the surface form data.
    chunk_id:
        ID of the chunk where the mention was found.
    """
    for sf in entity.surface_forms:
        if sf.source == mention.source:
            sf.occurrence_count += 1
            if mention.english != sf.english and mention.english not in sf.english_variants:
                sf.english_variants.append(mention.english)
            # Backfill reading.
            if not sf.reading and mention.reading:
                sf.reading = mention.reading
            # Append context hint if new.
            if mention.context_hint and mention.context_hint not in sf.context_hints:
                sf.context_hints.append(mention.context_hint)
            # Merge notes.
            if mention.notes and mention.notes != sf.notes:
                if sf.notes:
                    if mention.notes not in sf.notes:
                        sf.notes = sf.notes + " " + mention.notes
                else:
                    sf.notes = mention.notes
            return

    # New surface form.
    entity.surface_forms.append(
        SurfaceForm(
            source=mention.source,
            reading=mention.reading,
            english=mention.english,
            context_hints=[mention.context_hint] if mention.context_hint else [],
            notes=mention.notes,
            first_seen_chunk=chunk_id,
            occurrence_count=1,
        )
    )


def merge_entity_summary(
    entity: GlossaryEntity,
    summary_update: str | None,
    chunk_id: str,
) -> None:
    """Merge a summary observation into an entity.

    Simple concatenation with deduplication and max-length truncation.

    Parameters
    ----------
    entity:
        The entity to update.
    summary_update:
        New summary observation, or ``None``.
    chunk_id:
        ID of the chunk providing the observation.
    """
    if not summary_update:
        return
    entity.latest_evidence_chunk = chunk_id
    if not entity.summary:
        entity.summary = summary_update
    elif summary_update not in entity.summary:
        merged = entity.summary + " " + summary_update
        if len(merged) > _MAX_SUMMARY_LENGTH:
            merged = merged[:_MAX_SUMMARY_LENGTH].rsplit(" ", 1)[0] + "..."
        entity.summary = merged


def merge_aliases_nicknames_speech_notes(
    entity: GlossaryEntity,
    mention: ExtractedMention,
) -> None:
    """Merge aliases, nicknames, speech_style, and notes from a mention.

    Parameters
    ----------
    entity:
        The entity to update.
    mention:
        The extracted mention providing the data.
    """
    # Union aliases.
    alias_set = set(entity.aliases)
    for alias in mention.aliases:
        if alias not in alias_set:
            entity.aliases.append(alias)
            alias_set.add(alias)

    # Merge nicknames (existing wins on key conflict).
    for speaker, nick in mention.nicknames.items():
        if speaker not in entity.nicknames:
            entity.nicknames[speaker] = nick

    # Accumulate speech_style observations.
    if mention.speech_style:
        if entity.speech_style:
            existing_observations = entity.speech_style.split(_SPEECH_STYLE_DELIMITER)
            if mention.speech_style not in existing_observations:
                entity.speech_style = (
                    entity.speech_style + _SPEECH_STYLE_DELIMITER + mention.speech_style
                )
        else:
            entity.speech_style = mention.speech_style

    # Concatenate notes if new info.
    if mention.notes and mention.notes != entity.notes:
        if entity.notes:
            if mention.notes not in entity.notes:
                entity.notes = entity.notes + " " + mention.notes
        else:
            entity.notes = mention.notes


# ---------------------------------------------------------------------------
# Glossary rendering (for prompt injection during build)
# ---------------------------------------------------------------------------


def _render_existing_glossary(glossary: Glossary, max_tokens: int) -> str:
    """Render the current glossary compactly for prompt injection.

    Groups entities by category, one line per surface form.  If the total
    exceeds *max_tokens*, truncates to the most recently added entities
    with a note.
    """
    if not glossary.entities:
        return "(no entries yet)"

    # Group by category.
    by_category: dict[str, list[GlossaryEntity]] = defaultdict(list)
    for entity in glossary.entities:
        by_category[entity.category].append(entity)

    lines: list[str] = []
    for cat in sorted(by_category.keys()):
        lines.append(f"[{cat}]")
        for entity in by_category[cat]:
            for sf in entity.surface_forms:
                parts = [f"{sf.source} -> {sf.english}"]
                if sf.reading:
                    parts.append(f"reading: {sf.reading}")
                lines.append("  " + " | ".join(parts))

    rendered = "\n".join(lines)

    # Check token count and truncate if needed.
    token_count = count_tokens(rendered)
    if token_count > max_tokens:
        # Truncate: keep the tail (most recently added) entities.
        all_forms: list[tuple[str, SurfaceForm]] = []
        for entity in glossary.entities:
            for sf in entity.surface_forms:
                all_forms.append((entity.category, sf))

        truncated_forms = list(reversed(all_forms))
        kept: list[tuple[str, SurfaceForm]] = []
        running_tokens = 0
        for cat, sf in truncated_forms:
            line = f"  {sf.source} -> {sf.english}"
            line_tokens = count_tokens(line)
            if running_tokens + line_tokens > max_tokens - 50:
                break
            kept.append((cat, sf))
            running_tokens += line_tokens

        kept.reverse()
        by_cat_truncated: dict[str, list[SurfaceForm]] = defaultdict(list)
        for cat, sf in kept:
            by_cat_truncated[cat].append(sf)

        total_forms = sum(len(e.surface_forms) for e in glossary.entities)
        trunc_lines = [f"(truncated — showing {len(kept)} of {total_forms} forms, most recent)"]
        for cat in sorted(by_cat_truncated.keys()):
            trunc_lines.append(f"[{cat}]")
            for sf in by_cat_truncated[cat]:
                parts = [f"{sf.source} -> {sf.english}"]
                if sf.reading:
                    parts.append(f"reading: {sf.reading}")
                trunc_lines.append("  " + " | ".join(parts))

        rendered = "\n".join(trunc_lines)

    return rendered


# ---------------------------------------------------------------------------
# Stage invalidation cascade
# ---------------------------------------------------------------------------

# Pipeline ordering: glossary_build -> glossary_cluster -> glossary_reconcile
_DOWNSTREAM_STAGES: dict[str, list[str]] = {
    "glossary_build": ["glossary_cluster", "glossary_reconcile"],
    "glossary_cluster": ["glossary_reconcile"],
}

# Mapping from stage name to output file path functions and meta sidecars
# that should be deleted when the stage is invalidated.
_STAGE_OUTPUT_FILES: dict[str, list[str]] = {
    "glossary_cluster": ["glossary_cluster", "cluster_meta"],
    "glossary_reconcile": ["glossary"],
}


def _invalidate_downstream_stages(
    work_dir: Path,
    state: PipelineState,
    from_stage: str,
) -> None:
    """Reset downstream stages and delete their output/meta files.

    Called when *from_stage* produces new output that makes downstream
    outputs stale.  For each downstream stage, this:

    1. Resets the stage status and all its items to ``"pending"``.
    2. Deletes the stage's output files and meta sidecars.

    This is the **only** place downstream invalidation logic lives,
    ensuring consistent behaviour across ``--force``, ``--spine``,
    ``--batch``, and any future partial-rerun modes.
    """
    for downstream in _DOWNSTREAM_STAGES.get(from_stage, []):
        reset_stage(work_dir, state, downstream)

    # Delete output files for all downstream stages.
    file_getters = {
        "glossary_cluster": glossary_cluster_path,
        "cluster_meta": _cluster_meta_path,
        "glossary": glossary_path,
    }
    for downstream in _DOWNSTREAM_STAGES.get(from_stage, []):
        for file_key in _STAGE_OUTPUT_FILES.get(downstream, []):
            getter = file_getters.get(file_key)
            if getter:
                fp = getter(work_dir)
                if fp.exists():
                    fp.unlink()


# ---------------------------------------------------------------------------
# Build-meta sidecar helpers
# ---------------------------------------------------------------------------


def _build_meta_path(work_dir: Path) -> Path:
    """Return the path to the build-meta sidecar file."""
    return work_dir / _BUILD_META_FILENAME


def _load_build_meta(work_dir: Path) -> _BuildMeta:
    """Load the build-meta sidecar, returning a fresh one if absent."""
    p = _build_meta_path(work_dir)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return _BuildMeta(**data)
    return _BuildMeta()


def _save_build_meta(work_dir: Path, meta: _BuildMeta) -> None:
    """Atomically save the build-meta sidecar."""
    atomic_write(_build_meta_path(work_dir), meta.model_dump_json(indent=2))


def _remap_build_meta_conflicts(
    meta: _BuildMeta,
    loser_id: str,
    winner_id: str,
) -> None:
    """Update conflict records in *meta* after merging *loser_id* into *winner_id*.

    If the loser has a conflict record, its ``entity_id`` is remapped to the
    winner.  If both the winner and loser have records, their ``alternatives``
    and ``category_variants`` are merged (deduplicated) and the loser's record
    is removed.
    """
    winner_record: _ConflictRecord | None = None
    loser_record: _ConflictRecord | None = None

    for conflict in meta.conflicts:
        if conflict.entity_id == winner_id:
            winner_record = conflict
        elif conflict.entity_id == loser_id:
            loser_record = conflict

    if loser_record is not None:
        if winner_record is None:
            # Simple case: just remap the entity_id.
            loser_record.entity_id = winner_id
        else:
            # Both have records — merge loser into winner and remove loser.
            existing_english = {a["english"] for a in winner_record.alternatives}
            for alt in loser_record.alternatives:
                if alt["english"] not in existing_english:
                    winner_record.alternatives.append(alt)
                    existing_english.add(alt["english"])
            for cat in loser_record.category_variants:
                if cat not in winner_record.category_variants:
                    winner_record.category_variants.append(cat)
            meta.conflicts.remove(loser_record)


# ---------------------------------------------------------------------------
# Cluster-meta sidecar helpers
# ---------------------------------------------------------------------------


def _cluster_meta_path(work_dir: Path) -> Path:
    """Return the path to the cluster-meta sidecar file."""
    return work_dir / _CLUSTER_META_FILENAME


def _load_cluster_meta(work_dir: Path) -> _ClusterMeta:
    """Load the cluster-meta sidecar, returning a fresh one if absent."""
    p = _cluster_meta_path(work_dir)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return _ClusterMeta(**data)
    return _ClusterMeta()


def _save_cluster_meta(work_dir: Path, meta: _ClusterMeta) -> None:
    """Atomically save the cluster-meta sidecar."""
    atomic_write(_cluster_meta_path(work_dir), meta.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Glossary load / save helpers
# ---------------------------------------------------------------------------


def _load_glossary(work_dir: Path, path: Path | None = None) -> Glossary:
    """Load a glossary JSON file, or return a fresh one if absent.

    Parameters
    ----------
    work_dir:
        Work directory (used to derive the default path).
    path:
        Explicit file path.  When *None*, falls back to
        ``glossary_path(work_dir)`` (``glossary.json``).
    """
    gp = path or glossary_path(work_dir)
    if gp.exists():
        data = json.loads(gp.read_text(encoding="utf-8"))
        return Glossary(**data)
    return Glossary()


def load_glossary(work_dir: Path, config: AppConfig) -> Glossary:
    """Load ``glossary.json`` and validate categories against *config*.

    Public convenience wrapper used by consumer stages (translate, rebuild,
    toc) so that category mismatches are caught at load time rather than
    silently propagated into prompts.

    Raises
    ------
    ValueError
        If any glossary entity has a category not in ``config.glossary.categories``.
    """
    glossary = _load_glossary(work_dir)
    if glossary.entities:
        validate_glossary_categories(glossary, config.glossary.categories)
    return glossary


def _save_glossary(work_dir: Path, glossary: Glossary, path: Path | None = None) -> None:
    """Atomically save a glossary JSON file.

    Parameters
    ----------
    work_dir:
        Work directory (used to derive the default path).
    glossary:
        The glossary to persist.
    path:
        Explicit file path.  When *None*, falls back to
        ``glossary_path(work_dir)`` (``glossary.json``).
    """
    glossary.updated_at = datetime.now(timezone.utc)
    gp = path or glossary_path(work_dir)
    atomic_write(gp, glossary.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Merge logic (build stage)
# ---------------------------------------------------------------------------


def _merge_mention_into_glossary(
    glossary: Glossary,
    mention: ExtractedMention,
    batch_id: str,
    chunk_id: str,
    meta: _BuildMeta,
) -> None:
    """Merge a single extracted mention into the glossary.

    - If the mention links to an existing entity: add/update surface form,
      merge summary, merge aliases/nicknames/speech/notes.  Record conflicts
      if English or category differs.
    - If no match: create a new entity with one surface form.
    - ``source="user"`` entities are never modified.

    Parameters
    ----------
    glossary:
        The glossary to mutate.
    mention:
        A single extracted mention from the LLM.
    batch_id:
        Identifier for the current batch (for conflict tracking).
    chunk_id:
        ID of the first chunk in the batch.
    meta:
        Build metadata sidecar (mutated with conflict info).
    """
    entity = find_entity_for_mention(glossary, mention)

    if entity is None:
        # New entity.
        sf = SurfaceForm(
            source=mention.source,
            reading=mention.reading,
            english=mention.english,
            context_hints=[mention.context_hint] if mention.context_hint else [],
            notes=mention.notes,
            first_seen_chunk=chunk_id,
            occurrence_count=1,
        )
        new_entity = GlossaryEntity(
            entity_id=next_entity_id(mention.category, glossary),
            category=mention.category,
            canonical_english=mention.english,
            summary=mention.summary_update,
            surface_forms=[sf],
            aliases=list(mention.aliases),
            nicknames=dict(mention.nicknames),
            speech_style=mention.speech_style,
            notes=mention.notes,
            source="extracted",
            first_seen_chunk=chunk_id,
            latest_evidence_chunk=chunk_id,
        )
        glossary.entities.append(new_entity)
        return

    # Existing entity — never modify user-sourced entities.
    if entity.source == "user":
        logger.debug("Skipping user-sourced entity: %s", entity.canonical_english)
        return

    # Add or update the surface form on the entity.
    # Same-source translation disagreements are captured as
    # translation_variants on the SurfaceForm and resolved by
    # surface-form reconciliation.  New source forms with different
    # translations are legitimate (e.g. full name vs short name) and
    # do not warrant an entity-level conflict record.
    add_or_update_surface_form(entity, mention, chunk_id)

    # Merge summary.
    merge_entity_summary(entity, mention.summary_update, chunk_id)

    # Merge aliases, nicknames, speech_style, notes.
    merge_aliases_nicknames_speech_notes(entity, mention)

    # Update temporal tracking.
    entity.latest_evidence_chunk = chunk_id

    # Check for category conflict.
    if mention.category != entity.category:
        _record_category_conflict(
            meta,
            entity_id=entity.entity_id,
            source_form=mention.source,
            reading=mention.reading,
            current_english=entity.canonical_english,
            category=mention.category,
        )


def _merge_extraction_into_glossary(
    glossary: Glossary,
    response: GlossaryExtractionResponse,
    batch_id: str,
    first_chunk_id: str,
    meta: _BuildMeta,
) -> None:
    """Merge all extracted mentions and corrections into the glossary.

    Parameters
    ----------
    glossary:
        The glossary to mutate.
    response:
        The LLM extraction response containing mentions and corrections.
    batch_id:
        Identifier for the current batch.
    first_chunk_id:
        ID of the first chunk in the batch.
    meta:
        Build metadata sidecar (mutated with conflict info).
    """
    for mention in response.mentions:
        _merge_mention_into_glossary(glossary, mention, batch_id, first_chunk_id, meta)

    # Log corrections (not applied directly — recorded as conflicts for reconcile).
    for corr in response.corrections:
        meta.corrections.append(
            {
                "existing_english": corr.existing_english,
                "source_term": corr.source_term,
                "corrected_english": corr.corrected_english,
                "reason": corr.reason,
                "batch_id": batch_id,
            }
        )
        # Find the entity that owns this correction target.
        target_entity = _find_entity_for_correction(
            glossary, corr.existing_english, corr.source_term
        )
        _record_conflict(
            meta,
            entity_id=target_entity.entity_id if target_entity else f"unknown:{corr.source_term}",
            source_form=corr.source_term,
            reading=None,
            current_english=corr.existing_english,
            proposed_english=corr.corrected_english,
            batch_id=batch_id,
            context_snippet=f"Correction: {corr.reason}",
        )


def _record_conflict(
    meta: _BuildMeta,
    *,
    entity_id: str,
    source_form: str,
    reading: str | None,
    current_english: str,
    proposed_english: str,
    batch_id: str,
    context_snippet: str,
) -> None:
    """Record or append to an existing conflict in the build metadata."""
    for conflict in meta.conflicts:
        if conflict.entity_id == entity_id:
            # Append alternative if not already present.
            existing_proposals = {a["english"] for a in conflict.alternatives}
            if proposed_english not in existing_proposals:
                conflict.alternatives.append(
                    {
                        "english": proposed_english,
                        "context_snippet": context_snippet,
                        "batch_id": batch_id,
                    }
                )
            return

    meta.conflicts.append(
        _ConflictRecord(
            entity_id=entity_id,
            source_form=source_form,
            reading=reading,
            current_english=current_english,
            alternatives=[
                {
                    "english": proposed_english,
                    "context_snippet": context_snippet,
                    "batch_id": batch_id,
                }
            ],
        )
    )


def _record_category_conflict(
    meta: _BuildMeta,
    *,
    entity_id: str,
    source_form: str,
    reading: str | None,
    current_english: str,
    category: str,
) -> None:
    """Record a category variant for an entity."""
    for conflict in meta.conflicts:
        if conflict.entity_id == entity_id:
            if category not in conflict.category_variants:
                conflict.category_variants.append(category)
            return

    meta.conflicts.append(
        _ConflictRecord(
            entity_id=entity_id,
            source_form=source_form,
            reading=reading,
            current_english=current_english,
            category_variants=[category],
        )
    )


# ---------------------------------------------------------------------------
# glossary_build
# ---------------------------------------------------------------------------


def _build_work_items(
    all_chunks: list[Chunk],
    target_tokens: int,
    min_batch_tokens: int,
    redistribute_threshold: float,
    spine_width: int,
) -> list[_GlossaryBatch]:
    """Enumerate spine-aligned work items for glossary build.

    Returns a list of :class:`_GlossaryBatch` objects, each representing
    one LLM call.  Item IDs use the format ``"NNNN.bM"`` where NNNN is
    the padded spine index and M is the 1-based sub-batch index within
    that spine.

    Parameters
    ----------
    all_chunks:
        All chunks from the book, sorted by ``(spine_index, chunk_index)``.
    target_tokens:
        Token budget per LLM call.
    min_batch_tokens:
        Absorb final sub-batch into previous if below this.
    redistribute_threshold:
        Redistribute last two sub-batches if final is below
        ``target_tokens * redistribute_threshold``.
    spine_width:
        Zero-padding width for spine indices (from manifest).

    Returns
    -------
    list[_GlossaryBatch]
        One batch per LLM call, in spine order then sub-batch order.
    """
    spine_groups = _group_chunks_by_spine(all_chunks)
    work_items: list[_GlossaryBatch] = []

    for spine_index, spine_chunks in spine_groups:
        sub_batches = _pack_spine_batches(
            spine_chunks, target_tokens, min_batch_tokens, redistribute_threshold
        )
        spine_batch_count = len(sub_batches)
        for bi, batch in enumerate(sub_batches, 1):
            work_items.append(
                _GlossaryBatch(
                    item_id=f"{pad_spine(spine_index, spine_width)}.b{bi}",
                    spine_index=spine_index,
                    chunks=tuple(batch),
                    spine_batch_count=spine_batch_count,
                )
            )

    return work_items


def glossary_build(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
    retry_failed: bool = False,
    target_spine: int | None = None,
    target_batch: str | None = None,
    on_progress: Callable[[GlossaryBuildProgress], None] | None = None,
) -> Glossary:
    """Extract the per-book glossary from chunked source text.

    Groups chunks by spine item and packs each spine's chunks into
    sub-batches up to ``config.glossary_phase.target_tokens_per_call``.
    Each sub-batch is a separately tracked, resumable work item with an
    ID like ``"0003.b2"`` (spine 3, sub-batch 2).

    Parameters
    ----------
    work_dir:
        Resolved work directory.
    config:
        Application configuration.
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, reset and rebuild from scratch.
    retry_failed:
        If *True*, re-enter a completed stage to retry only failed batches.
        Preserves completed batch state (unlike ``force``).
    target_spine:
        If set, redo only the sub-batches for this spine index.
        Implies force for the targeted items.  Ignored if *target_batch*
        is also set.
    target_batch:
        If set, redo only this specific sub-batch (e.g. ``"0003.b2"``).
        Implies force for the targeted item.  Takes precedence over
        *target_spine*.
    on_progress:
        Optional callback invoked with a :class:`GlossaryBuildProgress`
        after each sub-batch is processed.

    Returns
    -------
    Glossary
        The accumulated per-book glossary.
    """
    from dao_bridge.schemas import Manifest

    stage = "glossary_build"

    # Load manifest and validate prerequisites.
    mp = manifest_path(work_dir)
    if not mp.exists():
        raise RuntimeError("Manifest not found. Run 'dao-bridge extract' first.")
    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    # Gate: chunk stage must be completed.
    if not is_stage_completed(state, "chunk"):
        raise RuntimeError("Chunk stage not completed. Run 'dao-bridge chunk' first.")

    # Determine if this is a targeted (partial) run.
    targeted = target_batch is not None or target_spine is not None

    # Handle force / already-completed.
    # Targeted runs handle their own reset below; skip full-stage reset.
    if force and not targeted:
        reset_stage(work_dir, state, stage)
        # Delete build output.
        bp = glossary_build_path(work_dir)
        if bp.exists():
            bp.unlink()
        # Delete build meta.
        bmp = _build_meta_path(work_dir)
        if bmp.exists():
            bmp.unlink()
        # Invalidate downstream stages (cluster + reconcile) — their
        # outputs depend on build output and are now stale.
        _invalidate_downstream_stages(work_dir, state, stage)
        # Migration cleanup: remove old snapshot files if they exist.
        for legacy in ["glossary_pre_cluster.json", "glossary_pre_reconcile.json"]:
            lp = work_dir / legacy
            if lp.exists():
                lp.unlink()

    # Handle --retry-failed: re-open a completed stage without wiping items.
    if retry_failed and not force and not targeted:
        reopen_stage(work_dir, state, stage)

    if not force and not retry_failed and not targeted and is_stage_completed(state, stage):
        logger.info("Glossary build already completed — skipping (use --force to re-run)")
        return _load_glossary(work_dir, glossary_build_path(work_dir))

    mark_stage_started(work_dir, state, stage)

    # Load all chunks and build spine-aligned work items.
    all_chunks = _load_all_chunks(work_dir, manifest)
    if not all_chunks:
        logger.warning("No chunks found — glossary will be empty.")
        glossary = _load_glossary(work_dir, glossary_build_path(work_dir))
        glossary.book_id = manifest.book_id
        glossary.book_metadata = manifest.metadata
        _save_glossary(work_dir, glossary, glossary_build_path(work_dir))
        mark_stage_completed(work_dir, state, stage)
        return glossary

    target_tokens = config.glossary_phase.target_tokens_per_call
    all_work_items = _build_work_items(
        all_chunks,
        target_tokens=target_tokens,
        min_batch_tokens=config.glossary_phase.min_batch_tokens,
        redistribute_threshold=config.glossary_phase.redistribute_threshold,
        spine_width=manifest.spine_padding_width,
    )
    all_item_ids = [batch.item_id for batch in all_work_items]

    # Filter to targeted items and reset their state.
    if targeted:
        if target_batch is not None:
            # --batch takes precedence: redo a single sub-batch.
            target_ids = [bid for bid in all_item_ids if bid == target_batch]
            if not target_ids:
                raise RuntimeError(
                    f"Batch '{target_batch}' not found.  Valid batch IDs: {', '.join(all_item_ids)}"
                )
        else:
            # --spine: redo all sub-batches for that spine.
            spine_prefix = f"{pad_spine(target_spine, manifest.spine_padding_width)}."  # type: ignore[arg-type]
            target_ids = [bid for bid in all_item_ids if bid.startswith(spine_prefix)]
            if not target_ids:
                raise RuntimeError(f"Spine {target_spine} has no glossary batches.")
        reset_stage_items(work_dir, state, stage, target_ids)
        # Targeted reruns change build output, making downstream stale.
        _invalidate_downstream_stages(work_dir, state, stage)
        work_items = [b for b in all_work_items if b.item_id in set(target_ids)]
    else:
        work_items = all_work_items

    item_ids = [batch.item_id for batch in work_items]
    items_total = len(item_ids)

    # Determine pending items.
    pending = set(iter_pending_items(state, stage, item_ids))

    # Load existing glossary and meta (for resume).
    glossary = _load_glossary(work_dir, glossary_build_path(work_dir))
    glossary.book_id = manifest.book_id
    glossary.book_metadata = manifest.metadata
    meta = _load_build_meta(work_dir)

    # Validate categories on any pre-existing entities (e.g. user-seeded).
    if glossary.entities:
        validate_glossary_categories(glossary, config.glossary.categories)

    # Resolve language names.
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    # Category info for the prompt.
    categories_str = "\n".join(f"- {cat}" for cat in config.glossary.categories)
    hints_lines = []
    for cat in config.glossary.categories:
        hint = config.glossary.category_hints.get(cat, "")
        if hint:
            hints_lines.append(f"- {cat}: {hint}")
        else:
            hints_lines.append(f"- {cat}")
    category_hints_str = "\n".join(hints_lines)

    # Load prompt template.
    template = _load_prompt_template("glossary_extract.txt")

    # Lazy LLM client.
    _llm_client: LLMClient | None = None

    def _get_llm_client() -> LLMClient:
        nonlocal _llm_client
        if _llm_client is None:
            _llm_client = LLMClient(config.models.glossary, config.llm)
        return _llm_client

    # Process each sub-batch.
    for batch in work_items:
        if batch.item_id not in pending:
            if on_progress:
                on_progress(
                    GlossaryBuildProgress(
                        item_id=batch.item_id,
                        spine_batch_count=batch.spine_batch_count,
                        items_total=items_total,
                    )
                )
            continue

        mark_item_started(work_dir, state, stage, batch.item_id)

        try:
            # Build the chunk text for the prompt.
            chunk_texts = []
            for c in batch.chunks:
                chunk_texts.append(f"--- chunk {c.chunk_id} ---\n{c.text}")
            chunk_batch_str = "\n\n".join(chunk_texts)

            # Render prompt.
            prompt = template.format(
                source_language=source_lang,
                target_language=target_lang,
                categories=categories_str,
                category_hints=category_hints_str,
                existing_glossary=_EXTRACTION_GLOSSARY_PLACEHOLDER,
                chunk_batch=chunk_batch_str,
            )

            # Call LLM.
            messages = [{"role": "user", "content": prompt}]
            client = _get_llm_client()
            response = client.complete_json(
                messages,
                response_model=GlossaryExtractionResponse,
                context_label=batch.item_id,
            )

            # Merge results.
            first_chunk_id = batch.chunks[0].chunk_id
            _merge_extraction_into_glossary(glossary, response, batch.item_id, first_chunk_id, meta)

            # Save after each sub-batch (resumable).
            _save_glossary(work_dir, glossary, glossary_build_path(work_dir))
            _save_build_meta(work_dir, meta)
            mark_item_completed(work_dir, state, stage, batch.item_id)

            logger.debug(
                "Item %s (%s): extracted %d mentions, %d corrections",
                batch.item_id,
                batch.chunk_range_label,
                len(response.mentions),
                len(response.corrections),
            )

        except LLMStructuredOutputError as exc:
            logger.error(
                "Structured output failed for item %s (%s): %s",
                batch.item_id,
                batch.chunk_range_label,
                exc,
            )
            mark_item_failed(work_dir, state, stage, batch.item_id, str(exc))
            # Save progress so far even on failure.
            _save_glossary(work_dir, glossary, glossary_build_path(work_dir))
            _save_build_meta(work_dir, meta)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in item %s (%s): %s",
                batch.item_id,
                batch.chunk_range_label,
                exc,
            )
            mark_item_failed(work_dir, state, stage, batch.item_id, str(exc))
            _save_glossary(work_dir, glossary, glossary_build_path(work_dir))
            _save_build_meta(work_dir, meta)
            raise

        if on_progress:
            on_progress(
                GlossaryBuildProgress(
                    item_id=batch.item_id,
                    spine_batch_count=batch.spine_batch_count,
                    items_total=items_total,
                )
            )

    # Mark stage completed only for full (non-targeted) runs.
    if not targeted:
        remaining = list(iter_pending_items(state, stage, all_item_ids))
        if not remaining:
            mark_stage_completed(work_dir, state, stage)

    logger.info(
        "Glossary build complete: %d entities, %d conflicts",
        len(glossary.entities),
        len(meta.conflicts),
    )

    return glossary


# ---------------------------------------------------------------------------
# glossary_cluster
# ---------------------------------------------------------------------------


def glossary_cluster(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
    retry_failed: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Glossary:
    """Find and merge duplicate entities that build-time linking missed.

    Iteratively generates candidate entity pairs using deterministic
    heuristics (substring containment, English containment, shared
    reading, alias overlap, Jaro-Winkler similarity), then sends each
    batch of candidates to the LLM for confirmation.  Confirmed pairs
    are merged and the glossary is saved after each batch.

    Candidate pairs are recomputed fresh each iteration — merges in one
    iteration may create new heuristic matches (e.g. merging A + B
    exposes a substring match to C).  Iterations are capped by
    ``config.glossary.cluster.max_iterations``.

    State is tracked at the **iteration** level (one item per iteration).
    If an iteration fails mid-batch, the completed merges from earlier
    batches are preserved in the glossary and the iteration is re-run
    from fresh candidates on resume.

    Reads from ``glossary_build.json`` (never mutated) and writes to
    ``glossary_cluster.json``.  Running with ``--force`` deletes the
    cluster output and re-reads from the pristine build output.

    Writes a clustering report to ``<work_dir>/glossary_cluster_report.md``.

    Parameters
    ----------
    work_dir:
        Resolved work directory.
    config:
        Application configuration.
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, delete cluster output, reset state and cluster meta,
        and cluster from scratch using ``glossary_build.json``.
    retry_failed:
        If *True*, re-enter a completed stage to retry only failed
        iterations.  Preserves completed iteration state.
    on_progress:
        Optional callback invoked with the iteration item ID (e.g.
        ``"iter1"``) after each iteration is processed.

    Returns
    -------
    Glossary
        The updated glossary with duplicate entities merged.
    """
    from dao_bridge.glossary_clustering import (
        _resolve_id,
        generate_cluster_candidates,
        merge_entities,
        remap_entity_id,
        render_entity_for_cluster_prompt,
        write_cluster_report,
    )
    from dao_bridge.schemas import GlossaryClusterResponse

    stage = "glossary_cluster"

    # Gate: glossary_build stage must be completed.
    if not is_stage_completed(state, "glossary_build"):
        raise RuntimeError(
            "Glossary build stage not completed. Run 'dao-bridge glossary-build' first."
        )

    # Handle force: delete cluster output, reset state + meta.
    if force:
        reset_stage(work_dir, state, stage)
        # Delete cluster output — will be regenerated from glossary_build.json.
        cp = glossary_cluster_path(work_dir)
        if cp.exists():
            cp.unlink()
        # Delete cluster meta.
        cmp = _cluster_meta_path(work_dir)
        if cmp.exists():
            cmp.unlink()
        # Invalidate downstream stages (reconcile) — output depends on
        # cluster output and is now stale.
        _invalidate_downstream_stages(work_dir, state, stage)

    # Handle --retry-failed: re-open a completed stage without wiping items.
    if retry_failed and not force:
        reopen_stage(work_dir, state, stage)

    if not force and not retry_failed and is_stage_completed(state, stage):
        logger.info("Glossary cluster already completed — skipping (use --force to re-run)")
        return _load_glossary(work_dir, glossary_cluster_path(work_dir))

    # Validate build output exists.
    bp = glossary_build_path(work_dir)
    if not bp.exists():
        raise RuntimeError(
            "Glossary build output not found. Run 'dao-bridge glossary-build' first."
        )

    # Load glossary: on resume load the in-progress cluster output if it
    # exists; on clean start load from build output.
    cp = glossary_cluster_path(work_dir)
    if cp.exists():
        glossary = _load_glossary(work_dir, cp)
    else:
        glossary = _load_glossary(work_dir, bp)
    validate_glossary_categories(glossary, config.glossary.categories)

    mark_stage_started(work_dir, state, stage)

    cluster_config = config.glossary.cluster
    batch_size = cluster_config.batch_size

    # Resolve language names for the prompt.
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    # Load prompt template.
    template = _load_prompt_template("glossary_cluster.txt")

    # Lazy LLM client.
    _llm_client: LLMClient | None = None

    def _get_llm_client() -> LLMClient:
        nonlocal _llm_client
        if _llm_client is None:
            _llm_client = LLMClient(config.models.glossary, config.llm)
        return _llm_client

    # Load cluster meta (for resume — contains merge_log from prior iterations).
    cluster_meta = _load_cluster_meta(work_dir)

    # Load build meta so we can remap conflict entity_ids as merges occur.
    # Conflicts are keyed by entity_id; when clustering absorbs an entity,
    # its conflict records must follow the surviving entity so that
    # reconcile can still find and apply them.
    build_meta = _load_build_meta(work_dir)

    # Seed report accumulators from cluster meta (so resumed runs include
    # data from prior completed iterations).
    merge_log: list[dict] = list(cluster_meta.merge_log)
    total_candidates_evaluated = cluster_meta.total_candidates_evaluated

    # Entity ID lookup helper.
    def _entity_by_id(eid: str) -> GlossaryEntity | None:
        for entity in glossary.entities:
            if entity.entity_id == eid:
                return entity
        return None

    # Iterative clustering loop.
    iteration = 0
    for iteration in range(1, cluster_config.max_iterations + 1):
        item_id = f"iter{iteration}"

        # Skip completed iterations.
        pending = list(iter_pending_items(state, stage, [item_id]))
        if not pending:
            if on_progress:
                on_progress(item_id)
            continue

        mark_item_started(work_dir, state, stage, item_id)

        # Generate candidates fresh from the current glossary state.
        candidates = generate_cluster_candidates(glossary, cluster_config)

        if not candidates:
            logger.debug("Clustering iteration %d: no candidates — stopping early", iteration)
            mark_item_completed(work_dir, state, stage, item_id)
            if on_progress:
                on_progress(item_id)
            break

        # Sort for deterministic processing order.
        candidate_list = sorted(candidates)

        # Split into batches.
        batches: list[list[tuple[str, str]]] = []
        for start in range(0, len(candidate_list), batch_size):
            batches.append(candidate_list[start : start + batch_size])

        merges_this_iteration = 0
        iteration_merge_entries: list[dict] = []
        iteration_candidates = 0
        # ID remap table for this iteration: absorbed_id -> surviving_id.
        id_map: dict[str, str] = {}

        try:
            for batch_idx, batch_pairs in enumerate(batches, 1):
                # Remap entity IDs in case earlier batches in this
                # iteration already merged some of these entities.
                raw_decisions = [(a, b, None) for a, b in batch_pairs]
                remapped = remap_entity_id(raw_decisions, id_map)
                # Rebuild pairs after remapping, dropping self-merges.
                active_pairs = [(a, b) for a, b, _ in remapped]

                if not active_pairs:
                    continue

                # Build prompt with entity pair renderings.
                pair_texts: list[str] = []
                for pair_idx, (eid_a, eid_b) in enumerate(active_pairs, 1):
                    ea = _entity_by_id(eid_a)
                    eb = _entity_by_id(eid_b)
                    if ea is None or eb is None:
                        continue
                    pair_texts.append(
                        f"--- Pair {pair_idx} ---\n"
                        f"Entity A:\n{render_entity_for_cluster_prompt(ea)}\n\n"
                        f"Entity B:\n{render_entity_for_cluster_prompt(eb)}"
                    )

                if not pair_texts:
                    continue

                entity_pairs_str = "\n\n".join(pair_texts)
                prompt = template.format(
                    source_language=source_lang,
                    target_language=target_lang,
                    entity_pairs=entity_pairs_str,
                )

                # Call LLM.
                messages = [{"role": "user", "content": prompt}]
                client = _get_llm_client()
                response = client.complete_json(
                    messages,
                    response_model=GlossaryClusterResponse,
                    context_label=f"cluster.{item_id}.batch{batch_idx}",
                )
                iteration_candidates += len(active_pairs)

                # Execute merges directly from the original LLM
                # decisions, resolving IDs on the fly through the remap
                # chain.  This handles same-batch chaining (A+B then
                # B+C) correctly regardless of chain depth, without
                # needing to re-remap a separate decisions list.
                for dec in response.decisions:
                    if not dec.same_entity:
                        continue

                    # Resolve both IDs through any prior merges.
                    resolved_a = _resolve_id(dec.entity_id_a, id_map)
                    resolved_b = _resolve_id(dec.entity_id_b, id_map)

                    # Already the same entity after earlier merges.
                    if resolved_a == resolved_b:
                        continue

                    ea = _entity_by_id(resolved_a)
                    eb = _entity_by_id(resolved_b)
                    if ea is None or eb is None:
                        logger.warning(
                            "Cluster merge skipped: entity %s or %s not found",
                            resolved_a,
                            resolved_b,
                        )
                        continue

                    # Determine winner from the LLM's preferred_entity_id,
                    # resolved through the remap chain.
                    winner, loser = ea, eb
                    if dec.preferred_entity_id:
                        resolved_pref = _resolve_id(dec.preferred_entity_id, id_map)
                        if resolved_pref == ea.entity_id:
                            winner, loser = ea, eb
                        elif resolved_pref == eb.entity_id:
                            winner, loser = eb, ea
                        else:
                            logger.warning(
                                "Cluster merge: preferred_entity_id %s "
                                "(resolved: %s) matches neither %s nor %s "
                                "— using default winner",
                                dec.preferred_entity_id,
                                resolved_pref,
                                ea.entity_id,
                                eb.entity_id,
                            )

                    pref_english = dec.preferred_canonical_english
                    reasoning = dec.reasoning or ""

                    # Record surface forms being added before merge.
                    existing_sources = {sf.source for sf in winner.surface_forms}
                    new_sf_labels = [
                        f"`{sf.source}` -> {sf.english}"
                        for sf in loser.surface_forms
                        if sf.source not in existing_sources
                    ]

                    merge_entities(winner, loser, pref_english)
                    glossary.entities.remove(loser)

                    # Update ID remap table.
                    id_map[loser.entity_id] = winner.entity_id

                    # Remap build-meta conflict records so reconcile can
                    # still find and apply them to the surviving entity.
                    _remap_build_meta_conflicts(build_meta, loser.entity_id, winner.entity_id)

                    iteration_merge_entries.append(
                        {
                            "winner_id": winner.entity_id,
                            "loser_id": loser.entity_id,
                            "winner_english": winner.canonical_english,
                            "loser_english": loser.canonical_english
                            if loser.canonical_english != winner.canonical_english
                            else pref_english or loser.canonical_english,
                            "result_english": winner.canonical_english,
                            "reasoning": reasoning,
                            "surface_forms_added": new_sf_labels,
                        }
                    )

                    merges_this_iteration += 1

                # Save glossary and build meta after each batch for crash safety.
                _save_glossary(work_dir, glossary, glossary_cluster_path(work_dir))
                _save_build_meta(work_dir, build_meta)

                logger.debug(
                    "Cluster %s batch %d: %d pairs evaluated, %d merges",
                    item_id,
                    batch_idx,
                    len(active_pairs),
                    sum(1 for d in response.decisions if d.same_entity),
                )

            # All batches in this iteration succeeded.
            merge_log.extend(iteration_merge_entries)
            total_candidates_evaluated += iteration_candidates
            cluster_meta.completed_iterations.append(iteration)
            cluster_meta.merge_log.extend(iteration_merge_entries)
            cluster_meta.total_candidates_evaluated += iteration_candidates
            _save_cluster_meta(work_dir, cluster_meta)
            mark_item_completed(work_dir, state, stage, item_id)

        except LLMStructuredOutputError as exc:
            logger.error("Structured output failed in cluster %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary, glossary_cluster_path(work_dir))
            _save_cluster_meta(work_dir, cluster_meta)
            _save_build_meta(work_dir, build_meta)
            raise
        except Exception as exc:
            logger.error("Unexpected error in cluster %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary, glossary_cluster_path(work_dir))
            _save_cluster_meta(work_dir, cluster_meta)
            _save_build_meta(work_dir, build_meta)
            raise

        if on_progress:
            on_progress(item_id)

        logger.info(
            "Clustering iteration %d: %d candidates, %d merges",
            iteration,
            len(candidate_list),
            merges_this_iteration,
        )

        # If no merges happened this iteration, further iterations will
        # produce the same candidates — stop early.
        if merges_this_iteration == 0:
            break

    # Ensure the cluster output file exists (e.g. zero candidates, no batches).
    _save_glossary(work_dir, glossary, glossary_cluster_path(work_dir))

    # Persist build meta with remapped conflict entity_ids.
    _save_build_meta(work_dir, build_meta)

    # Write clustering report.
    report_path = work_dir / "glossary_cluster_report.md"
    write_cluster_report(report_path, merge_log, iteration, total_candidates_evaluated)

    # Mark stage completed.
    mark_stage_completed(work_dir, state, stage)

    logger.info(
        "Glossary cluster complete: %d merges across %d iterations, %d entities remaining",
        len(merge_log),
        iteration,
        len(glossary.entities),
    )

    return glossary


# ---------------------------------------------------------------------------
# glossary_reconcile
# ---------------------------------------------------------------------------


def glossary_reconcile(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
    retry_failed: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Glossary:
    """Resolve within-book glossary conflicts from the build stage.

    For each entity conflict, calls the LLM to choose the best English
    form.  For character entities with multiple accumulated speech-style
    observations, consolidates them into a single coherent description.

    Reads from ``glossary_cluster.json`` (never mutated) and writes to
    ``glossary.json`` (the final glossary consumed by translation and
    export).  Running with ``--force`` deletes the reconcile output and
    re-reads from the pristine cluster output.

    Writes a reconciliation report to
    ``<work_dir>/glossary_reconcile_report.md``.

    Parameters
    ----------
    work_dir:
        Resolved work directory.
    config:
        Application configuration.
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, delete reconcile output, reset state, and reconcile
        from scratch using ``glossary_cluster.json``.
    retry_failed:
        If *True*, re-enter a completed stage to retry only failed items.
        Preserves completed item state (unlike ``force``).
    on_progress:
        Optional callback invoked with the item ID after each item
        is processed.

    Returns
    -------
    Glossary
        The updated glossary with conflicts resolved.
    """
    stage = "glossary_reconcile"

    # Gate: glossary_cluster stage must be completed.
    if not is_stage_completed(state, "glossary_cluster"):
        raise RuntimeError(
            "Glossary cluster stage not completed. Run 'dao-bridge glossary-cluster' first."
        )

    # Handle force: delete reconcile output and reset state.
    if force:
        reset_stage(work_dir, state, stage)
        # Delete reconcile output — will be regenerated from glossary_cluster.json.
        gp = glossary_path(work_dir)
        if gp.exists():
            gp.unlink()

    # Handle --retry-failed: re-open a completed stage without wiping items.
    if retry_failed and not force:
        reopen_stage(work_dir, state, stage)

    if not force and not retry_failed and is_stage_completed(state, stage):
        logger.info("Glossary reconcile already completed — skipping (use --force to re-run)")
        return _load_glossary(work_dir)

    # Validate cluster output exists.
    cp = glossary_cluster_path(work_dir)
    if not cp.exists():
        raise RuntimeError(
            "Glossary cluster output not found. Run 'dao-bridge glossary-cluster' first."
        )

    # Load glossary: on resume load the in-progress reconcile output
    # (glossary.json) if it exists; on clean start load from cluster output.
    gp = glossary_path(work_dir)
    if gp.exists() and not is_stage_completed(state, stage):
        # Resuming — load in-progress reconcile output.
        glossary = _load_glossary(work_dir, gp)
    else:
        glossary = _load_glossary(work_dir, cp)

    meta = _load_build_meta(work_dir)

    # Validate categories.
    validate_glossary_categories(glossary, config.glossary.categories)

    mark_stage_started(work_dir, state, stage)

    # Resolve language names.
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    # Build work items.
    # 0. Surface-form conflicts (resolved first so that entity-level
    #    conflicts see clean surface forms).
    sf_conflict_items: list[tuple[str, str, SurfaceForm]] = []
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.english_variants:
                item_id = f"glossary_reconcile.sf.{entity.entity_id}.{sf.source}"
                sf_conflict_items.append((item_id, entity.entity_id, sf))

    # 1. Entity conflicts (English form or category mismatches).
    term_items: list[tuple[str, _ConflictRecord]] = []
    for conflict in meta.conflicts:
        if conflict.alternatives or conflict.category_variants:
            item_id = f"glossary_reconcile.term.{conflict.entity_id}"
            term_items.append((item_id, conflict))

    # 2. Speech-style consolidation (per entity).
    speech_items: list[tuple[str, GlossaryEntity]] = []
    for entity in glossary.entities:
        if entity.speech_style and _SPEECH_STYLE_DELIMITER in entity.speech_style:
            item_id = f"glossary_reconcile.speech.{entity.entity_id}"
            speech_items.append((item_id, entity))

    all_item_ids = [item_id for item_id, _ in term_items] + [item_id for item_id, _ in speech_items]

    # If no work to do, complete immediately.
    if not sf_conflict_items and not all_item_ids:
        logger.info("No conflicts or speech-style consolidation needed.")
        _save_glossary(work_dir, glossary)
        mark_stage_completed(work_dir, state, stage)
        # Write empty report.
        _write_reconcile_report(work_dir, [], [], [])
        return glossary

    pending = set(iter_pending_items(state, stage, all_item_ids))

    # Load prompt templates.
    sf_conflict_template = _load_prompt_template("glossary_reconcile_surface_form.txt")
    term_template = _load_prompt_template("glossary_reconcile_term.txt")
    speech_template = _load_prompt_template("glossary_reconcile_speech.txt")

    # Lazy LLM client.
    _llm_client: LLMClient | None = None

    def _get_llm_client() -> LLMClient:
        nonlocal _llm_client
        if _llm_client is None:
            _llm_client = LLMClient(config.models.glossary, config.llm)
        return _llm_client

    # Track decisions for the report.
    sf_conflict_decisions: list[dict] = []
    term_decisions: list[dict] = []
    speech_decisions: list[dict] = []

    # --- Resolve surface-form conflicts (before entity-level) ---
    for item_id, entity_id, sf in sf_conflict_items:
        try:
            variants = list(sf.english_variants)
            alternatives_str = ", ".join(f'"{variant}"' for variant in variants)

            prompt = sf_conflict_template.format(
                source_language=source_lang,
                target_language=target_lang,
                source_term=sf.source,
                reading=sf.reading or "(none)",
                current_english=sf.english,
                alternatives=alternatives_str,
            )

            messages = [{"role": "user", "content": prompt}]
            client = _get_llm_client()
            result = client.complete_json(
                messages,
                response_model=GlossaryReconcileResponse,
                context_label=item_id,
            )

            # Apply chosen English to the specific surface form.
            entity = _find_entity_by_id(glossary, entity_id)
            old_english = sf.english
            resolved_sf: SurfaceForm | None = None
            if entity:
                for entity_sf in entity.surface_forms:
                    if entity_sf.source == sf.source:
                        entity_sf.english = result.chosen_english
                        entity_sf.english_variants = []
                        resolved_sf = entity_sf
                        break

            _save_glossary(work_dir, glossary)

            sf_conflict_decisions.append(
                {
                    "entity_id": entity_id,
                    "source_form": sf.source,
                    "old_english": old_english,
                    "chosen_english": result.chosen_english,
                    "reasoning": result.reasoning,
                    "alternatives": [
                        {
                            "english": variant,
                            "context_snippet": "Alternate English from glossary merge",
                        }
                        for variant in variants
                    ],
                }
            )

            logger.debug(
                "Resolved surface-form conflict %s (%s): '%s' -> '%s'",
                entity_id,
                sf.source,
                old_english,
                result.chosen_english,
            )
        except LLMStructuredOutputError as exc:
            logger.error("Structured output failed for %s: %s", item_id, exc)
            _save_glossary(work_dir, glossary)
            raise
        except Exception as exc:
            logger.error("Unexpected error for %s: %s", item_id, exc)
            _save_glossary(work_dir, glossary)
            raise

        if on_progress:
            on_progress(item_id)

    # --- Resolve entity conflicts ---
    for item_id, conflict in term_items:
        if item_id not in pending:
            if on_progress:
                on_progress(item_id)
            continue

        mark_item_started(work_dir, state, stage, item_id)

        try:
            if conflict.alternatives:
                # English-form conflict — resolve via LLM.
                alt_lines = []
                for alt in conflict.alternatives:
                    alt_lines.append(f'- "{alt["english"]}" ({alt["context_snippet"]})')
                alternatives_str = "\n".join(alt_lines)

                prompt = term_template.format(
                    source_language=source_lang,
                    target_language=target_lang,
                    source_term=conflict.source_form,
                    reading=conflict.reading or "(none)",
                    current_english=conflict.current_english,
                    alternatives=alternatives_str,
                )

                messages = [{"role": "user", "content": prompt}]
                client = _get_llm_client()
                result = client.complete_json(
                    messages,
                    response_model=GlossaryReconcileResponse,
                    context_label=item_id,
                )

                # Apply the winner.
                entity = _find_entity_by_id(glossary, conflict.entity_id)
                old_english = entity.canonical_english if entity else conflict.current_english
                if entity:
                    entity.canonical_english = result.chosen_english

                _save_glossary(work_dir, glossary)

                term_decisions.append(
                    {
                        "entity_id": conflict.entity_id,
                        "source_form": conflict.source_form,
                        "old_english": old_english,
                        "chosen_english": result.chosen_english,
                        "reasoning": result.reasoning,
                        "alternatives": conflict.alternatives,
                        "category_variants": conflict.category_variants,
                    }
                )

                logger.debug(
                    "Resolved %s (%s): '%s' -> '%s'",
                    conflict.entity_id,
                    conflict.source_form,
                    old_english,
                    result.chosen_english,
                )
            else:
                # Category-only conflict — log for the report, no LLM call.
                entity = _find_entity_by_id(glossary, conflict.entity_id)
                term_decisions.append(
                    {
                        "entity_id": conflict.entity_id,
                        "source_form": conflict.source_form,
                        "old_english": (
                            entity.canonical_english if entity else conflict.current_english
                        ),
                        "chosen_english": (
                            entity.canonical_english if entity else conflict.current_english
                        ),
                        "reasoning": (
                            "Category conflict only — kept existing category; review manually."
                        ),
                        "alternatives": [],
                        "category_variants": conflict.category_variants,
                    }
                )

                logger.debug(
                    "Category conflict for %s: variants %s — flagged for review",
                    conflict.entity_id,
                    conflict.category_variants,
                )

            mark_item_completed(work_dir, state, stage, item_id)

        except LLMStructuredOutputError as exc:
            logger.error("Structured output failed for term %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary)
            raise
        except Exception as exc:
            logger.error("Unexpected error reconciling term %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary)
            raise

        if on_progress:
            on_progress(item_id)

    # --- Consolidate speech styles ---
    for item_id, entity in speech_items:
        if item_id not in pending:
            if on_progress:
                on_progress(item_id)
            continue

        mark_item_started(work_dir, state, stage, item_id)

        try:
            observations = entity.speech_style.split(_SPEECH_STYLE_DELIMITER)
            observations_str = "\n".join(f"- {obs.strip()}" for obs in observations if obs.strip())

            prompt = speech_template.format(
                source_language=source_lang,
                character_name=entity.canonical_english,
                observations=observations_str,
            )

            messages = [{"role": "user", "content": prompt}]
            client = _get_llm_client()
            result = client.complete_json(
                messages,
                response_model=GlossarySpeechMergeResponse,
                context_label=item_id,
            )

            entity.speech_style = result.consolidated_speech_style

            _save_glossary(work_dir, glossary)

            speech_decisions.append(
                {
                    "entity_id": entity.entity_id,
                    "character": entity.canonical_english,
                    "old_observations": observations,
                    "consolidated": result.consolidated_speech_style,
                }
            )

            mark_item_completed(work_dir, state, stage, item_id)
            logger.debug(
                "Consolidated speech style for %s (%s)",
                entity.canonical_english,
                entity.entity_id,
            )

        except LLMStructuredOutputError as exc:
            logger.error("Structured output failed for speech style %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary)
            raise
        except Exception as exc:
            logger.error("Unexpected error consolidating speech style %s: %s", item_id, exc)
            mark_item_failed(work_dir, state, stage, item_id, str(exc))
            _save_glossary(work_dir, glossary)
            raise

        if on_progress:
            on_progress(item_id)

    # Save updated glossary in case only non-mutating items were processed.
    _save_glossary(work_dir, glossary)

    # Write reconciliation report.
    _write_reconcile_report(work_dir, sf_conflict_decisions, term_decisions, speech_decisions)

    # Mark stage completed if all items done.
    remaining = list(iter_pending_items(state, stage, all_item_ids))
    if not remaining:
        mark_stage_completed(work_dir, state, stage)

    logger.info(
        "Glossary reconcile complete: %d surface-form conflicts, "
        "%d term conflicts, %d speech styles consolidated",
        len(sf_conflict_decisions),
        len(term_decisions),
        len(speech_decisions),
    )

    return glossary


def _write_reconcile_report(
    work_dir: Path,
    sf_conflict_decisions: list[dict],
    term_decisions: list[dict],
    speech_decisions: list[dict],
) -> None:
    """Write the reconciliation report as markdown."""
    lines = ["# Glossary Reconciliation Report", ""]

    if not sf_conflict_decisions and not term_decisions and not speech_decisions:
        lines.append("No conflicts to resolve.")
        atomic_write(work_dir / "glossary_reconcile_report.md", "\n".join(lines))
        return

    if sf_conflict_decisions:
        lines.append("## Surface-Form Conflicts Resolved")
        lines.append("")
        for dec in sf_conflict_decisions:
            lines.append(f"### {dec.get('entity_id', 'unknown')} / {dec.get('source_form', '')}")
            lines.append(f"- **Previous:** {dec['old_english']}")
            lines.append(f"- **Chosen:** {dec['chosen_english']}")
            lines.append(f"- **Reasoning:** {dec['reasoning']}")
            if dec.get("alternatives"):
                lines.append("- **Alternatives considered:**")
                for alt in dec["alternatives"]:
                    lines.append(f'  - "{alt["english"]}" ({alt["context_snippet"]})')
            lines.append("")

    if term_decisions:
        lines.append("## Term Conflicts Resolved")
        lines.append("")
        for dec in term_decisions:
            lines.append(f"### {dec.get('entity_id', 'unknown')} ({dec.get('source_form', '')})")
            lines.append(f"- **Previous:** {dec['old_english']}")
            lines.append(f"- **Chosen:** {dec['chosen_english']}")
            lines.append(f"- **Reasoning:** {dec['reasoning']}")
            if dec.get("alternatives"):
                lines.append("- **Alternatives considered:**")
                for alt in dec["alternatives"]:
                    lines.append(f'  - "{alt["english"]}" ({alt["context_snippet"]})')
            if dec.get("category_variants"):
                lines.append(f"- **Category variants:** {', '.join(dec['category_variants'])}")
            lines.append("")

    if speech_decisions:
        lines.append("## Speech Style Consolidations")
        lines.append("")
        for dec in speech_decisions:
            lines.append(f"### {dec['character']} ({dec.get('entity_id', '')})")
            lines.append("- **Original observations:**")
            for obs in dec["old_observations"]:
                if obs.strip():
                    lines.append(f"  - {obs.strip()}")
            lines.append(f"- **Consolidated:** {dec['consolidated']}")
            lines.append("")

    atomic_write(work_dir / "glossary_reconcile_report.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# glossary_export
# ---------------------------------------------------------------------------


def glossary_export(
    work_dir: Path,
    config: AppConfig,
    *,
    stdout: bool = False,
    output_path: Path | None = None,
) -> str:
    """Produce a human-readable markdown view of the per-book glossary.

    Parameters
    ----------
    work_dir:
        Resolved work directory.
    config:
        Application configuration.
    stdout:
        If *True*, only return the markdown string (caller prints it).
    output_path:
        Custom output path.  Defaults to ``<work_dir>/glossary.md``.

    Returns
    -------
    str
        The rendered markdown.
    """
    glossary = _load_glossary(work_dir)

    # Validate categories.
    validate_glossary_categories(glossary, config.glossary.categories)

    if not glossary.entities:
        md = "# Glossary\n\nNo entities.\n"
        if not stdout:
            dest = output_path or (work_dir / "glossary.md")
            atomic_write(dest, md)
        return md

    # Group by category, sort alphabetically by canonical_english within each group.
    by_category: dict[str, list[GlossaryEntity]] = defaultdict(list)
    for entity in glossary.entities:
        by_category[entity.category].append(entity)

    # Order categories by config order, then any remaining alphabetically.
    category_order = list(config.glossary.categories)
    extra_cats = sorted(set(by_category.keys()) - set(category_order))
    ordered_cats = [c for c in category_order if c in by_category] + extra_cats

    lines = ["# Glossary", ""]

    for cat in ordered_cats:
        entities = sorted(by_category[cat], key=lambda e: e.canonical_english.lower())
        lines.append(f"## {cat.replace('_', ' ').title()}")
        lines.append("")

        for entity in entities:
            lines.append(f"### {entity.canonical_english} (`{entity.entity_id}`)")
            lines.append("")

            # Summary.
            if entity.summary:
                lines.append(f"Summary: {entity.summary}")
                lines.append("")

            # Surface forms.
            if entity.surface_forms:
                lines.append("Surface forms:")
                for sf in entity.surface_forms:
                    hint_str = ""
                    if sf.context_hints:
                        hint_str = f" (hints: {'; '.join(sf.context_hints)})"
                    lines.append(f"- `{sf.source}` -> {sf.english}{hint_str}")
                lines.append("")

            # Optional fields.
            if entity.aliases:
                lines.append(f"- Aliases: {', '.join(entity.aliases)}")
            if entity.nicknames:
                nick_parts = [f"{speaker} -> {nick}" for speaker, nick in entity.nicknames.items()]
                lines.append(f"- Nicknames: {'; '.join(nick_parts)}")
            if entity.speech_style:
                lines.append(f"- Speech style: {entity.speech_style}")
            if entity.notes:
                lines.append(f"- Notes: {entity.notes}")

            lines.append("")

    md = "\n".join(lines)

    if not stdout:
        dest = output_path or (work_dir / "glossary.md")
        atomic_write(dest, md)

    return md
