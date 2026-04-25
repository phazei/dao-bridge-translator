"""Glossary extraction, reconciliation, and export.

Three-stage glossary pipeline:

1. **Build** — extracts proper nouns and character information from chunked
   source text via batched LLM calls.  Accumulates a per-book glossary,
   saving after each batch for resumability.
2. **Reconcile** — resolves within-book conflicts (differing English
   proposals, corrections from the LLM, category mismatches) and
   consolidates multiple speech-style observations per character.
3. **Export** — renders the glossary as human-readable markdown for review.

The ``source`` field on each :class:`~dao_bridge.schemas.GlossaryEntry`
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
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from dao_bridge.chunk import count_tokens
from dao_bridge.config import AppConfig, resolve_language_name
from dao_bridge.llm_client import LLMClient, LLMStructuredOutputError
from dao_bridge.schemas import (
    Chunk,
    Glossary,
    GlossaryEntry,
    GlossaryExtractionResponse,
    GlossaryReconcileResponse,
    GlossarySpeechMergeResponse,
)
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
)
from dao_bridge.workdir import (
    atomic_write,
    chunk_dir,
    glossary_path,
    manifest_path,
)

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parent / "prompts"
_BUILD_META_FILENAME = "_glossary_build_meta.json"

# Delimiter used to accumulate multiple speech-style observations before
# the reconcile stage consolidates them.
_SPEECH_STYLE_DELIMITER = "\n"

# ---------------------------------------------------------------------------
# Build metadata sidecar (internal, not part of the public glossary format)
# ---------------------------------------------------------------------------


class _ConflictRecord(BaseModel):
    """A single term conflict detected during the build stage."""

    source_term: str
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
    """Validate that every entry's category is in the allowed list.

    Raises
    ------
    ValueError
        With a clear message listing each invalid category and the entries
        that use it.
    """
    valid = set(categories)
    invalid: dict[str, list[str]] = defaultdict(list)
    for entry in glossary.entries:
        if entry.category not in valid:
            label = entry.english or entry.source_term or "(unknown)"
            invalid[entry.category].append(label)
    if invalid:
        lines = []
        for cat, entries in sorted(invalid.items()):
            lines.append(f"  '{cat}': used by {', '.join(entries)}")
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
# Batch packing
# ---------------------------------------------------------------------------


def _pack_batches(chunks: list[Chunk], target_tokens: int) -> list[list[Chunk]]:
    """Greedy-pack chunks into batches up to *target_tokens*.

    Accumulates chunks in order until adding the next chunk would exceed
    the target.  Emits the current batch and starts a new one.  The last
    batch is emitted as-is regardless of size.

    Parameters
    ----------
    chunks:
        Flat list of chunks in spine+chunk order.
    target_tokens:
        Maximum token count per batch.

    Returns
    -------
    list[list[Chunk]]
        List of batches, each batch a list of chunks.
    """
    if not chunks:
        return []

    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        if current_batch and current_tokens + chunk.token_count > target_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += chunk.token_count

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Glossary rendering (for prompt injection)
# ---------------------------------------------------------------------------


def _render_existing_glossary(glossary: Glossary, max_tokens: int) -> str:
    """Render the current glossary compactly for prompt injection.

    Groups entries by category, one line per entry.  If the total exceeds
    *max_tokens*, truncates to the most recently added entries with a note.
    """
    if not glossary.entries:
        return "(no entries yet)"

    # Group by category.
    by_category: dict[str, list[GlossaryEntry]] = defaultdict(list)
    for entry in glossary.entries:
        by_category[entry.category].append(entry)

    lines: list[str] = []
    for cat in sorted(by_category.keys()):
        lines.append(f"[{cat}]")
        for e in by_category[cat]:
            parts = [f"{e.source_term or '?'} -> {e.english}"]
            if e.reading:
                parts.append(f"reading: {e.reading}")
            if e.aliases:
                parts.append(f"aliases: {', '.join(e.aliases)}")
            lines.append("  " + " | ".join(parts))

    rendered = "\n".join(lines)

    # Check token count and truncate if needed.
    token_count = count_tokens(rendered)
    if token_count > max_tokens:
        # Truncate: keep the tail (most recently added) entries.
        # Rebuild from the end of the entry list.
        truncated_entries = list(reversed(glossary.entries))
        kept: list[GlossaryEntry] = []
        running_tokens = 0
        for entry in truncated_entries:
            line = f"  {entry.source_term or '?'} -> {entry.english}"
            line_tokens = count_tokens(line)
            if running_tokens + line_tokens > max_tokens - 50:  # reserve space for header
                break
            kept.append(entry)
            running_tokens += line_tokens

        kept.reverse()
        by_cat_truncated: dict[str, list[GlossaryEntry]] = defaultdict(list)
        for e in kept:
            by_cat_truncated[e.category].append(e)

        trunc_lines = [
            f"(truncated — showing {len(kept)} of {len(glossary.entries)} entries, most recent)"
        ]
        for cat in sorted(by_cat_truncated.keys()):
            trunc_lines.append(f"[{cat}]")
            for e in by_cat_truncated[cat]:
                parts = [f"{e.source_term or '?'} -> {e.english}"]
                if e.reading:
                    parts.append(f"reading: {e.reading}")
                if e.aliases:
                    parts.append(f"aliases: {', '.join(e.aliases)}")
                trunc_lines.append("  " + " | ".join(parts))

        rendered = "\n".join(trunc_lines)

    return rendered


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


# ---------------------------------------------------------------------------
# Glossary load / save helpers
# ---------------------------------------------------------------------------


def _load_glossary(work_dir: Path) -> Glossary:
    """Load ``glossary.json`` from the work directory, or return a fresh one."""
    gp = glossary_path(work_dir)
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
        If any glossary entry has a category not in ``config.glossary.categories``.
    """
    glossary = _load_glossary(work_dir)
    if glossary.entries:
        validate_glossary_categories(glossary, config.glossary.categories)
    return glossary


def _save_glossary(work_dir: Path, glossary: Glossary) -> None:
    """Atomically save ``glossary.json``."""
    glossary.updated_at = datetime.now(timezone.utc)
    atomic_write(glossary_path(work_dir), glossary.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Merge logic (build stage)
# ---------------------------------------------------------------------------


def _find_entry_by_source_term(glossary: Glossary, source_term: str) -> GlossaryEntry | None:
    """Find an existing entry by source_term (case-sensitive)."""
    for entry in glossary.entries:
        if entry.source_term == source_term:
            return entry
    return None


def _merge_extraction_into_glossary(
    glossary: Glossary,
    response: GlossaryExtractionResponse,
    batch_id: str,
    first_chunk_id: str,
    meta: _BuildMeta,
) -> None:
    """Merge extracted entries into the glossary, logging conflicts.

    - New source_term: add with ``source="extracted"``.
    - Existing source_term: union aliases, merge nicknames, accumulate
      speech_style.  Differing English proposals go to the conflict list.
    - ``source="user"`` entries: never modified.
    - Corrections are logged to the meta sidecar, not applied.
    """
    for ext_entry in response.entries:
        existing = _find_entry_by_source_term(glossary, ext_entry.source_term)

        if existing is None:
            # New entry.
            glossary.entries.append(
                GlossaryEntry(
                    source_term=ext_entry.source_term,
                    reading=ext_entry.reading,
                    english=ext_entry.english_proposed,
                    category=ext_entry.category,
                    first_seen_chunk=first_chunk_id,
                    aliases=list(ext_entry.aliases),
                    nicknames=dict(ext_entry.nicknames),
                    speech_style=ext_entry.speech_style,
                    notes=ext_entry.notes,
                    source="extracted",
                )
            )
            continue

        # Existing entry — never modify user-sourced entries.
        if existing.source == "user":
            logger.debug("Skipping user-sourced entry: %s", existing.english)
            continue

        # Backfill reading if an earlier extraction did not have one.
        if not existing.reading and ext_entry.reading:
            existing.reading = ext_entry.reading

        # Union aliases.
        alias_set = set(existing.aliases)
        for alias in ext_entry.aliases:
            if alias not in alias_set:
                existing.aliases.append(alias)
                alias_set.add(alias)

        # Merge nicknames (existing wins on key conflict).
        for speaker, nick in ext_entry.nicknames.items():
            if speaker not in existing.nicknames:
                existing.nicknames[speaker] = nick

        # Accumulate speech_style observations.
        if ext_entry.speech_style:
            if existing.speech_style:
                # Only add if not already present (dedup identical sentences).
                existing_observations = existing.speech_style.split(_SPEECH_STYLE_DELIMITER)
                if ext_entry.speech_style not in existing_observations:
                    existing.speech_style = (
                        existing.speech_style + _SPEECH_STYLE_DELIMITER + ext_entry.speech_style
                    )
            else:
                existing.speech_style = ext_entry.speech_style

        # Concatenate notes if new info.
        if ext_entry.notes and ext_entry.notes != existing.notes:
            if existing.notes:
                if ext_entry.notes not in existing.notes:
                    existing.notes = existing.notes + " " + ext_entry.notes
            else:
                existing.notes = ext_entry.notes

        # Check for English-form conflict.
        if ext_entry.english_proposed != existing.english:
            _record_conflict(
                meta,
                source_term=ext_entry.source_term,
                reading=ext_entry.reading or existing.reading,
                current_english=existing.english,
                proposed_english=ext_entry.english_proposed,
                batch_id=batch_id,
                context_snippet=f"Batch {batch_id}",
            )

        # Check for category conflict.
        if ext_entry.category != existing.category:
            _record_category_conflict(
                meta,
                source_term=ext_entry.source_term,
                reading=ext_entry.reading or existing.reading,
                current_english=existing.english,
                category=ext_entry.category,
            )

    # Log corrections (not applied).
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
        # Also record as a conflict for reconcile to handle.
        _record_conflict(
            meta,
            source_term=corr.source_term,
            reading=None,
            current_english=corr.existing_english,
            proposed_english=corr.corrected_english,
            batch_id=batch_id,
            context_snippet=f"Correction: {corr.reason}",
        )


def _record_conflict(
    meta: _BuildMeta,
    *,
    source_term: str,
    reading: str | None,
    current_english: str,
    proposed_english: str,
    batch_id: str,
    context_snippet: str,
) -> None:
    """Record or append to an existing conflict in the build metadata."""
    for conflict in meta.conflicts:
        if conflict.source_term == source_term:
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
            source_term=source_term,
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
    source_term: str,
    reading: str | None,
    current_english: str,
    category: str,
) -> None:
    """Record a category variant for a term."""
    for conflict in meta.conflicts:
        if conflict.source_term == source_term:
            if category not in conflict.category_variants:
                conflict.category_variants.append(category)
            return

    meta.conflicts.append(
        _ConflictRecord(
            source_term=source_term,
            reading=reading,
            current_english=current_english,
            category_variants=[category],
        )
    )


# ---------------------------------------------------------------------------
# glossary_build
# ---------------------------------------------------------------------------


def glossary_build(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
    retry_failed: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Glossary:
    """Extract the per-book glossary from chunked source text.

    Greedy-packs chunks into batches up to
    ``config.glossary_phase.target_tokens_per_call``, then sends each
    batch to the LLM for extraction.  Entries are merged into the
    glossary progressively, saving after each batch.

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
    on_progress:
        Optional callback invoked with the batch ID after each batch
        is processed.

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

    # Handle force / already-completed.
    if force:
        reset_stage(work_dir, state, stage)
        # Also reset glossary and meta files.
        gp = glossary_path(work_dir)
        if gp.exists():
            gp.unlink()
        bmp = _build_meta_path(work_dir)
        if bmp.exists():
            bmp.unlink()

    # Handle --retry-failed: re-open a completed stage without wiping items.
    if retry_failed and not force:
        reopen_stage(work_dir, state, stage)

    if not force and not retry_failed and is_stage_completed(state, stage):
        logger.info("Glossary build already completed — skipping (use --force to re-run)")
        return _load_glossary(work_dir)

    mark_stage_started(work_dir, state, stage)

    # Load all chunks and pack into batches.
    all_chunks = _load_all_chunks(work_dir, manifest)
    if not all_chunks:
        logger.warning("No chunks found — glossary will be empty.")
        glossary = _load_glossary(work_dir)
        glossary.book_id = manifest.book_id
        glossary.book_metadata = manifest.metadata
        _save_glossary(work_dir, glossary)
        mark_stage_completed(work_dir, state, stage)
        return glossary

    target_tokens = config.glossary_phase.target_tokens_per_call
    batches = _pack_batches(all_chunks, target_tokens)
    batch_ids = [f"glossary_build.batch.{i + 1:03d}" for i in range(len(batches))]

    # Determine pending batches.
    pending = set(iter_pending_items(state, stage, batch_ids))

    # Load existing glossary and meta (for resume).
    glossary = _load_glossary(work_dir)
    glossary.book_id = manifest.book_id
    glossary.book_metadata = manifest.metadata
    meta = _load_build_meta(work_dir)

    # Validate categories on any pre-existing entries (e.g. user-seeded).
    if glossary.entries:
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

    # Process each batch.
    for batch, batch_id in zip(batches, batch_ids):
        if batch_id not in pending:
            if on_progress:
                on_progress(batch_id)
            continue

        mark_item_started(work_dir, state, stage, batch_id)

        try:
            # Build the chunk text for the prompt.
            chunk_texts = []
            for c in batch:
                chunk_texts.append(f"--- chunk {c.chunk_id} ---\n{c.text}")
            chunk_batch_str = "\n\n".join(chunk_texts)

            # Render existing glossary (limit to half target_tokens).
            existing_glossary_str = _render_existing_glossary(
                glossary, max_tokens=target_tokens // 2
            )

            # Render prompt.
            prompt = template.format(
                source_language=source_lang,
                target_language=target_lang,
                categories=categories_str,
                category_hints=category_hints_str,
                existing_glossary=existing_glossary_str,
                chunk_batch=chunk_batch_str,
            )

            # Call LLM.
            messages = [{"role": "user", "content": prompt}]
            client = _get_llm_client()
            response = client.complete_json(
                messages,
                response_model=GlossaryExtractionResponse,
                context_label=batch_id,
            )

            # Merge results.
            first_chunk_id = batch[0].chunk_id
            _merge_extraction_into_glossary(glossary, response, batch_id, first_chunk_id, meta)

            # Save after each batch (resumable).
            _save_glossary(work_dir, glossary)
            _save_build_meta(work_dir, meta)
            mark_item_completed(work_dir, state, stage, batch_id)

            logger.debug(
                "Batch %s: extracted %d entries, %d corrections",
                batch_id,
                len(response.entries),
                len(response.corrections),
            )

        except LLMStructuredOutputError as exc:
            logger.error("Structured output failed for batch %s: %s", batch_id, exc)
            mark_item_failed(work_dir, state, stage, batch_id, str(exc))
            # Save progress so far even on failure.
            _save_glossary(work_dir, glossary)
            _save_build_meta(work_dir, meta)
            raise
        except Exception as exc:
            logger.error("Unexpected error in batch %s: %s", batch_id, exc)
            mark_item_failed(work_dir, state, stage, batch_id, str(exc))
            _save_glossary(work_dir, glossary)
            _save_build_meta(work_dir, meta)
            raise

        if on_progress:
            on_progress(batch_id)

    # Mark stage completed.
    remaining = list(iter_pending_items(state, stage, batch_ids))
    if not remaining:
        mark_stage_completed(work_dir, state, stage)

    logger.info(
        "Glossary build complete: %d entries, %d conflicts",
        len(glossary.entries),
        len(meta.conflicts),
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

    For each term conflict, calls the LLM to choose the best English
    form.  For characters with multiple accumulated speech-style
    observations, consolidates them into a single coherent description.

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
        If *True*, reset and reconcile from scratch.
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

    # Gate: glossary_build stage must be completed.
    if not is_stage_completed(state, "glossary_build"):
        raise RuntimeError(
            "Glossary build stage not completed. Run 'dao-bridge glossary-build' first."
        )

    # Handle force / already-completed.
    if force:
        reset_stage(work_dir, state, stage)

    # Handle --retry-failed: re-open a completed stage without wiping items.
    if retry_failed and not force:
        reopen_stage(work_dir, state, stage)

    if not force and not retry_failed and is_stage_completed(state, stage):
        logger.info("Glossary reconcile already completed — skipping (use --force to re-run)")
        return _load_glossary(work_dir)

    # Load glossary and build meta.
    glossary = _load_glossary(work_dir)
    meta = _load_build_meta(work_dir)

    # Validate categories.
    validate_glossary_categories(glossary, config.glossary.categories)

    mark_stage_started(work_dir, state, stage)

    # Resolve language names.
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    # Build work items.
    # 1. Term conflicts (English form or category mismatches).
    term_items: list[tuple[str, _ConflictRecord]] = []
    for conflict in meta.conflicts:
        if conflict.alternatives or conflict.category_variants:
            item_id = f"glossary_reconcile.term.{conflict.source_term}"
            term_items.append((item_id, conflict))

    # 2. Speech-style consolidation.
    speech_items: list[tuple[str, GlossaryEntry]] = []
    for entry in glossary.entries:
        if entry.speech_style and _SPEECH_STYLE_DELIMITER in entry.speech_style:
            item_id = f"glossary_reconcile.speech.{entry.english}"
            speech_items.append((item_id, entry))

    all_item_ids = [item_id for item_id, _ in term_items] + [item_id for item_id, _ in speech_items]

    # If no work to do, complete immediately.
    if not all_item_ids:
        logger.info("No conflicts or speech-style consolidation needed.")
        _save_glossary(work_dir, glossary)
        mark_stage_completed(work_dir, state, stage)
        # Write empty report.
        _write_reconcile_report(work_dir, [], [])
        return glossary

    pending = set(iter_pending_items(state, stage, all_item_ids))

    # Load prompt templates.
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
    term_decisions: list[dict] = []
    speech_decisions: list[dict] = []

    # --- Resolve term conflicts ---
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
                    source_term=conflict.source_term,
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
                entry = _find_entry_by_source_term(glossary, conflict.source_term)
                old_english = entry.english if entry else conflict.current_english
                if entry:
                    entry.english = result.chosen_english

                _save_glossary(work_dir, glossary)

                term_decisions.append(
                    {
                        "source_term": conflict.source_term,
                        "old_english": old_english,
                        "chosen_english": result.chosen_english,
                        "reasoning": result.reasoning,
                        "alternatives": conflict.alternatives,
                        "category_variants": conflict.category_variants,
                    }
                )

                logger.debug(
                    "Resolved %s: '%s' -> '%s'",
                    conflict.source_term,
                    old_english,
                    result.chosen_english,
                )
            else:
                # Category-only conflict — log for the report, no LLM call.
                entry = _find_entry_by_source_term(glossary, conflict.source_term)
                term_decisions.append(
                    {
                        "source_term": conflict.source_term,
                        "old_english": entry.english if entry else conflict.current_english,
                        "chosen_english": entry.english if entry else conflict.current_english,
                        "reasoning": (
                            "Category conflict only — kept existing category; review manually."
                        ),
                        "alternatives": [],
                        "category_variants": conflict.category_variants,
                    }
                )

                logger.debug(
                    "Category conflict for %s: variants %s — flagged for review",
                    conflict.source_term,
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
    for item_id, entry in speech_items:
        if item_id not in pending:
            if on_progress:
                on_progress(item_id)
            continue

        mark_item_started(work_dir, state, stage, item_id)

        try:
            observations = entry.speech_style.split(_SPEECH_STYLE_DELIMITER)
            observations_str = "\n".join(f"- {obs.strip()}" for obs in observations if obs.strip())

            prompt = speech_template.format(
                source_language=source_lang,
                character_name=entry.english,
                observations=observations_str,
            )

            messages = [{"role": "user", "content": prompt}]
            client = _get_llm_client()
            result = client.complete_json(
                messages,
                response_model=GlossarySpeechMergeResponse,
                context_label=item_id,
            )

            entry.speech_style = result.consolidated_speech_style

            _save_glossary(work_dir, glossary)

            speech_decisions.append(
                {
                    "character": entry.english,
                    "old_observations": observations,
                    "consolidated": result.consolidated_speech_style,
                }
            )

            mark_item_completed(work_dir, state, stage, item_id)
            logger.debug("Consolidated speech style for %s", entry.english)

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
    _write_reconcile_report(work_dir, term_decisions, speech_decisions)

    # Mark stage completed if all items done.
    remaining = list(iter_pending_items(state, stage, all_item_ids))
    if not remaining:
        mark_stage_completed(work_dir, state, stage)

    logger.info(
        "Glossary reconcile complete: %d term conflicts resolved, %d speech styles consolidated",
        len(term_decisions),
        len(speech_decisions),
    )

    return glossary


def _write_reconcile_report(
    work_dir: Path,
    term_decisions: list[dict],
    speech_decisions: list[dict],
) -> None:
    """Write the reconciliation report as markdown."""
    lines = ["# Glossary Reconciliation Report", ""]

    if not term_decisions and not speech_decisions:
        lines.append("No conflicts to resolve.")
        atomic_write(work_dir / "glossary_reconcile_report.md", "\n".join(lines))
        return

    if term_decisions:
        lines.append("## Term Conflicts Resolved")
        lines.append("")
        for dec in term_decisions:
            lines.append(f"### {dec['source_term']}")
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
            lines.append(f"### {dec['character']}")
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

    if not glossary.entries:
        md = "# Glossary\n\nNo entries.\n"
        if not stdout:
            dest = output_path or (work_dir / "glossary.md")
            atomic_write(dest, md)
        return md

    # Group by category, sort alphabetically by english within each group.
    by_category: dict[str, list[GlossaryEntry]] = defaultdict(list)
    for entry in glossary.entries:
        by_category[entry.category].append(entry)

    # Order categories by config order, then any remaining alphabetically.
    category_order = list(config.glossary.categories)
    extra_cats = sorted(set(by_category.keys()) - set(category_order))
    ordered_cats = [c for c in category_order if c in by_category] + extra_cats

    lines = ["# Glossary", ""]

    for cat in ordered_cats:
        entries = sorted(by_category[cat], key=lambda e: e.english.lower())
        lines.append(f"## {cat.replace('_', ' ').title()}")
        lines.append("")

        for entry in entries:
            # Header line.
            if entry.source_term:
                lines.append(f"**{entry.english}** ({entry.source_term})")
            else:
                lines.append(f"**{entry.english}**")

            # Optional fields.
            if entry.reading:
                lines.append(f"- Reading: {entry.reading}")
            if entry.aliases:
                lines.append(f"- Aliases: {', '.join(entry.aliases)}")
            if entry.nicknames:
                nick_parts = [f"{speaker} -> {nick}" for speaker, nick in entry.nicknames.items()]
                lines.append(f"- Nicknames: {'; '.join(nick_parts)}")
            if entry.speech_style:
                lines.append(f"- Speech style: {entry.speech_style}")
            if entry.notes:
                lines.append(f"- Notes: {entry.notes}")

            lines.append("")

    md = "\n".join(lines)

    if not stdout:
        dest = output_path or (work_dir / "glossary.md")
        atomic_write(dest, md)

    return md
