"""Reassembly of translated chunks into per-spine markdown files.

For each spine item that has translated chunks, this module concatenates
the translations in chunk order and writes the result to
``assembled/NNN.md``.

Items with ``chunk_count == 0`` (illustrations, auto-TOCs, etc.) are
skipped — the rebuild stage handles those by passing through raw XHTML.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dao_bridge.chunk import count_tokens
from dao_bridge.config import AppConfig
from dao_bridge.schemas import Chunk, Manifest, ManifestItem, TranslatedChunk
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    iter_pending_items,
    mark_item_completed,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
    reset_stage,
)
from dao_bridge.workdir import (
    assembled_path,
    atomic_write,
    chunk_dir,
    format_chunk_id,
    manifest_path,
    pad_spine,
    translation_path,
)

logger = logging.getLogger("dao_bridge")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_translation(work_dir: Path, chunk_id: str) -> TranslatedChunk:
    """Load a translated chunk from disk."""
    tp = translation_path(work_dir, chunk_id)
    if not tp.exists():
        raise FileNotFoundError(f"Translation file missing: {tp}")
    raw = json.loads(tp.read_text(encoding="utf-8"))
    return TranslatedChunk(**raw)


def _load_chunks_for_spine(work_dir: Path, spine_index: int) -> list[Chunk]:
    """Load all chunk JSON files for a spine item, sorted by chunk_index."""
    cd = chunk_dir(work_dir, spine_index)
    if not cd.exists():
        return []
    chunk_files = sorted(cd.glob("*.json"))
    chunks = []
    for cf in chunk_files:
        raw = json.loads(cf.read_text(encoding="utf-8"))
        chunks.append(Chunk(**raw))
    # Sort by chunk_index, not filename, for correctness.
    chunks.sort(key=lambda c: c.chunk_index)
    return chunks


# ---------------------------------------------------------------------------
# Single spine item assembly
# ---------------------------------------------------------------------------


def assemble_spine_item(
    work_dir: Path,
    spine_index: int,
    chunk_count: int,
) -> str:
    """Assemble translated chunks for a spine item into a single markdown string.

    Parameters
    ----------
    work_dir:
        Work directory root.
    spine_index:
        Spine index to assemble.
    chunk_count:
        Expected number of chunks (from manifest).

    Returns
    -------
    str
        Concatenated translated text.

    Raises
    ------
    FileNotFoundError
        If any expected translation file is missing.
    ValueError
        If the assembled output is empty.
    """
    # Load chunks to get expected chunk IDs.
    chunks = _load_chunks_for_spine(work_dir, spine_index)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found for spine {spine_index:03d}")

    # Verify we have the expected count.
    if len(chunks) != chunk_count:
        logger.warning(
            "Spine %03d: expected %d chunks but found %d chunk files",
            spine_index,
            chunk_count,
            len(chunks),
        )

    # Check all translations exist before loading any.
    missing: list[str] = []
    for c in chunks:
        tp = translation_path(work_dir, c.chunk_id)
        if not tp.exists():
            missing.append(c.chunk_id)

    if missing:
        raise FileNotFoundError(
            f"Missing translations for spine {spine_index:03d}: {', '.join(missing)}"
        )

    # Load translations in chunk order.
    translations: list[TranslatedChunk] = []
    for c in chunks:
        tc = _load_translation(work_dir, c.chunk_id)
        translations.append(tc)

    # Concatenate translated text.
    assembled = "\n\n".join(tc.translated_text for tc in translations)

    if not assembled.strip():
        raise ValueError(f"Assembled output for spine {spine_index:03d} is empty")

    # Sanity check: rough token count comparison.
    assembled_tokens = count_tokens(assembled)
    sum_translation_tokens = sum(count_tokens(tc.translated_text) for tc in translations)
    # Allow generous tolerance — join separators and whitespace differences.
    tolerance = max(len(translations) * 5, 10)
    if abs(assembled_tokens - sum_translation_tokens) > tolerance:
        logger.warning(
            "Spine %03d: assembled token count (%d) differs from sum of "
            "translation token counts (%d) by more than tolerance (%d)",
            spine_index,
            assembled_tokens,
            sum_translation_tokens,
            tolerance,
        )

    return assembled


# ---------------------------------------------------------------------------
# Translation completeness check
# ---------------------------------------------------------------------------


def _has_all_translations(work_dir: Path, spine_index: int, chunk_count: int) -> bool:
    """Return True if all expected translations exist for a spine item."""
    for ci in range(1, chunk_count + 1):
        chunk_id = format_chunk_id(spine_index, ci)
        tp = translation_path(work_dir, chunk_id)
        if not tp.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def assemble_all(
    config: AppConfig,
    manifest: Manifest,
    state: PipelineState,
    *,
    force: bool = False,
    spine_filter: int | None = None,
) -> Manifest:
    """Assemble all eligible spine items.

    Parameters
    ----------
    config:
        Application configuration.
    manifest:
        The manifest (not mutated by assembly).
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, reassemble even if already completed.
    spine_filter:
        If set, only assemble this spine index.

    Returns
    -------
    Manifest
        The manifest (unchanged).
    """
    work_dir = config.work_dir_path

    # Determine which items to process.
    if spine_filter is not None:
        items = [item for item in manifest.spine if item.spine_index == spine_filter]
        if not items:
            raise ValueError(f"Spine index {spine_filter} not found in manifest")
    else:
        items = list(manifest.spine)

    if force:
        reset_stage(work_dir, state, "assemble")

    if not force and is_stage_completed(state, "assemble") and spine_filter is None:
        logger.info("Assemble stage already completed — skipping (use --force to re-run)")
        return manifest

    mark_stage_started(work_dir, state, "assemble")

    # Build list of item IDs for pending check.
    item_ids = [pad_spine(item.spine_index) for item in items]
    pending = set(iter_pending_items(state, "assemble", item_ids))

    assembled_count = 0
    skipped_count = 0

    for item in items:
        padded = pad_spine(item.spine_index)

        # Skip if already completed (unless force).
        if not force and padded not in pending:
            continue

        # Skip items with no chunks (illustrations, auto-toc, etc.).
        chunk_count = item.chunk_count or 0
        if chunk_count == 0:
            mark_item_started(work_dir, state, "assemble", padded)
            mark_item_completed(work_dir, state, "assemble", padded)
            skipped_count += 1
            logger.debug(
                "Spine %s: chunk_count=0, nothing to assemble",
                padded,
            )
            continue

        # Check if translations are available.
        if not _has_all_translations(work_dir, item.spine_index, chunk_count):
            logger.warning(
                "Spine %s: translations incomplete (%d chunks expected), skipping",
                padded,
                chunk_count,
            )
            skipped_count += 1
            continue

        mark_item_started(work_dir, state, "assemble", padded)

        try:
            text = assemble_spine_item(work_dir, item.spine_index, chunk_count)

            # Write assembled markdown.
            ap = assembled_path(work_dir, item.spine_index)
            ap.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(ap, text)

            mark_item_completed(work_dir, state, "assemble", padded)
            assembled_count += 1
            logger.debug(
                "Spine %s: assembled %d chunks into %s",
                padded,
                chunk_count,
                ap.name,
            )
        except Exception as exc:
            mark_item_failed(work_dir, state, "assemble", padded, str(exc))
            raise

    # Mark stage complete only if processing all items (not a single --spine).
    if spine_filter is None:
        mark_stage_completed(work_dir, state, "assemble")

    logger.info(
        "Assembly complete: %d items assembled, %d items skipped",
        assembled_count,
        skipped_count,
    )

    return manifest
