"""Deterministic paragraph-aware chunking for spine items.

For each spine item whose classification is chunkable, this module:

1. Parses ``clean/NNN.md`` into a list of :class:`Block` objects.
2. Detects and optionally normalises scene breaks.
3. Packs blocks into chunks using a greedy algorithm that prefers
   natural break points (scene breaks, headings, HRs).
4. Validates block coverage (no gaps, no overlaps).
5. Writes each chunk as ``chunks/NNN/NNN.MMM.json``.

The chunker is deterministic — the same input always produces the same
output, which is critical for resumability and debugging.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tiktoken

from dao_bridge.config import AppConfig, ChunkingConfig
from dao_bridge.schemas import Chunk, Manifest, ManifestItem
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
    atomic_write,
    chunk_dir,
    chunk_path,
    clean_path,
    format_chunk_id,
    manifest_path,
    pad_spine,
)

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Tokeniser (cached)
# ---------------------------------------------------------------------------

_tokeniser: tiktoken.Encoding | None = None


def _get_tokeniser() -> tiktoken.Encoding:
    global _tokeniser
    if _tokeniser is None:
        _tokeniser = tiktoken.get_encoding("cl100k_base")
    return _tokeniser


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_get_tokeniser().encode(text))


# ---------------------------------------------------------------------------
# Block model (internal — not exposed in schemas.py)
# ---------------------------------------------------------------------------

BlockKind = Literal["paragraph", "scene_break", "heading", "hr"]


@dataclass
class Block:
    """Atomic unit the chunker works with — never split across chunks."""

    index: int
    kind: BlockKind
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# HR detection
# ---------------------------------------------------------------------------

_HR_RE = re.compile(r"^(\s*[-*_]\s*){3,}$")


def _is_hr(line: str) -> bool:
    """Return True if *line* is a markdown horizontal rule."""
    return bool(_HR_RE.match(line.strip()))


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


def _is_heading(line: str) -> bool:
    """Return True if *line* is an ATX-style markdown heading."""
    return line.lstrip().startswith("#")


# ---------------------------------------------------------------------------
# Scene break detection
# ---------------------------------------------------------------------------


def _is_scene_break(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return True if *text* matches any configured scene break pattern."""
    stripped = text.strip()
    return any(p.match(stripped) for p in patterns)


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------


def parse_blocks(
    md_text: str,
    config: ChunkingConfig,
) -> list[Block]:
    """Parse cleaned markdown into a list of :class:`Block` objects.

    A block is the atomic unit the chunker works with.  Block types:

    - ``paragraph`` — run of non-empty lines separated by blank lines.
    - ``scene_break`` — paragraph matching a ``scene_break_patterns`` regex.
    - ``heading`` — ATX-style heading (line starting with ``#``).
    - ``hr`` — horizontal rule (``---``, ``***``, ``___`` alone).

    Scene break and HR blocks have their ``text`` replaced with the
    normalised form if ``normalize_scene_breaks`` is set.
    """
    # Pre-compile scene break patterns.
    sb_patterns = [re.compile(p) for p in config.scene_break_patterns]
    normalise_to = config.normalize_scene_breaks

    lines = md_text.split("\n")
    blocks: list[Block] = []
    current_lines: list[str] = []

    def _flush_paragraph() -> None:
        """Emit accumulated lines as a paragraph (or scene_break) block."""
        if not current_lines:
            return
        text = "\n".join(current_lines)
        kind: BlockKind = "paragraph"
        if _is_scene_break(text, sb_patterns):
            kind = "scene_break"
            if normalise_to is not None:
                text = normalise_to
        blocks.append(
            Block(
                index=len(blocks),
                kind=kind,
                text=text,
                token_count=count_tokens(text),
            )
        )
        current_lines.clear()

    for line in lines:
        stripped = line.strip()

        # Blank line — flush current paragraph.
        if not stripped:
            _flush_paragraph()
            continue

        # Heading — flush current paragraph, emit heading as own block.
        if _is_heading(line):
            _flush_paragraph()
            blocks.append(
                Block(
                    index=len(blocks),
                    kind="heading",
                    text=line.rstrip(),
                    token_count=count_tokens(line.rstrip()),
                )
            )
            continue

        # HR — flush current paragraph, emit HR as own block.
        if _is_hr(line):
            _flush_paragraph()
            hr_text = normalise_to if normalise_to is not None else line.rstrip()
            blocks.append(
                Block(
                    index=len(blocks),
                    kind="hr",
                    text=hr_text,
                    token_count=count_tokens(hr_text),
                )
            )
            continue

        # Normal line — accumulate into current paragraph.
        current_lines.append(line.rstrip())

    # Flush any remaining lines.
    _flush_paragraph()

    return blocks


# ---------------------------------------------------------------------------
# Break-point finder for the flex window
# ---------------------------------------------------------------------------


def find_last_break_point_in_range(
    blocks: list[Block],
    min_cumulative_tokens: float,
    max_cumulative_tokens: float,
) -> int | None:
    """Find the latest natural break point within the token range.

    Scans *blocks* for the latest block whose ``kind`` is ``scene_break``,
    ``heading``, or ``hr`` AND whose cumulative token position (sum of
    token_counts from block 0 through this block, inclusive) falls between
    *min_cumulative_tokens* and *max_cumulative_tokens*.

    Returns the index within *blocks*, or ``None`` if no such break point
    exists.
    """
    _BREAK_KINDS: set[BlockKind] = {"scene_break", "heading", "hr"}
    best: int | None = None
    cumulative = 0

    for i, block in enumerate(blocks):
        cumulative += block.token_count
        if cumulative > max_cumulative_tokens:
            break
        if cumulative >= min_cumulative_tokens and block.kind in _BREAK_KINDS:
            best = i

    return best


# ---------------------------------------------------------------------------
# Chunk emission helper
# ---------------------------------------------------------------------------


def _make_chunk(
    blocks: list[Block],
    chunk_index: int,
    spine_index: int,
    source_file: str,
    *,
    extended_for_remainder: bool = False,
) -> Chunk:
    """Build a :class:`Chunk` from a list of blocks."""
    text = "\n\n".join(b.text for b in blocks)
    last_kind = blocks[-1].kind if blocks else "paragraph"
    return Chunk(
        chunk_id=format_chunk_id(spine_index, chunk_index),
        spine_index=spine_index,
        chunk_index=chunk_index,
        source_file=source_file,
        block_range=(blocks[0].index, blocks[-1].index),
        token_count=count_tokens(text),
        extended_for_remainder=extended_for_remainder,
        text=text,
        ends_at_scene_break=last_kind in ("scene_break", "hr"),
    )


# ---------------------------------------------------------------------------
# Greedy packing algorithm
# ---------------------------------------------------------------------------


def chunk_blocks(
    blocks: list[Block],
    config: ChunkingConfig,
    spine_index: int,
    source_file: str,
) -> list[Chunk]:
    """Pack *blocks* into chunks using the greedy algorithm.

    Parameters
    ----------
    blocks:
        Parsed blocks from :func:`parse_blocks`.
    config:
        Chunking configuration (target_tokens, max_tokens, etc.).
    spine_index:
        Spine index for chunk ID generation.
    source_file:
        Path to the clean markdown file (stored in each chunk).

    Returns
    -------
    list[Chunk]
        Deterministic sequence of chunks.
    """
    if not blocks:
        return []

    target_tokens = config.target_tokens
    min_chunk_tokens = config.min_chunk_tokens
    flex_min = target_tokens * (1 - config.flex_window_ratio)

    chunks: list[Chunk] = []
    current_blocks: list[Block] = []
    current_tokens = 0
    chunk_index = 1

    for block in blocks:
        if current_blocks and current_tokens + block.token_count > target_tokens:
            # Current accumulation would exceed target — time to emit.
            break_idx = find_last_break_point_in_range(
                current_blocks,
                min_cumulative_tokens=flex_min,
                max_cumulative_tokens=target_tokens,
            )
            if break_idx is not None:
                emit_blocks = current_blocks[: break_idx + 1]
                leftover = current_blocks[break_idx + 1 :]
                chunks.append(
                    _make_chunk(
                        emit_blocks,
                        chunk_index,
                        spine_index,
                        source_file,
                    )
                )
                current_blocks = leftover + [block]
                current_tokens = sum(b.token_count for b in current_blocks)
            else:
                chunks.append(
                    _make_chunk(
                        current_blocks,
                        chunk_index,
                        spine_index,
                        source_file,
                    )
                )
                current_blocks = [block]
                current_tokens = block.token_count
            chunk_index += 1
        else:
            current_blocks.append(block)
            current_tokens += block.token_count

    # Handle final accumulated chunk.
    if current_blocks:
        if chunks and current_tokens < min_chunk_tokens:
            # Absorb into previous chunk.
            previous = chunks.pop()
            # Reconstruct combined blocks from previous chunk's block range.
            prev_start = previous.block_range[0]
            prev_end = previous.block_range[1]
            # Find original blocks for the previous chunk.
            prev_blocks = [b for b in blocks if prev_start <= b.index <= prev_end]
            combined = prev_blocks + current_blocks
            chunks.append(
                _make_chunk(
                    combined,
                    previous.chunk_index,
                    spine_index,
                    source_file,
                    extended_for_remainder=True,
                )
            )
        else:
            chunks.append(
                _make_chunk(
                    current_blocks,
                    chunk_index,
                    spine_index,
                    source_file,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ChunkValidationError(Exception):
    """Raised when chunk validation fails before writing."""


def validate_chunks(blocks: list[Block], chunks: list[Chunk]) -> None:
    """Validate chunk coverage and consistency.

    Raises :class:`ChunkValidationError` if any check fails.
    """
    if not blocks and not chunks:
        return  # Empty file — zero blocks, zero chunks — OK.

    if not chunks:
        raise ChunkValidationError("Blocks exist but no chunks were produced")

    # 1. Every chunk has at least one block.
    for c in chunks:
        if c.block_range[0] > c.block_range[1]:
            raise ChunkValidationError(
                f"Chunk {c.chunk_id} has invalid block_range: {c.block_range}"
            )

    # 2. Chunk indices are sequential starting from 1.
    expected_idx = 1
    for c in chunks:
        if c.chunk_index != expected_idx:
            raise ChunkValidationError(
                f"Expected chunk_index {expected_idx}, got {c.chunk_index} for chunk {c.chunk_id}"
            )
        expected_idx += 1

    # 3. No gaps or overlaps in block coverage.
    all_covered: set[int] = set()
    for c in chunks:
        for bi in range(c.block_range[0], c.block_range[1] + 1):
            if bi in all_covered:
                raise ChunkValidationError(
                    f"Block {bi} appears in multiple chunks (found in {c.chunk_id})"
                )
            all_covered.add(bi)

    expected_indices = set(range(len(blocks)))
    if all_covered != expected_indices:
        missing = expected_indices - all_covered
        extra = all_covered - expected_indices
        parts = []
        if missing:
            parts.append(f"missing blocks: {sorted(missing)}")
        if extra:
            parts.append(f"extra blocks: {sorted(extra)}")
        raise ChunkValidationError(f"Block coverage mismatch: {', '.join(parts)}")

    # 4. Token count consistency (approximate).
    total_block_tokens = sum(b.token_count for b in blocks)
    total_chunk_tokens = sum(c.token_count for c in chunks)
    # The join separator (\n\n) between blocks within a chunk can add or
    # change token counts vs summing individual block counts.  Each join
    # may add 1-2 tokens.  Allow generous tolerance.
    n_joins = sum(max(0, c.block_range[1] - c.block_range[0]) for c in chunks)
    tolerance = max(n_joins * 2 + len(chunks) * 2, 10)
    if abs(total_chunk_tokens - total_block_tokens) > tolerance:
        raise ChunkValidationError(
            f"Token count mismatch: blocks total {total_block_tokens}, "
            f"chunks total {total_chunk_tokens} (tolerance: ±{tolerance})"
        )


# ---------------------------------------------------------------------------
# Single spine item chunking
# ---------------------------------------------------------------------------


def chunk_spine_item(
    work_dir: Path,
    item: ManifestItem,
    config: ChunkingConfig,
) -> int:
    """Chunk a single spine item, writing chunk files to disk.

    Parameters
    ----------
    work_dir:
        Work directory root.
    item:
        Manifest item to chunk.
    config:
        Chunking configuration.

    Returns
    -------
    int
        Number of chunks produced.
    """
    cp = clean_path(work_dir, item.spine_index)
    if not cp.exists():
        raise FileNotFoundError(f"Clean file missing for spine {item.padded_id}: {cp}")

    md_text = cp.read_text(encoding="utf-8")
    blocks = parse_blocks(md_text, config)

    if not blocks:
        logger.warning("Spine %s: empty file, producing zero chunks", item.padded_id)
        return 0

    # Check for oversized single blocks.
    for b in blocks:
        if b.token_count > config.max_tokens:
            logger.warning(
                "Spine %s: block %d has %d tokens (exceeds max_tokens=%d). "
                "This block will become its own oversized chunk.",
                item.padded_id,
                b.index,
                b.token_count,
                config.max_tokens,
            )

    # Use forward slashes for cross-platform consistency in stored paths.
    source_file = cp.relative_to(work_dir).as_posix()
    chunks = chunk_blocks(blocks, config, item.spine_index, source_file)
    validate_chunks(blocks, chunks)

    # Write chunk files.
    cd = chunk_dir(work_dir, item.spine_index)
    cd.mkdir(parents=True, exist_ok=True)

    for c in chunks:
        cp_out = chunk_path(work_dir, c.chunk_id)
        atomic_write(cp_out, c.model_dump_json(indent=2))

    return len(chunks)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def chunk_all(
    config: AppConfig,
    manifest: Manifest,
    state: PipelineState,
    *,
    force: bool = False,
    spine_filter: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Manifest:
    """Chunk all eligible spine items.

    Parameters
    ----------
    config:
        Application configuration.
    manifest:
        The manifest (mutated in place with ``chunk_count``).
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, rechunk even if already completed.
    spine_filter:
        If set, only chunk this spine index.
    on_progress:
        Optional callback invoked with the padded spine ID after each
        item is processed (chunked, skipped, or already-completed).

    Returns
    -------
    Manifest
        The updated manifest.

    Raises
    ------
    RuntimeError
        If any spine item has ``classification is None`` (classify not run).
    """
    work_dir = config.work_dir_path
    chunk_cfg = config.chunking
    chunkable = set(chunk_cfg.chunkable_classifications)

    # Gate: classification must be set for all items.
    unclassified = [item for item in manifest.spine if item.classification is None]
    if unclassified:
        ids = ", ".join(item.padded_id for item in unclassified)
        raise RuntimeError(
            f"Classification required before chunking. "
            f"Run `dao-bridge classify` first. "
            f"Unclassified items: {ids}"
        )

    # Determine which items to process.
    if spine_filter is not None:
        items = [item for item in manifest.spine if item.spine_index == spine_filter]
        if not items:
            raise ValueError(f"Spine index {spine_filter} not found in manifest")
    else:
        items = list(manifest.spine)

    if force:
        reset_stage(work_dir, state, "chunk")

    if not force and is_stage_completed(state, "chunk") and spine_filter is None:
        logger.info("Chunk stage already completed — skipping (use --force to re-run)")
        return manifest

    mark_stage_started(work_dir, state, "chunk")

    # Build list of item IDs for pending check.
    item_ids = [pad_spine(item.spine_index) for item in items]
    pending = set(iter_pending_items(state, "chunk", item_ids))

    skipped = 0
    chunked = 0
    total_chunks = 0

    for item in items:
        padded = pad_spine(item.spine_index)

        # Skip if already completed (unless force).
        if not force and padded not in pending:
            if on_progress:
                on_progress(padded)
            continue

        # Classification filtering.
        classification = item.classification
        if classification == "unknown":
            logger.warning(
                "Spine %s has classification 'unknown' — treating as chunkable",
                padded,
            )
        elif classification not in chunkable:
            # Not chunkable — skip.
            item.chunk_count = 0
            mark_item_started(work_dir, state, "chunk", padded)
            mark_item_completed(work_dir, state, "chunk", padded)
            skipped += 1
            logger.debug(
                "Spine %s: classification '%s' not chunkable, skipping",
                padded,
                classification,
            )
            if on_progress:
                on_progress(padded)
            continue

        mark_item_started(work_dir, state, "chunk", padded)

        try:
            # Delete existing chunks if forcing.
            if force:
                cd = chunk_dir(work_dir, item.spine_index)
                if cd.exists():
                    shutil.rmtree(cd)

            n_chunks = chunk_spine_item(work_dir, item, chunk_cfg)
            item.chunk_count = n_chunks
            mark_item_completed(work_dir, state, "chunk", padded)
            chunked += 1
            total_chunks += n_chunks
            logger.debug(
                "Spine %s: produced %d chunk(s)",
                padded,
                n_chunks,
            )
        except Exception as exc:
            mark_item_failed(work_dir, state, "chunk", padded, str(exc))
            raise

        if on_progress:
            on_progress(padded)

    # Persist updated manifest.
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))

    # Mark stage complete only if processing all items (not a single --spine).
    if spine_filter is None:
        mark_stage_completed(work_dir, state, "chunk")

    logger.info(
        "Chunking complete: %d items chunked (%d chunks), %d items skipped",
        chunked,
        total_chunks,
        skipped,
    )

    return manifest
