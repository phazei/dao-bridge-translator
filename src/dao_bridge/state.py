"""JSON-backed pipeline state tracking.

The state file (``state.json``) lives in the work directory and records
which stages and individual items have been processed.  All writes go
through :func:`~dao_bridge.workdir.atomic_write` so readers never see
partially-written data.

All operations are **idempotent** — marking something completed twice is
a no-op.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from dao_bridge.workdir import atomic_write, state_path

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------

STAGE_NAMES = (
    "extract",
    "clean",
    "classify",
    "chunk",
    "glossary_build",
    "glossary_reconcile",
    "glossary_crosscheck",
    "translate",
    "assemble",
    "rebuild",
)

StageName = Literal[
    "extract",
    "clean",
    "classify",
    "chunk",
    "glossary_build",
    "glossary_reconcile",
    "glossary_crosscheck",
    "translate",
    "assemble",
    "rebuild",
]

ItemStatus = Literal["pending", "started", "completed", "failed", "failed_qa"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunState(BaseModel):
    """Top-level run metadata."""

    source_epub: str = ""
    started_at: str = ""
    status: str = "idle"


class StageState(BaseModel):
    """Status of a single pipeline stage."""

    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


class ItemState(BaseModel):
    """Status of a single work item (spine item, chunk, batch)."""

    status: ItemStatus = "pending"
    completed_at: str | None = None
    error_message: str | None = None


class PipelineState(BaseModel):
    """Complete pipeline state, persisted to ``state.json``."""

    run: RunState = Field(default_factory=RunState)
    stages: dict[str, StageState] = Field(default_factory=dict)
    items: dict[str, ItemState] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_state(work_dir: Path) -> PipelineState:
    """Load state from disk, returning a fresh state if the file is missing."""
    path = state_path(work_dir)
    if not path.exists():
        return PipelineState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PipelineState(**raw)


def save_state(work_dir: Path, state: PipelineState) -> None:
    """Persist *state* to ``state.json`` atomically."""
    path = state_path(work_dir)
    atomic_write(path, state.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def mark_stage_started(work_dir: Path, state: PipelineState, stage: StageName) -> None:
    """Mark *stage* as running.  Idempotent if already started/completed."""
    info = state.stages.get(stage)
    if info and info.status in ("running", "completed"):
        return
    state.stages[stage] = StageState(status="running", started_at=_now())
    save_state(work_dir, state)
    logger.info("Stage [bold]%s[/bold] started", stage)


def mark_stage_completed(work_dir: Path, state: PipelineState, stage: StageName) -> None:
    """Mark *stage* as completed.  Idempotent if already completed."""
    info = state.stages.get(stage)
    if info and info.status == "completed":
        return
    if stage not in state.stages:
        state.stages[stage] = StageState()
    state.stages[stage].status = "completed"
    state.stages[stage].completed_at = _now()
    save_state(work_dir, state)
    logger.info("Stage [bold]%s[/bold] completed", stage)


def mark_stage_failed(
    work_dir: Path, state: PipelineState, stage: StageName, error: str = ""
) -> None:
    """Mark *stage* as failed."""
    if stage not in state.stages:
        state.stages[stage] = StageState()
    state.stages[stage].status = "failed"
    state.stages[stage].error_message = error
    save_state(work_dir, state)
    logger.error("Stage [bold]%s[/bold] failed: %s", stage, error)


def reset_stage(work_dir: Path, state: PipelineState, stage: StageName) -> None:
    """Reset *stage* and its items so it can be re-run cleanly.

    Clears the stage status back to ``"pending"`` and removes all item
    entries belonging to the stage.  Used by ``--force`` to ensure the
    state file stays consistent while data is being rewritten.
    """
    state.stages[stage] = StageState(status="pending")
    prefix = f"{stage}:"
    state.items = {k: v for k, v in state.items.items() if not k.startswith(prefix)}
    save_state(work_dir, state)
    logger.info("Stage [bold]%s[/bold] reset for re-run", stage)


def reset_stage_items(
    work_dir: Path,
    state: PipelineState,
    stage: StageName,
    item_ids: list[str],
) -> None:
    """Reset specific items within a stage so they can be retranslated.

    Unlike :func:`reset_stage`, this preserves the stage status and all
    item entries *not* in *item_ids*.  The stage status is set to
    ``"running"`` so processing can proceed.
    """
    for item_id in item_ids:
        key = _item_key(stage, item_id)
        state.items.pop(key, None)
    # Ensure the stage is in a runnable state.
    if stage not in state.stages or state.stages[stage].status in ("completed", "failed"):
        state.stages[stage] = StageState(status="running", started_at=_now())
    save_state(work_dir, state)
    logger.info(
        "Stage [bold]%s[/bold]: reset %d item(s) for re-run",
        stage,
        len(item_ids),
    )


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------


def _item_key(stage: StageName, item_id: str) -> str:
    return f"{stage}:{item_id}"


def mark_item_started(work_dir: Path, state: PipelineState, stage: StageName, item_id: str) -> None:
    """Mark an individual item as started.  Idempotent if already started/completed."""
    key = _item_key(stage, item_id)
    existing = state.items.get(key)
    if existing and existing.status in ("started", "completed"):
        return
    state.items[key] = ItemState(status="started")
    save_state(work_dir, state)


def mark_item_completed(
    work_dir: Path, state: PipelineState, stage: StageName, item_id: str
) -> None:
    """Mark an individual item as completed.  Idempotent."""
    key = _item_key(stage, item_id)
    existing = state.items.get(key)
    if existing and existing.status == "completed":
        return
    state.items[key] = ItemState(status="completed", completed_at=_now())
    save_state(work_dir, state)


def mark_item_failed(
    work_dir: Path,
    state: PipelineState,
    stage: StageName,
    item_id: str,
    error: str = "",
    *,
    status: ItemStatus = "failed",
) -> None:
    """Mark an individual item as failed (or failed_qa)."""
    key = _item_key(stage, item_id)
    state.items[key] = ItemState(status=status, error_message=error, completed_at=_now())
    save_state(work_dir, state)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def is_stage_completed(state: PipelineState, stage: StageName) -> bool:
    """Return *True* if *stage* has status ``"completed"``."""
    info = state.stages.get(stage)
    return info is not None and info.status == "completed"


def iter_pending_items(state: PipelineState, stage: StageName, item_ids: list[str]) -> list[str]:
    """Return item IDs from *item_ids* that are not yet completed.

    Items with status ``"completed"`` are skipped; everything else
    (pending, started, failed, failed_qa, or not present) is included.
    """
    result = []
    for item_id in item_ids:
        key = _item_key(stage, item_id)
        existing = state.items.get(key)
        if existing and existing.status == "completed":
            continue
        result.append(item_id)
    return result


def has_failed_items(state: PipelineState, stage: StageName) -> bool:
    """Return *True* if *stage* has any items with ``failed`` or ``failed_qa`` status.

    Useful for ``--retry-failed`` to check whether there is work to do
    before re-entering a completed stage.
    """
    prefix = f"{stage}:"
    for key, item in state.items.items():
        if key.startswith(prefix) and item.status in ("failed", "failed_qa"):
            return True
    return False


def reopen_stage(work_dir: Path, state: PipelineState, stage: StageName) -> None:
    """Re-open a completed stage so failed items can be retried.

    Unlike :func:`reset_stage` (used by ``--force``), this preserves all
    existing item state — completed items stay completed.  Only the stage-
    level status is set back to ``"running"`` so that ``iter_pending_items``
    based processing can resume for non-completed items.

    If the stage is already ``"running"`` or ``"pending"``, this is a no-op.
    """
    info = state.stages.get(stage)
    if info and info.status in ("running", "pending"):
        return
    state.stages[stage] = StageState(status="running", started_at=_now())
    save_state(work_dir, state)
    logger.info("Stage [bold]%s[/bold] reopened for retry-failed", stage)
