"""Tests for dao_bridge.state — JSON-backed pipeline state tracking."""

import json
from pathlib import Path

import pytest

from dao_bridge.state import (
    PipelineState,
    StageState,
    has_failed_items,
    is_stage_completed,
    iter_pending_items,
    load_state,
    mark_item_completed,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_started,
    reopen_stage,
    reset_stage,
    save_state,
)
from dao_bridge.workdir import ensure_dirs, state_path


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    wd = tmp_path / "work"
    ensure_dirs(wd)
    return wd


# ---------------------------------------------------------------------------
# Load / save round-trip
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_fresh_state_when_missing(self, work_dir: Path):
        state = load_state(work_dir)
        assert state.run.status == "idle"
        assert state.stages == {}
        assert state.items == {}

    def test_save_and_reload(self, work_dir: Path):
        state = PipelineState()
        state.stages["extract"] = StageState(status="completed", completed_at="2025-01-01T00:00:00")
        save_state(work_dir, state)

        reloaded = load_state(work_dir)
        assert reloaded.stages["extract"].status == "completed"

    def test_json_valid(self, work_dir: Path):
        state = PipelineState()
        mark_stage_started(work_dir, state, "extract")
        raw = json.loads(state_path(work_dir).read_text(encoding="utf-8"))
        assert "run" in raw
        assert "stages" in raw
        assert raw["stages"]["extract"]["status"] == "running"


# ---------------------------------------------------------------------------
# Stage operations
# ---------------------------------------------------------------------------


class TestStageOperations:
    def test_mark_stage_started(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        assert state.stages["extract"].status == "running"
        assert state.stages["extract"].started_at is not None

    def test_mark_stage_completed(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "clean")
        mark_stage_completed(work_dir, state, "clean")
        assert state.stages["clean"].status == "completed"
        assert state.stages["clean"].completed_at is not None

    def test_mark_stage_failed(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "translate")
        mark_stage_failed(work_dir, state, "translate", "connection refused")
        assert state.stages["translate"].status == "failed"
        assert state.stages["translate"].error_message == "connection refused"

    def test_is_stage_completed(self, work_dir: Path):
        state = load_state(work_dir)
        assert not is_stage_completed(state, "extract")
        mark_stage_started(work_dir, state, "extract")
        assert not is_stage_completed(state, "extract")
        mark_stage_completed(work_dir, state, "extract")
        assert is_stage_completed(state, "extract")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_double_complete_stage_is_noop(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_stage_completed(work_dir, state, "extract")
        ts1 = state.stages["extract"].completed_at
        mark_stage_completed(work_dir, state, "extract")
        ts2 = state.stages["extract"].completed_at
        assert ts1 == ts2  # timestamp unchanged — was a no-op

    def test_double_start_stage_is_noop(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "clean")
        ts1 = state.stages["clean"].started_at
        mark_stage_started(work_dir, state, "clean")
        ts2 = state.stages["clean"].started_at
        assert ts1 == ts2

    def test_double_complete_item_is_noop(self, work_dir: Path):
        state = load_state(work_dir)
        mark_item_started(work_dir, state, "extract", "001")
        mark_item_completed(work_dir, state, "extract", "001")
        ts1 = state.items["extract:001"].completed_at
        mark_item_completed(work_dir, state, "extract", "001")
        ts2 = state.items["extract:001"].completed_at
        assert ts1 == ts2

    def test_start_completed_stage_is_noop(self, work_dir: Path):
        """Starting a stage that is already completed should be a no-op."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_stage_completed(work_dir, state, "extract")
        mark_stage_started(work_dir, state, "extract")
        assert state.stages["extract"].status == "completed"


# ---------------------------------------------------------------------------
# Item operations
# ---------------------------------------------------------------------------


class TestItemOperations:
    def test_mark_item_lifecycle(self, work_dir: Path):
        state = load_state(work_dir)
        mark_item_started(work_dir, state, "extract", "003")
        assert state.items["extract:003"].status == "started"
        mark_item_completed(work_dir, state, "extract", "003")
        assert state.items["extract:003"].status == "completed"

    def test_mark_item_failed(self, work_dir: Path):
        state = load_state(work_dir)
        mark_item_failed(work_dir, state, "translate", "003.015", "QA check failed")
        assert state.items["translate:003.015"].status == "failed"
        assert state.items["translate:003.015"].error_message == "QA check failed"

    def test_mark_item_failed_qa(self, work_dir: Path):
        state = load_state(work_dir)
        mark_item_failed(
            work_dir, state, "translate", "003.015", "length ratio", status="failed_qa"
        )
        assert state.items["translate:003.015"].status == "failed_qa"

    def test_iter_pending_items(self, work_dir: Path):
        state = load_state(work_dir)
        all_ids = ["001", "002", "003", "004"]
        mark_item_completed(work_dir, state, "extract", "001")
        mark_item_completed(work_dir, state, "extract", "003")
        pending = iter_pending_items(state, "extract", all_ids)
        assert pending == ["002", "004"]

    def test_iter_pending_includes_failed(self, work_dir: Path):
        """Failed items should be returned by iter_pending_items."""
        state = load_state(work_dir)
        mark_item_failed(work_dir, state, "translate", "002", "error")
        pending = iter_pending_items(state, "translate", ["001", "002", "003"])
        assert "002" in pending  # failed items should be retried


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_partial_stage_recovery(self, work_dir: Path):
        """Simulate a crash mid-stage: stage is running, some items complete."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_item_completed(work_dir, state, "extract", "001")
        mark_item_completed(work_dir, state, "extract", "002")
        # "Crash" — reload from disk
        reloaded = load_state(work_dir)
        assert reloaded.stages["extract"].status == "running"
        assert reloaded.items["extract:001"].status == "completed"
        assert reloaded.items["extract:002"].status == "completed"
        # Items 003+ are not present — iter_pending handles this
        pending = iter_pending_items(reloaded, "extract", ["001", "002", "003", "004"])
        assert pending == ["003", "004"]

    def test_state_file_persists_across_reloads(self, work_dir: Path):
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "clean")
        mark_item_completed(work_dir, state, "clean", "005")
        # Reload completely
        state2 = load_state(work_dir)
        assert state2.stages["clean"].status == "running"
        assert state2.items["clean:005"].status == "completed"


# ---------------------------------------------------------------------------
# Reset stage (for --force)
# ---------------------------------------------------------------------------


class TestResetStage:
    def test_reset_clears_status_and_items(self, work_dir: Path):
        """reset_stage should set stage to pending and remove its items."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_item_completed(work_dir, state, "extract", "001")
        mark_item_completed(work_dir, state, "extract", "002")
        mark_item_completed(work_dir, state, "extract", "003")
        mark_stage_completed(work_dir, state, "extract")

        assert is_stage_completed(state, "extract")
        assert len([k for k in state.items if k.startswith("extract:")]) == 3

        reset_stage(work_dir, state, "extract")

        assert state.stages["extract"].status == "pending"
        assert state.stages["extract"].completed_at is None
        assert len([k for k in state.items if k.startswith("extract:")]) == 0

    def test_reset_preserves_other_stages(self, work_dir: Path):
        """Resetting one stage should not affect other stages or their items."""
        state = load_state(work_dir)
        # Set up extract as completed.
        mark_stage_started(work_dir, state, "extract")
        mark_item_completed(work_dir, state, "extract", "001")
        mark_stage_completed(work_dir, state, "extract")
        # Set up clean as completed.
        mark_stage_started(work_dir, state, "clean")
        mark_item_completed(work_dir, state, "clean", "001")
        mark_stage_completed(work_dir, state, "clean")

        reset_stage(work_dir, state, "extract")

        # Extract reset.
        assert state.stages["extract"].status == "pending"
        assert "extract:001" not in state.items
        # Clean untouched.
        assert state.stages["clean"].status == "completed"
        assert state.items["clean:001"].status == "completed"

    def test_reset_persists_to_disk(self, work_dir: Path):
        """Reset should be persisted so a reload sees the clean state."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_item_completed(work_dir, state, "extract", "001")
        mark_stage_completed(work_dir, state, "extract")

        reset_stage(work_dir, state, "extract")

        reloaded = load_state(work_dir)
        assert reloaded.stages["extract"].status == "pending"
        assert "extract:001" not in reloaded.items

    def test_reset_allows_restart(self, work_dir: Path):
        """After reset, mark_stage_started should work (not be blocked by idempotency)."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "extract")
        mark_stage_completed(work_dir, state, "extract")

        reset_stage(work_dir, state, "extract")
        mark_stage_started(work_dir, state, "extract")

        assert state.stages["extract"].status == "running"
        assert state.stages["extract"].started_at is not None


# ---------------------------------------------------------------------------
# has_failed_items
# ---------------------------------------------------------------------------


class TestHasFailedItems:
    def test_no_items(self, work_dir: Path):
        """No items at all returns False."""
        state = load_state(work_dir)
        assert not has_failed_items(state, "classify")

    def test_all_completed(self, work_dir: Path):
        """All completed items returns False."""
        state = load_state(work_dir)
        mark_item_completed(work_dir, state, "classify", "001")
        mark_item_completed(work_dir, state, "classify", "002")
        assert not has_failed_items(state, "classify")

    def test_has_failed(self, work_dir: Path):
        """A failed item returns True."""
        state = load_state(work_dir)
        mark_item_completed(work_dir, state, "classify", "001")
        mark_item_failed(work_dir, state, "classify", "002", "LLM error")
        assert has_failed_items(state, "classify")

    def test_has_failed_qa(self, work_dir: Path):
        """A failed_qa item returns True."""
        state = load_state(work_dir)
        mark_item_failed(work_dir, state, "translate", "001.002", "QA fail", status="failed_qa")
        assert has_failed_items(state, "translate")

    def test_other_stage_not_affected(self, work_dir: Path):
        """Failed items in one stage don't affect another."""
        state = load_state(work_dir)
        mark_item_failed(work_dir, state, "classify", "001", "error")
        assert not has_failed_items(state, "chunk")


# ---------------------------------------------------------------------------
# reopen_stage (for --retry-failed)
# ---------------------------------------------------------------------------


class TestReopenStage:
    def test_reopen_completed_stage(self, work_dir: Path):
        """reopen_stage sets a completed stage back to running."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "classify")
        mark_item_completed(work_dir, state, "classify", "001")
        mark_item_failed(work_dir, state, "classify", "002", "error")
        mark_stage_completed(work_dir, state, "classify")

        reopen_stage(work_dir, state, "classify")

        assert state.stages["classify"].status == "running"
        # Items are preserved.
        assert state.items["classify:001"].status == "completed"
        assert state.items["classify:002"].status == "failed"

    def test_reopen_preserves_completed_items(self, work_dir: Path):
        """reopen_stage does NOT remove any item entries."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "chunk")
        mark_item_completed(work_dir, state, "chunk", "001")
        mark_item_completed(work_dir, state, "chunk", "002")
        mark_item_failed(work_dir, state, "chunk", "003", "error")
        mark_stage_completed(work_dir, state, "chunk")

        reopen_stage(work_dir, state, "chunk")

        # All items still present.
        assert state.items["chunk:001"].status == "completed"
        assert state.items["chunk:002"].status == "completed"
        assert state.items["chunk:003"].status == "failed"

    def test_reopen_running_is_noop(self, work_dir: Path):
        """reopen_stage on an already running stage is a no-op."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "classify")
        ts = state.stages["classify"].started_at

        reopen_stage(work_dir, state, "classify")

        assert state.stages["classify"].status == "running"
        assert state.stages["classify"].started_at == ts  # unchanged

    def test_reopen_pending_is_noop(self, work_dir: Path):
        """reopen_stage on a pending (never started) stage is a no-op."""
        state = load_state(work_dir)
        state.stages["classify"] = StageState(status="pending")
        save_state(work_dir, state)

        reopen_stage(work_dir, state, "classify")

        assert state.stages["classify"].status == "pending"

    def test_reopen_failed_stage(self, work_dir: Path):
        """reopen_stage on a failed stage sets it to running."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "classify")
        mark_stage_failed(work_dir, state, "classify", "something broke")

        reopen_stage(work_dir, state, "classify")

        assert state.stages["classify"].status == "running"

    def test_reopen_persists_to_disk(self, work_dir: Path):
        """reopen_stage changes are persisted to disk."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "classify")
        mark_item_completed(work_dir, state, "classify", "001")
        mark_item_failed(work_dir, state, "classify", "002", "error")
        mark_stage_completed(work_dir, state, "classify")

        reopen_stage(work_dir, state, "classify")

        reloaded = load_state(work_dir)
        assert reloaded.stages["classify"].status == "running"
        assert reloaded.items["classify:001"].status == "completed"
        assert reloaded.items["classify:002"].status == "failed"

    def test_reopen_then_iter_pending_returns_only_failed(self, work_dir: Path):
        """After reopen, iter_pending_items returns failed items but not completed."""
        state = load_state(work_dir)
        mark_stage_started(work_dir, state, "classify")
        mark_item_completed(work_dir, state, "classify", "001")
        mark_item_completed(work_dir, state, "classify", "002")
        mark_item_failed(work_dir, state, "classify", "003", "error")
        mark_stage_completed(work_dir, state, "classify")

        reopen_stage(work_dir, state, "classify")

        pending = iter_pending_items(state, "classify", ["001", "002", "003", "004"])
        # 001, 002 = completed (skipped); 003 = failed (included); 004 = not present (included)
        assert pending == ["003", "004"]
