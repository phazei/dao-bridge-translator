"""CLI entry point for dao-bridge.

All commands accept ``--verbose`` to enable DEBUG-level console logging.
Commands check state before starting and skip completed work unless ``--force``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import yaml

from dao_bridge.config import AppConfig, load_config
from dao_bridge.logging import _make_utf8_console, setup_logging
from dao_bridge.state import (
    RunState,
    load_state,
    save_state,
)
from dao_bridge.workdir import ensure_dirs, manifest_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _swap_logger_to_progress(progress, verbose: bool = False):
    """Redirect the dao_bridge logger's Rich console handler through *progress*.

    Returns a cleanup callable that restores the original handlers.
    Rich internally coordinates output through the same console, so log
    lines printed via the swapped handler won't fight with the progress bar.
    """
    import logging as _logging

    from rich.logging import RichHandler

    _logger = _logging.getLogger("dao_bridge")
    old_handlers = []
    for h in list(_logger.handlers):
        if isinstance(h, RichHandler):
            old_handlers.append(h)
            _logger.removeHandler(h)
    progress_handler = RichHandler(
        level=_logging.DEBUG if verbose else _logging.INFO,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        console=progress.console,
    )
    progress_handler.setFormatter(_logging.Formatter("%(message)s", datefmt="[%X]"))
    _logger.addHandler(progress_handler)

    def _restore():
        _logger.removeHandler(progress_handler)
        for h in old_handlers:
            _logger.addHandler(h)

    return _restore


def _run_translate_with_progress(
    *,
    work: Path,
    config: AppConfig,
    state,
    manifest,
    force: bool = False,
    retry_failed: bool = False,
    from_chunk: str | None = None,
    to_chunk: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run the translate stage wrapped in a Rich Progress bar.

    Returns the result dict from :func:`run_translate_stage`.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from dao_bridge.translate import TranslationProgress, run_translate_stage

    start_time = time.monotonic()

    progress = Progress(
        TextColumn("Translating"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("{task.fields[spine]}", style="cyan"),
        TextColumn("{task.fields[chunk]}", style="bold"),
        TextColumn("{task.fields[pass_name]}", style="dim"),
        TextColumn("{task.fields[tokens]} tok", style="green"),
        TextColumn("{task.fields[tok_per_sec]}/s", style="green dim"),
        TextColumn("avg {task.fields[avg_time]}", style="dim"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_make_utf8_console(),
        transient=True,
        refresh_per_second=2,
    )

    task_id = None

    def _on_progress(p: TranslationProgress) -> None:
        nonlocal task_id
        if task_id is None:
            return
        spine = p.chunk_id.rsplit(".", 1)[0] if "." in p.chunk_id else p.chunk_id
        elapsed = time.monotonic() - start_time
        avg = f"{elapsed / p.chunks_completed:.1f}s" if p.chunks_completed > 0 else "--"
        tok_sec = f"{p.tokens_so_far / elapsed:.0f}" if elapsed > 0 and p.tokens_so_far else "--"
        progress.update(
            task_id,
            completed=p.chunks_completed,
            total=p.chunks_total,
            spine=f"spine {spine}",
            chunk=p.chunk_id,
            pass_name=p.pass_name,
            tokens=f"{p.tokens_so_far:,}",
            tok_per_sec=tok_sec,
            avg_time=avg,
        )

    with progress:
        restore = _swap_logger_to_progress(progress, verbose=verbose)
        try:
            task_id = progress.add_task(
                "Translating",
                total=0,
                spine="",
                chunk="",
                pass_name="",
                tokens="0",
                tok_per_sec="--",
                avg_time="--",
            )
            result = run_translate_stage(
                work_dir=work,
                config=config,
                state=state,
                manifest=manifest,
                force=force,
                retry_failed=retry_failed,
                from_chunk=from_chunk,
                to_chunk=to_chunk,
                on_progress=_on_progress,
            )
        finally:
            restore()

    return result


def _run_glossary_build_with_progress(
    *,
    work: Path,
    config: AppConfig,
    state,
    force: bool = False,
    retry_failed: bool = False,
    force_summaries: bool = False,
    target_spine: int | None = None,
    target_batch: str | None = None,
    verbose: bool = False,
):
    """Run glossary-build wrapped in a Rich Progress bar.

    The single task follows the whole stage across both sub-phases: extraction
    batches (``phase="extract"``) and the deferred summary-compression pass
    (``phase="compress"``).  On a phase switch the task is reset and relabelled
    ("Building glossary" -> "Compressing summaries") with a fresh total.

    ``force_summaries`` runs the Phase 2B recompression-only path through the
    same bar (it reports only the ``"compress"`` phase).

    Returns the :class:`Glossary` from :func:`glossary_build`.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from dao_bridge.glossary import GlossaryBuildProgress, glossary_build

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("{task.fields[item]}", style="bold"),
        TextColumn("{task.fields[of_batch]}", style="cyan"),
        TimeElapsedColumn(),
        console=_make_utf8_console(),
        transient=True,
        refresh_per_second=2,
    )

    current_phase: list[str | None] = [None]

    def _on_progress(p: GlossaryBuildProgress) -> None:
        if task_id is None:
            return
        # Compression reports entity IDs (no ".bN" suffix); extraction reports
        # spine-aligned batch IDs (e.g. "0003.b2").
        if p.phase == "extract":
            _dot, batch_part = p.item_id.rsplit(".", 1)
            of_batch = f"of {batch_part[0]}{p.spine_batch_count}"
        else:
            of_batch = ""

        if p.phase != current_phase[0]:
            # Phase switched: reset the task for the new phase.  Rich's reset()
            # CLEARS every custom field, so each one (item, of_batch) must be
            # re-supplied here or the next render raises KeyError.
            current_phase[0] = p.phase
            progress.reset(
                task_id,
                total=p.items_total,
                completed=1,
                description=p.phase_label,
                item=p.item_id,
                of_batch=of_batch,
            )
        else:
            progress.update(
                task_id,
                total=p.items_total,
                advance=1,
                item=p.item_id,
                of_batch=of_batch,
            )

    task_id = None
    with progress:
        restore = _swap_logger_to_progress(progress, verbose=verbose)
        try:
            task_id = progress.add_task(
                "Building glossary",
                total=None,
                item="",
                of_batch="",
            )
            glossary = glossary_build(
                work,
                config,
                state,
                force=force,
                retry_failed=retry_failed,
                force_summaries=force_summaries,
                target_spine=target_spine,
                target_batch=target_batch,
                on_progress=_on_progress,
            )
        finally:
            restore()

    return glossary


def _run_glossary_reconcile_with_progress(
    *,
    work: Path,
    config: AppConfig,
    state,
    force: bool = False,
    retry_failed: bool = False,
    verbose: bool = False,
):
    """Run glossary-reconcile wrapped in a Rich Progress bar.

    Returns the :class:`Glossary` from :func:`glossary_reconcile`.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from dao_bridge.glossary import GlossaryReconcileProgress, glossary_reconcile

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("{task.fields[item]}", style="bold"),
        TimeElapsedColumn(),
        console=_make_utf8_console(),
        transient=True,
        refresh_per_second=2,
    )

    current_phase: list[str | None] = [None]

    def _on_progress(p: GlossaryReconcileProgress) -> None:
        if task_id is None:
            return
        if p.phase != current_phase[0]:
            # Phase changed — reset the task for the new phase.
            current_phase[0] = p.phase
            progress.update(
                task_id,
                description=p.phase_label,
                completed=p.completed,
                total=p.total,
                item=p.item_label,
            )
        else:
            progress.update(
                task_id,
                completed=p.completed,
                item=p.item_label,
            )

    task_id = None
    with progress:
        restore = _swap_logger_to_progress(progress, verbose=verbose)
        try:
            task_id = progress.add_task(
                "Reconciling glossary...",
                total=None,
                item="",
            )
            glossary = glossary_reconcile(
                work,
                config,
                state,
                force=force,
                retry_failed=retry_failed,
                on_progress=_on_progress,
            )
        finally:
            restore()

    return glossary


def _run_glossary_cluster_with_progress(
    *,
    work: Path,
    config: AppConfig,
    state,
    force: bool = False,
    retry_failed: bool = False,
    verbose: bool = False,
):
    """Run glossary-cluster wrapped in a Rich Progress bar.

    Clustering nests iterations over batches; neither count is known up front
    (iterations can stop early, batch counts depend on candidates generated each
    iteration).  The single task re-points its total to the current iteration's
    batch count and advances per LLM batch, relabelling "Clustering iterN" as
    iterations progress.  Absolute ``completed``/``total`` are set via
    ``update`` (not ``reset``), so no custom field needs re-supplying.

    Returns the :class:`Glossary` from :func:`glossary_cluster`.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from dao_bridge.glossary import GlossaryClusterProgress, glossary_cluster

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("{task.fields[item]}", style="bold"),
        TimeElapsedColumn(),
        console=_make_utf8_console(),
        transient=True,
        refresh_per_second=2,
    )

    def _on_progress(p: GlossaryClusterProgress) -> None:
        if task_id is None:
            return
        progress.update(
            task_id,
            description=p.phase_label,
            total=p.batches_this_iteration,
            completed=p.batch,
            item=p.item_label,
        )

    task_id = None
    with progress:
        restore = _swap_logger_to_progress(progress, verbose=verbose)
        try:
            task_id = progress.add_task(
                "Clustering glossary entities...",
                total=None,
                item="",
            )
            glossary = glossary_cluster(
                work,
                config,
                state,
                force=force,
                retry_failed=retry_failed,
                on_progress=_on_progress,
            )
        finally:
            restore()

    return glossary


def _resolve_config(work_dir: Path) -> AppConfig:
    """Load config.yaml from the work directory."""
    cfg_path = work_dir / "config.yaml"
    if not cfg_path.exists():
        raise click.ClickException(
            f"Config file not found: {cfg_path}\n"
            "Run 'dao-bridge init <epub>' first to create a work directory."
        )
    return load_config(cfg_path)


def _default_config_yaml(epub_path: str, work_dir: str) -> str:
    """Generate a default config.yaml string for init."""
    cfg = {
        "source_epub": epub_path,
        "work_dir": work_dir,
        "models": {
            "classify": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "model": "qwen3-30b-a3b",
                "temperature": 0.6,
                # Reasoning/thinking control for reasoning models.
                # null => server/app default. For LM Studio, "none" disables
                # thinking entirely. Other values: "low"/"medium"/"high".
                "reasoning_effort": None,
            },
            "glossary": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "model": "gemma-4-26b-a4b",
                "temperature": 0.7,
                "reasoning_effort": None,
            },
            "translate": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "model": "gemma-4-26b-a4b",
                "temperature": 0.9,
                "reasoning_effort": None,
            },
            # "summarize" falls back to "translate" if absent.
            # "summarize": {
            #     "base_url": "http://localhost:8080/v1",
            #     "api_key": "not-needed",
            #     "model": "qwen3-30b-a3b",
            # },
            # "qa" (translation QA judge) falls back to "translate" if absent.
            # Point it at a model that detects defects/omissions well. Set a
            # large "ttl" on both this and "translate" so LM Studio keeps them
            # resident and does not reload on every translate<->QA switch.
            # "qa": {
            #     "base_url": "http://localhost:8080/v1",
            #     "api_key": "not-needed",
            #     "model": "gemma-4-31b-it",
            #     "reasoning_effort": "none",
            #     "ttl": 3600,
            # },
        },
        "chunking": {
            "target_tokens": 2000,
            "max_tokens": 2400,
            "min_chunk_tokens": 400,
            "flex_window_ratio": 0.2,
            "normalize_scene_breaks": "* * *",
        },
        "glossary": {
            "toc_categories": [],
            "master_glossary_path": None,
            "promote_on_complete": False,
            # When True, entity summaries are produced by an LLM compression pass
            # at the tail of glossary-build (O(entities) calls) instead of naive
            # concatenation. Compressed summaries also improve embedding
            # clustering. Default False to keep glossary-build cheap.
            "summary_compress_enabled": False,
            "summary_max_length": 500,
            "cluster": {
                # auto_merge_enabled is only production-safe with
                # embedding_enabled=True. embedding_enabled requires the extra:
                #   pip install dao-bridge-translator[embeddings]
                "auto_merge_enabled": False,
                "embedding_enabled": False,
            },
        },
        "glossary_phase": {
            "target_tokens_per_call": 8000,
            "overlap_chunks": 0,
            "min_batch_tokens": 1000,
            "redistribute_threshold": 0.4,
        },
        "translation_phase": {
            "chunks_per_call": 1,
            "overlap_chunks": 1,
            "cross_spine_overlap": True,
            "double_pass": True,
            "rolling_summary": True,
            "summary_max_tokens": 2000,
            "glossary_injection": "relevant",
            "qa_check": True,
            # null => use the model/server default sampling temperature.
            "qa_temperature": None,
            "qa_max_retries": 2,
            "min_length_ratio": 0.3,
            "max_length_ratio": 2.0,
        },
        "output": {
            "epub_path": "./book.en.epub",
            "title_suffix": None,
            "new_identifier": False,
            "css": "original",
            "add_translation_note": True,
            "run_epubcheck": False,
        },
        "languages": {"source": "ja", "target": "en"},
        "llm": {
            "max_retries": 3,
            "retry_backoff_seconds": 2,
            "request_timeout_seconds": 300,
        },
    }
    return yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG-level console logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """dao-bridge: AI translation pipeline for Japanese light novel EPUBs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("epub", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    default=None,
    help="Path to an existing config.yaml to copy into the work directory.",
)
@click.option("--work-dir", type=click.Path(), default="./work", help="Work directory path.")
@click.pass_context
def init(ctx: click.Context, epub: str, config_path: str | None, work_dir: str) -> None:
    """Initialise a work directory for an EPUB file."""
    work = Path(work_dir).resolve()
    epub_abs = Path(epub).resolve()

    ensure_dirs(work)
    logger = setup_logging(work, ctx.obj["verbose"])

    cfg_dest = work / "config.yaml"

    if config_path:
        # Copy user-supplied config.
        src_cfg = Path(config_path)
        if not src_cfg.exists():
            raise click.ClickException(f"Config file not found: {src_cfg}")
        cfg_dest.write_text(src_cfg.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Copied config from %s", src_cfg)
    elif not cfg_dest.exists():
        # Generate default config.
        cfg_dest.write_text(
            _default_config_yaml(str(epub_abs), str(work)),
            encoding="utf-8",
        )
        logger.info("Generated default config at %s", cfg_dest)

    # Validate the config is loadable.
    load_config(cfg_dest)

    # Initialise state.
    state = load_state(work)
    if not state.run.source_epub:
        state.run = RunState(source_epub=str(epub_abs), started_at="", status="initialised")
        save_state(work, state)

    logger.info("Work directory initialised at %s", work)
    click.echo(f"Initialised: {work}")
    click.echo(f"Source EPUB: {epub_abs}")
    click.echo(f"Config: {cfg_dest}")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Re-extract even if already completed.")
@click.pass_context
def extract(ctx: click.Context, work_dir: str, force: bool) -> None:
    """Extract spine items from the source EPUB."""
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    from dao_bridge.extract import extract_epub

    manifest = extract_epub(config, state, force=force)

    click.echo(f"Extracted {len(manifest.spine)} spine items, {len(manifest.images)} images")


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Re-clean even if already completed.")
@click.pass_context
def clean(ctx: click.Context, work_dir: str, force: bool) -> None:
    """Clean extracted XHTML into markdown."""
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    # Load manifest.
    mp = manifest_path(work)
    if not mp.exists():
        raise click.ClickException("Manifest not found. Run 'dao-bridge extract' first.")

    from dao_bridge.schemas import Manifest

    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    from dao_bridge.clean import clean_all

    manifest = clean_all(config, manifest, state, force=force)

    total_tokens = sum(i.token_count or 0 for i in manifest.spine)
    click.echo(f"Cleaned {len(manifest.spine)} items ({total_tokens:,} total tokens)")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option(
    "--detail",
    is_flag=True,
    help="Show full error messages for failed items (untruncated).",
)
@click.pass_context
def status(ctx: click.Context, work_dir: str, detail: bool) -> None:
    """Show pipeline status for a work directory."""
    work = Path(work_dir).resolve()
    state = load_state(work)

    click.echo(f"Work directory: {work}")
    click.echo(f"Source EPUB: {state.run.source_epub or '(not set)'}")
    click.echo(f"Run status: {state.run.status}")
    click.echo()

    from collections import Counter, defaultdict

    from dao_bridge.state import STAGE_NAMES

    # Group items by stage prefix.
    stage_items: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for key, item in state.items.items():
        if ":" in key:
            stage_prefix, item_id = key.split(":", 1)
        else:
            stage_prefix, item_id = "unknown", key
        stage_items[stage_prefix].append((item_id, item))

    _ERROR_TRUNCATE = 100

    click.echo("Stages:")
    for stage_name in STAGE_NAMES:
        info = state.stages.get(stage_name)
        if info:
            status_str = info.status
            if info.error_message:
                status_str += f" ({info.error_message})"
        else:
            status_str = "not started"
        click.echo(f"  {stage_name:.<25s} {status_str}")

        # Show per-stage item breakdown if items exist for this stage.
        items = stage_items.get(stage_name, [])
        if not items:
            continue

        counts = Counter(item.status for _, item in items)
        # Show counts on one line, indented under the stage.
        parts = []
        for s in ("completed", "started", "failed", "pending"):
            c = counts.get(s, 0)
            if c > 0:
                parts.append(f"{s}={c}")
        if parts:
            click.echo(f"    {', '.join(parts)}")

        # List non-completed items with details.
        problem_items = [
            (item_id, item)
            for item_id, item in items
            if item.status in ("failed", "started")
        ]
        for item_id, item in problem_items:
            err = item.error_message or ""
            if err and not detail:
                err = (err[:_ERROR_TRUNCATE] + "...") if len(err) > _ERROR_TRUNCATE else err
            label = f"[{item.status}] {item_id}"
            if err:
                click.echo(f"      {label}: {err}")
            else:
                click.echo(f"      {label}")


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option(
    "--spine", "spine_index", type=int, default=None, help="Classify only this spine index."
)
@click.option(
    "--force",
    is_flag=True,
    help="Reclassify all items, overriding existing classifications.",
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed items.",
)
@click.pass_context
def classify(
    ctx: click.Context, work_dir: str, spine_index: int | None, force: bool, retry_failed: bool
) -> None:
    """Classify spine items by content type.

    Determines whether each spine item is a chapter, frontmatter,
    backmatter, table of contents, illustration, etc.  Uses structural
    hints when possible and falls back to LLM classification.

    Items with an existing classification in manifest.json are skipped
    unless --force is passed.  To manually override a classification,
    edit manifest.json directly and re-run subsequent pipeline stages.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    mp = manifest_path(work)
    if not mp.exists():
        raise click.ClickException("Manifest not found. Run 'dao-bridge extract' first.")

    from collections import Counter

    from rich.progress import Progress

    from dao_bridge.classify import run_classify_stage
    from dao_bridge.schemas import Manifest

    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))
    n_items = 1 if spine_index is not None else len(manifest.spine)

    with Progress(transient=True) as progress:
        task = progress.add_task("Classifying...", total=n_items)

        manifest = run_classify_stage(
            work,
            config,
            state,
            force=force,
            retry_failed=retry_failed,
            spine_filter=spine_index,
            on_progress=lambda _: progress.advance(task),
        )

    # Print summary: counts per classification value.
    counts = Counter(item.classification for item in manifest.spine)
    click.echo("Classification summary:")
    for cls_val in (
        "chapter",
        "frontmatter",
        "backmatter",
        "toc_auto",
        "toc_authored",
        "illustration",
        "unknown",
    ):
        n = counts.get(cls_val, 0)
        if n > 0:
            click.echo(f"  {cls_val}: {n}")

    unclassified = [i for i in manifest.spine if i.classification is None]
    if unclassified:
        click.echo(f"\nWarning: {len(unclassified)} item(s) still unclassified.")


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--spine", "spine_index", type=int, default=None, help="Chunk only this spine index.")
@click.option("--force", is_flag=True, help="Rechunk even if already completed.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed items.",
)
@click.pass_context
def chunk(
    ctx: click.Context, work_dir: str, spine_index: int | None, force: bool, retry_failed: bool
) -> None:
    """Chunk cleaned markdown into translation-ready segments."""
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    # Load manifest.
    mp = manifest_path(work)
    if not mp.exists():
        raise click.ClickException("Manifest not found. Run 'dao-bridge extract' first.")

    from dao_bridge.schemas import Manifest

    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    from rich.progress import Progress

    from dao_bridge.chunk import chunk_all

    n_items = 1 if spine_index is not None else len(manifest.spine)

    with Progress(transient=True) as progress:
        task = progress.add_task("Chunking...", total=n_items)

        manifest = chunk_all(
            config,
            manifest,
            state,
            force=force,
            retry_failed=retry_failed,
            spine_filter=spine_index,
            on_progress=lambda _: progress.advance(task),
        )

    total_chunks = sum(i.chunk_count or 0 for i in manifest.spine)
    click.echo(f"Chunked {len(manifest.spine)} items ({total_chunks} total chunks)")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option(
    "--spine", "spine_index", type=int, default=None, help="Assemble only this spine index."
)
@click.option("--force", is_flag=True, help="Reassemble even if already completed.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed items.",
)
@click.pass_context
def assemble(
    ctx: click.Context, work_dir: str, spine_index: int | None, force: bool, retry_failed: bool
) -> None:
    """Assemble translated chunks into per-spine markdown files."""
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    # Load manifest.
    mp = manifest_path(work)
    if not mp.exists():
        raise click.ClickException("Manifest not found. Run 'dao-bridge extract' first.")

    from dao_bridge.schemas import Manifest

    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    from rich.progress import Progress

    from dao_bridge.assemble import assemble_all

    n_items = 1 if spine_index is not None else len(manifest.spine)

    with Progress(transient=True) as progress:
        task = progress.add_task("Assembling...", total=n_items)

        manifest = assemble_all(
            config,
            manifest,
            state,
            force=force,
            retry_failed=retry_failed,
            spine_filter=spine_index,
            on_progress=lambda _: progress.advance(task),
        )

    click.echo("Assembly complete.")


# ---------------------------------------------------------------------------
# glossary-build
# ---------------------------------------------------------------------------


@cli.command("glossary-build")
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option(
    "--spine",
    "spine_index",
    type=int,
    default=None,
    help="Redo only this spine's glossary batches.",
)
@click.option(
    "--batch",
    "single_batch",
    type=str,
    default=None,
    help="Redo a single glossary batch (e.g. 0003.b2).",
)
@click.option("--force", is_flag=True, help="Rebuild glossary from scratch.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed batches.",
)
@click.option(
    "--force-summaries",
    "force_summaries",
    is_flag=True,
    help=(
        "Recompress entity summaries from existing observations without "
        "re-running extraction (requires summary_compress_enabled)."
    ),
)
@click.pass_context
def glossary_build_cmd(
    ctx: click.Context,
    work_dir: str,
    spine_index: int | None,
    single_batch: str | None,
    force: bool,
    retry_failed: bool,
    force_summaries: bool,
) -> None:
    """Extract a per-book glossary from chunked source text.

    Packs chunks into per-spine batches and sends each batch to the LLM
    for glossary extraction.  Entries are merged progressively and saved
    after each batch, making the stage resumable.

    Use --spine or --batch to redo specific items without rebuilding
    the entire glossary.  --batch takes precedence over --spine.

    Use --force-summaries to recompress entity summaries from their
    already-accumulated observations (Phase 2B) without re-running the
    extraction LLM calls.

    Requires: extract, clean, classify, chunk stages completed.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    targeted = single_batch is not None or spine_index is not None
    if targeted and (force or retry_failed):
        raise click.ClickException(
            "--spine/--batch cannot be combined with --force or --retry-failed."
        )
    if force_summaries and (force or retry_failed or targeted):
        raise click.ClickException(
            "--force-summaries cannot be combined with --force, --retry-failed, "
            "--spine, or --batch."
        )

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    if force_summaries:
        try:
            glossary = _run_glossary_build_with_progress(
                work=work,
                config=config,
                state=state,
                force_summaries=True,
                verbose=ctx.obj["verbose"],
            )
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"Summary recompression complete: {len(glossary.entities)} entities."
        )
        return

    glossary = _run_glossary_build_with_progress(
        work=work,
        config=config,
        state=state,
        force=force,
        retry_failed=retry_failed,
        target_spine=spine_index,
        target_batch=single_batch,
        verbose=ctx.obj["verbose"],
    )

    click.echo(f"Glossary build complete: {len(glossary.entities)} entities extracted.")


# ---------------------------------------------------------------------------
# glossary-cluster
# ---------------------------------------------------------------------------


@cli.command("glossary-cluster")
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Re-cluster from scratch.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed batches.",
)
@click.pass_context
def glossary_cluster_cmd(
    ctx: click.Context, work_dir: str, force: bool, retry_failed: bool
) -> None:
    """Find and merge duplicate glossary entities via LLM confirmation.

    Generates candidate entity pairs using deterministic heuristics
    (substring containment, English containment, shared reading, alias
    overlap, Jaro-Winkler similarity) and sends them to the LLM for
    confirmation.  Iterates until no new candidates are found or the
    iteration cap is reached.

    Writes a clustering report to glossary_cluster_report.md.

    Requires: glossary-build stage completed.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    glossary = _run_glossary_cluster_with_progress(
        work=work,
        config=config,
        state=state,
        force=force,
        retry_failed=retry_failed,
        verbose=ctx.obj["verbose"],
    )

    click.echo(f"Glossary cluster complete: {len(glossary.entities)} entities.")
    click.echo(f"Report: {work / 'glossary_cluster_report.md'}")


# ---------------------------------------------------------------------------
# glossary-reconcile
# ---------------------------------------------------------------------------


@cli.command("glossary-reconcile")
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Re-reconcile from scratch.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed items.",
)
@click.pass_context
def glossary_reconcile_cmd(
    ctx: click.Context, work_dir: str, force: bool, retry_failed: bool
) -> None:
    """Resolve within-book glossary conflicts from the build stage.

    Resolves differing English proposals and corrections via LLM calls,
    and consolidates multiple speech-style observations per character.
    Writes a reconciliation report to glossary_reconcile_report.md.

    Requires: glossary-cluster stage completed.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")

    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    glossary = _run_glossary_reconcile_with_progress(
        work=work,
        config=config,
        state=state,
        force=force,
        retry_failed=retry_failed,
        verbose=ctx.obj["verbose"],
    )

    click.echo(f"Glossary reconcile complete: {len(glossary.entities)} entities.")
    click.echo(f"Report: {work / 'glossary_reconcile_report.md'}")


# ---------------------------------------------------------------------------
# glossary-export
# ---------------------------------------------------------------------------


@cli.command("glossary-export")
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--stdout", "use_stdout", is_flag=True, help="Print to stdout instead of file.")
@click.option(
    "--output", "output_path", type=click.Path(), default=None, help="Custom output path."
)
@click.pass_context
def glossary_export_cmd(
    ctx: click.Context, work_dir: str, use_stdout: bool, output_path: str | None
) -> None:
    """Export the per-book glossary as human-readable markdown.

    By default writes to <work_dir>/glossary.md.  Use --stdout to print
    to the console, or --output to specify a custom path.
    """
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)

    from dao_bridge.glossary import glossary_export

    out = Path(output_path) if output_path else None

    md = glossary_export(work, config, stdout=use_stdout, output_path=out)

    if use_stdout:
        click.echo(md)
    else:
        dest = out or (work / "glossary.md")
        click.echo(f"Glossary exported to {dest}")


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--spine", "spine_index", type=int, default=None, help="Translate only this spine.")
@click.option(
    "--chunk",
    "single_chunk",
    type=str,
    default=None,
    help="Translate a single chunk (e.g. 0003.015).",
)
@click.option(
    "--from",
    "from_chunk",
    type=str,
    default=None,
    help="Start translating from this chunk ID (inclusive).",
)
@click.option(
    "--to",
    "to_chunk",
    type=str,
    default=None,
    help="Stop translating after this chunk ID (inclusive).",
)
@click.option("--force", is_flag=True, help="Retranslate even if already completed.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter a completed stage to retry only failed chunks.",
)
@click.pass_context
def translate(
    ctx: click.Context,
    work_dir: str,
    spine_index: int | None,
    single_chunk: str | None,
    from_chunk: str | None,
    to_chunk: str | None,
    force: bool,
    retry_failed: bool,
) -> None:
    """Translate chunked source text to English using LLM.

    By default translates all untranslated chunks in sequential order.
    Use --spine, --chunk, or --from/--to to limit the range.

    QA is advisory and non-blocking: when a chunk still has high-severity
    issues after the QA-fix retries, the best attempt is kept, the chunk is
    marked completed, and the run continues.  Per-attempt artifacts are saved
    for review.  Failed chunks (infrastructure errors, not QA) are retried
    automatically on re-run.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    mp = manifest_path(work)
    if not mp.exists():
        raise click.ClickException("Manifest not found. Run 'dao-bridge extract' first.")

    from dao_bridge.schemas import Manifest

    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    # Resolve range options.
    if single_chunk is not None:
        from_chunk = single_chunk
        to_chunk = single_chunk
    elif spine_index is not None:
        from dao_bridge.translate import _spine_range_for_filter

        from_chunk, to_chunk = _spine_range_for_filter(spine_index, manifest)

    if to_chunk is not None and from_chunk is None:
        raise click.ClickException("--to requires --from.")

    try:
        result = _run_translate_with_progress(
            work=work,
            config=config,
            state=state,
            manifest=manifest,
            force=force,
            retry_failed=retry_failed,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
            verbose=ctx.obj["verbose"],
        )
    except KeyboardInterrupt:
        click.echo("\nTranslation interrupted by user.")
        sys.exit(1)

    # End-of-run summary.
    if result["error"] is None:
        avg = result["avg_time"]
        click.echo(
            f"Translated {result['completed']} chunks. "
            f"Total tokens: {result['total_tokens']:,}. "
            f"Average time per chunk: {avg:.1f} seconds."
        )
    elif "QA failed" in (result["error"] or ""):
        click.echo(
            f"Translated {result['completed']} chunks successfully. "
            f"Halted at chunk {result['failed_chunk']}: {result['error']}. "
            "Fix the issue and re-run to continue."
        )
    else:
        click.echo(
            f"Translated {result['completed']} chunks successfully. "
            f"Failed at chunk {result['failed_chunk']}: {result['error']}. "
            "Re-run to retry."
        )

    if result["error"] is not None:
        sys.exit(1)


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Force rebuild even if already completed.")
@click.pass_context
def rebuild(ctx: click.Context, work_dir: str, force: bool) -> None:
    """Build translated EPUB from assembled markdown files.

    Copies the source EPUB at the ZIP level, replacing only translated
    XHTML body content, ToC entries, and metadata.  Preserves all original
    structure, images, fonts, and CSS.
    """
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)

    from dao_bridge.rebuild import run_rebuild_stage

    run_rebuild_stage(work, config, force=force)
    click.echo(f"Output EPUB: {config.output.epub_path}")


# ---------------------------------------------------------------------------
# run (chains all stages)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--work-dir", type=click.Path(exists=True), default="./work", help="Work directory path."
)
@click.option("--force", is_flag=True, help="Force re-run of all stages.")
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-enter completed stages to retry only failed items.",
)
@click.pass_context
def run(ctx: click.Context, work_dir: str, force: bool, retry_failed: bool) -> None:
    """Run the full translation pipeline.

    Chains: extract -> clean -> classify -> chunk -> glossary-build ->
    glossary-cluster -> glossary-reconcile -> translate -> assemble -> rebuild.

    Each stage skips completed work unless --force is passed.
    Use --retry-failed to re-enter completed stages and retry only
    failed items without reprocessing everything.
    Stops immediately on first stage failure.
    """
    if force and retry_failed:
        raise click.ClickException("--force and --retry-failed are mutually exclusive.")
    work = Path(work_dir).resolve()
    setup_logging(work, ctx.obj["verbose"])
    config = _resolve_config(work)
    state = load_state(work)

    from dao_bridge.schemas import Manifest

    manifest: Manifest | None = None
    mp = manifest_path(work)

    def _run_stage(name: str, fn):
        """Execute a stage, halting the pipeline on failure."""
        click.echo(f"=== {name} ===")
        try:
            return fn()
        except Exception as e:
            click.echo(f"\nPipeline halted at stage '{name}': {e}", err=True)
            click.echo("Fix the issue and re-run.  Completed stages will be skipped.", err=True)
            sys.exit(1)

    # --- Stage 1: extract ---
    def _extract():
        nonlocal manifest
        from dao_bridge.extract import extract_epub

        manifest = extract_epub(config, state, force=force)
        click.echo(f"  {len(manifest.spine)} spine items, {len(manifest.images)} images")

    _run_stage("extract", _extract)

    # Manifest must exist after extract.
    if manifest is None:
        if not mp.exists():
            raise click.ClickException(
                "Manifest not found after extract. Run 'dao-bridge init <epub>' first."
            )
        manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    # --- Stage 2: clean ---
    def _clean():
        nonlocal manifest
        from dao_bridge.clean import clean_all

        manifest = clean_all(config, manifest, state, force=force)
        total_tokens = sum(i.token_count or 0 for i in manifest.spine)
        click.echo(f"  {len(manifest.spine)} items ({total_tokens:,} total tokens)")

    _run_stage("clean", _clean)

    # --- Stage 3: classify ---
    def _classify():
        nonlocal manifest
        from dao_bridge.classify import run_classify_stage

        manifest = run_classify_stage(work, config, state, force=force, retry_failed=retry_failed)
        from collections import Counter

        counts = Counter(item.classification for item in manifest.spine)
        click.echo(f"  {dict(counts)}")

    _run_stage("classify", _classify)

    # --- Stage 4: chunk ---
    def _chunk():
        nonlocal manifest
        from dao_bridge.chunk import chunk_all

        manifest = chunk_all(config, manifest, state, force=force, retry_failed=retry_failed)
        total_chunks = sum(i.chunk_count or 0 for i in manifest.spine)
        click.echo(f"  {total_chunks} total chunks")

    _run_stage("chunk", _chunk)

    # --- Stage 5: glossary-build ---
    def _glossary_build():
        glossary = _run_glossary_build_with_progress(
            work=work,
            config=config,
            state=state,
            force=force,
            retry_failed=retry_failed,
            verbose=ctx.obj["verbose"],
        )
        click.echo(f"  {len(glossary.entities)} entities extracted")

    _run_stage("glossary-build", _glossary_build)

    # --- Stage 6: glossary-cluster ---
    def _glossary_cluster():
        glossary = _run_glossary_cluster_with_progress(
            work=work,
            config=config,
            state=state,
            force=force,
            retry_failed=retry_failed,
            verbose=ctx.obj["verbose"],
        )
        click.echo(f"  {len(glossary.entities)} entities (after clustering)")

    _run_stage("glossary-cluster", _glossary_cluster)

    # --- Stage 7: glossary-reconcile ---
    def _glossary_reconcile():
        glossary = _run_glossary_reconcile_with_progress(
            work=work,
            config=config,
            state=state,
            force=force,
            retry_failed=retry_failed,
            verbose=ctx.obj["verbose"],
        )
        click.echo(f"  {len(glossary.entities)} entities (reconciled)")

    _run_stage("glossary-reconcile", _glossary_reconcile)

    # --- Stage 7b: glossary-crosscheck (skip with warning if not implemented) ---
    if config.glossary.master_glossary_path and config.glossary.crosscheck.enabled:
        click.echo("=== glossary-crosscheck ===")
        click.echo("  WARNING: glossary-crosscheck is not yet implemented, skipping.")

    # --- Stage 8: translate ---
    def _translate():
        # Re-load manifest in case chunk stage updated it.
        nonlocal manifest
        manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

        result = _run_translate_with_progress(
            work=work,
            config=config,
            state=state,
            manifest=manifest,
            force=force,
            retry_failed=retry_failed,
            verbose=ctx.obj["verbose"],
        )
        if result["error"] is not None:
            raise RuntimeError(
                f"Translation failed at chunk {result['failed_chunk']}: {result['error']}"
            )
        click.echo(
            f"  {result['completed']} chunks translated, {result['total_tokens']:,} total tokens"
        )

    _run_stage("translate", _translate)

    # --- Stage 8: assemble ---
    def _assemble():
        nonlocal manifest
        manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

        from dao_bridge.assemble import assemble_all

        manifest = assemble_all(config, manifest, state, force=force, retry_failed=retry_failed)
        assembled_count = sum(
            1 for item in manifest.spine if item.chunk_count and item.chunk_count > 0
        )
        click.echo(f"  {assembled_count} spine items assembled")

    _run_stage("assemble", _assemble)

    # --- Stage 9: rebuild ---
    def _rebuild():
        from dao_bridge.rebuild import run_rebuild_stage

        run_rebuild_stage(work, config, force=force, state=state)
        click.echo(f"  Output EPUB: {config.output.epub_path}")

    _run_stage("rebuild", _rebuild)

    click.echo(f"\nPipeline complete.  Output: {config.output.epub_path}")


# ---------------------------------------------------------------------------
# Placeholder commands (not yet implemented)
# ---------------------------------------------------------------------------

_MASTER_GLOSSARY_COMMANDS = [
    "glossary-crosscheck",
    "glossary-promote",
    "glossary-import-reference",
]


def _make_placeholder(name: str, message: str | None = None):
    @cli.command(name=name)
    @click.pass_context
    def placeholder(ctx: click.Context) -> None:
        click.echo(message or f"'{name}' is not yet implemented.")
        sys.exit(0)

    placeholder.__doc__ = f"{name} (not yet implemented)."
    return placeholder


for _cmd_name in _MASTER_GLOSSARY_COMMANDS:
    _make_placeholder(
        _cmd_name,
        f"'{_cmd_name}' is not yet implemented — master glossary features "
        "coming in a future release.",
    )
