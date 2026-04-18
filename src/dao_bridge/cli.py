"""CLI entry point for dao-bridge.

All commands accept ``--verbose`` to enable DEBUG-level console logging.
Commands check state before starting and skip completed work unless ``--force``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from dao_bridge.config import AppConfig, load_config
from dao_bridge.logging import setup_logging
from dao_bridge.state import (
    RunState,
    load_state,
    save_state,
)
from dao_bridge.workdir import ensure_dirs, manifest_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
                "temperature": 0.0,
            },
            "glossary": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "model": "gemma-4-26b-a4b",
                "temperature": 0.2,
            },
            "translate": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "model": "gemma-4-26b-a4b",
                "temperature": 0.3,
            },
            # "summarize" falls back to "translate" if absent.
            # "summarize": {
            #     "base_url": "http://localhost:8080/v1",
            #     "api_key": "not-needed",
            #     "model": "qwen3-30b-a3b",
            #     "temperature": 0.2,
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
            "master_glossary_path": None,
            "promote_on_complete": False,
        },
        "glossary_phase": {
            "target_tokens_per_call": 8000,
            "overlap_chunks": 0,
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
            "qa_max_retries": 1,
            "min_length_ratio": 0.3,
            "max_length_ratio": 2.0,
        },
        "output": {
            "epub_path": "./book.en.epub",
            "title_suffix": " (English Translation)",
            "new_identifier": False,
            "css": "original",
            "add_translation_note": True,
            "validate": False,
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
@click.pass_context
def status(ctx: click.Context, work_dir: str) -> None:
    """Show pipeline status for a work directory."""
    work = Path(work_dir).resolve()
    state = load_state(work)

    click.echo(f"Work directory: {work}")
    click.echo(f"Source EPUB: {state.run.source_epub or '(not set)'}")
    click.echo(f"Run status: {state.run.status}")
    click.echo()

    from dao_bridge.state import STAGE_NAMES

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

    # Count item statuses.
    if state.items:
        click.echo()
        from collections import Counter

        counts = Counter(item.status for item in state.items.values())
        click.echo("Items: " + ", ".join(f"{s}={c}" for s, c in sorted(counts.items())))


# ---------------------------------------------------------------------------
# Placeholder commands (not yet implemented)
# ---------------------------------------------------------------------------

_PLACEHOLDER_COMMANDS = [
    "classify",
    "chunk",
    "translate",
    "assemble",
    "rebuild",
    "run",
    "glossary-build",
    "glossary-reconcile",
    "glossary-crosscheck",
    "glossary-promote",
    "glossary-import-reference",
    "glossary-export",
]


def _make_placeholder(name: str):
    @cli.command(name=name)
    @click.pass_context
    def placeholder(ctx: click.Context) -> None:
        click.echo(f"'{name}' is not yet implemented.")
        sys.exit(0)

    placeholder.__doc__ = f"{name} (not yet implemented)."
    return placeholder


for _cmd_name in _PLACEHOLDER_COMMANDS:
    _make_placeholder(_cmd_name)
