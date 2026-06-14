"""Logging setup with Rich console handler and file handler.

Call :func:`setup_logging` once at CLI entry to configure the ``"dao_bridge"``
logger with a Rich console handler and a file handler.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class _PlainMarkupFormatter(logging.Formatter):
    """File formatter that undoes Rich-escaping of bracket tokens.

    The console handler has ``markup=True``, so
    :func:`dao_bridge.llm_client._context_prefix` escapes the ``[label]`` token
    (``rich.markup.escape`` inserts a backslash before the ``[``) to stop Rich
    from parsing it as a style tag and dropping it.  On the console Rich strips
    that backslash automatically; the plain-text file handler would otherwise
    keep it (``\\[summary:<id>]``).

    ``rich.markup.escape`` only ever inserts ``\\`` immediately before a ``[``
    that opens a tag-like token, so reversing exactly ``\\[`` -> ``[`` is its
    precise inverse.  This is deliberately targeted: a blanket
    ``rich.markup.render().plain`` would also eat arbitrary markup-like text in
    messages (e.g. ``something [not a tag] here`` or raw LLM responses), which
    would lose content in ``run.log``.  Intentional styled tags elsewhere (e.g.
    ``[bold]`` in :mod:`dao_bridge.state`) are left as-is, unchanged from before.
    """

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace("\\[", "[")


def _make_utf8_console() -> Console:
    """Create a Rich Console that writes UTF-8 to stdout.

    On Windows with legacy code-pages (e.g. cp1252), Rich's default
    ``LegacyWindowsTerm`` renderer raises ``UnicodeEncodeError`` when log
    messages contain CJK characters. Reconfiguring ``sys.stdout`` to
    ``utf-8`` with ``errors="replace"`` avoids the crash without creating
    short-lived wrappers that can close the shared stdout stream when Rich
    progress consoles are torn down.
    """
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return Console(file=sys.stdout, force_terminal=True)


def setup_logging(work_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure the ``dao_bridge`` logger.

    Parameters
    ----------
    work_dir:
        Work directory root.  Log file is written to ``{work_dir}/logs/run.log``.
    verbose:
        If *True*, set console level to DEBUG; otherwise INFO.

    Returns
    -------
    logging.Logger
        The configured ``"dao_bridge"`` logger.
    """
    logger = logging.getLogger("dao_bridge")
    # Avoid duplicate handlers if called more than once (e.g. in tests).
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # --- Console handler (Rich) ---
    console = _make_utf8_console()
    console_handler = RichHandler(
        level=logging.DEBUG if verbose else logging.INFO,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        console=console,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console_handler)

    # --- File handler ---
    log_file = work_dir / "logs" / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        _PlainMarkupFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s.%(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger
