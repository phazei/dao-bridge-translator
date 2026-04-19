"""Logging setup with Rich console handler and file handler.

Call :func:`setup_logging` once at CLI entry to configure the ``"dao_bridge"``
logger with a Rich console handler and a file handler.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def _make_utf8_console() -> Console:
    """Create a Rich Console that writes UTF-8 to stdout.

    On Windows with legacy code-pages (e.g. cp1252), Rich's default
    ``LegacyWindowsTerm`` renderer raises ``UnicodeEncodeError`` when log
    messages contain CJK characters.  Wrapping ``sys.stdout`` in a
    ``TextIOWrapper`` with ``utf-8`` encoding and ``errors="replace"``
    avoids the crash while keeping output readable.
    """
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            stream = sys.stdout
    else:
        stream = sys.stdout
    return Console(file=stream, force_terminal=True)


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
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s.%(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger
