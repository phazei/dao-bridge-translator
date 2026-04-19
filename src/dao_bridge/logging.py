"""Logging setup with Rich console handler and file handler.

Call :func:`setup_logging` once at CLI entry to configure the ``"dao_bridge"``
logger with a Rich console handler and a file handler.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


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
    console_handler = RichHandler(
        level=logging.DEBUG if verbose else logging.INFO,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
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
