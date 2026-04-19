"""Work directory path helpers and atomic file operations.

Every module uses these helpers instead of constructing paths manually.
All JSON writes go through ``atomic_write`` to prevent corruption on crash.
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path

# ---------------------------------------------------------------------------
# Padding / formatting helpers
# ---------------------------------------------------------------------------


def pad_spine(spine_index: int, width: int = 4) -> str:
    """Return a zero-padded string for *spine_index*.

    Parameters
    ----------
    spine_index:
        The spine index to pad.
    width:
        Minimum number of digits.  Defaults to 4.  Pipeline code should
        always pass ``manifest.spine_padding_width`` explicitly; the
        default is a safety net for tests and ad-hoc usage.
    """
    return f"{spine_index:0{width}d}"


def format_chunk_id(spine_index: int, chunk_index: int, spine_width: int = 4) -> str:
    """Return chunk identifier, e.g. ``"NNNN.MMM"``.

    The spine portion uses *spine_width* digits (default 4).
    The chunk portion is always 3 digits.
    """
    return f"{pad_spine(spine_index, spine_width)}.{chunk_index:03d}"


def parse_chunk_id(chunk_id: str) -> tuple[int, int]:
    """Parse ``"NNN.MMM"`` back to ``(spine_index, chunk_index)``."""
    parts = chunk_id.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk_id format: {chunk_id!r}")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# ZIP / OPF path helpers
# ---------------------------------------------------------------------------


def resolve_zip_path(opf_dir: str, href: str) -> str:
    """Resolve an OPF-relative *href* to a ZIP-absolute path.

    Parameters
    ----------
    opf_dir:
        Directory of the OPF file within the ZIP (e.g. ``"OEBPS"``).
        Empty string if OPF is at the ZIP root.
    href:
        The ``href`` attribute from the OPF manifest item.

    Returns
    -------
    str
        ZIP-absolute path (forward-slash separated, normalised).

    Examples
    --------
    >>> resolve_zip_path("OEBPS", "Text/chapter1.xhtml")
    'OEBPS/Text/chapter1.xhtml'
    >>> resolve_zip_path("", "chapter1.xhtml")
    'chapter1.xhtml'
    """
    if not opf_dir:
        return posixpath.normpath(href)
    return posixpath.normpath(posixpath.join(opf_dir, href))


# ---------------------------------------------------------------------------
# Work directory path helpers
# ---------------------------------------------------------------------------


def raw_path(work_dir: Path, spine_index: int, spine_width: int = 4) -> Path:
    """``raw/NNNN.xhtml``"""
    return work_dir / "raw" / f"{pad_spine(spine_index, spine_width)}.xhtml"


def clean_path(work_dir: Path, spine_index: int, spine_width: int = 4) -> Path:
    """``clean/NNNN.md``"""
    return work_dir / "clean" / f"{pad_spine(spine_index, spine_width)}.md"


def chunk_dir(work_dir: Path, spine_index: int, spine_width: int = 4) -> Path:
    """``chunks/NNNN/``"""
    return work_dir / "chunks" / pad_spine(spine_index, spine_width)


def chunk_path(work_dir: Path, chunk_id: str, spine_width: int = 4) -> Path:
    """``chunks/NNNN/NNNN.MMM.json``"""
    spine_index, _ = parse_chunk_id(chunk_id)
    return work_dir / "chunks" / pad_spine(spine_index, spine_width) / f"{chunk_id}.json"


def translation_dir(work_dir: Path, spine_index: int, spine_width: int = 4) -> Path:
    """``translations/NNNN/``"""
    return work_dir / "translations" / pad_spine(spine_index, spine_width)


def translation_path(work_dir: Path, chunk_id: str, spine_width: int = 4) -> Path:
    """``translations/NNNN/NNNN.MMM.json``"""
    spine_index, _ = parse_chunk_id(chunk_id)
    return work_dir / "translations" / pad_spine(spine_index, spine_width) / f"{chunk_id}.json"


def assembled_path(work_dir: Path, spine_index: int, spine_width: int = 4) -> Path:
    """``assembled/NNNN.md``"""
    return work_dir / "assembled" / f"{pad_spine(spine_index, spine_width)}.md"


def summary_path(work_dir: Path) -> Path:
    """``summaries/rolling_summary.json``"""
    return work_dir / "summaries" / "rolling_summary.json"


def glossary_path(work_dir: Path) -> Path:
    """``glossary.json``"""
    return work_dir / "glossary.json"


def manifest_path(work_dir: Path) -> Path:
    """``manifest.json``"""
    return work_dir / "manifest.json"


def state_path(work_dir: Path) -> Path:
    """``state.json``"""
    return work_dir / "state.json"


def log_dir(work_dir: Path) -> Path:
    """``logs/``"""
    return work_dir / "logs"


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

_SUBDIRS = [
    "raw",
    "clean",
    "chunks",
    "translations",
    "assembled",
    "summaries",
    "logs",
]


def ensure_dirs(work_dir: Path) -> None:
    """Create the work directory and all standard subdirectories."""
    for subdir in _SUBDIRS:
        (work_dir / subdir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write *data* to *path* atomically via a temporary file.

    Writes to ``<path>.tmp`` first, then replaces the target with
    ``os.replace()``.  This ensures readers never see a partially-written
    file — they either see the old version or the new one.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(data, str):
            tmp_path.write_text(data, encoding="utf-8")
        else:
            tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
