"""EPUB extraction: spine items to raw XHTML files.

Opens an EPUB with ebooklib, iterates the spine in order, writes each
``ITEM_DOCUMENT`` to ``raw/NNN.xhtml``, and builds the :class:`Manifest`.
"""

from __future__ import annotations

import logging
import posixpath
import re
import unicodedata
import zipfile
from pathlib import Path

import ebooklib
from ebooklib import epub
from lxml import etree

from dao_bridge.config import AppConfig
from dao_bridge.schemas import Manifest, ManifestItem
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    mark_item_completed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
    reset_stage,
)
from dao_bridge.workdir import (
    atomic_write,
    ensure_dirs,
    manifest_path,
    pad_spine,
    raw_path,
)

logger = logging.getLogger("dao_bridge")


# ---------------------------------------------------------------------------
# book_id derivation
# ---------------------------------------------------------------------------


def _derive_book_id(book: epub.EpubBook, epub_path: Path) -> str:
    """Derive a stable book identifier from EPUB metadata.

    Priority:
    1. ISBN from ``dc:identifier`` metadata.
    2. Normalized title + volume (if extractable).
    3. Filename stem.
    """
    # Try ISBN first.
    identifiers = book.get_metadata("DC", "identifier")
    for value, attrs in identifiers:
        normalized = value.strip().upper()
        # Strip common prefixes.
        for prefix in ("URN:ISBN:", "ISBN:"):
            if normalized.startswith(prefix):
                isbn = normalized[len(prefix) :]
                isbn_clean = isbn.replace("-", "").replace(" ", "")
                if isbn_clean.isdigit() and len(isbn_clean) in (10, 13):
                    return f"isbn-{isbn_clean}"

    # Fall back to normalized title.
    titles = book.get_metadata("DC", "title")
    if titles:
        title_text = titles[0][0]
        return _normalize_for_id(title_text)

    # Fall back to filename stem.
    return _normalize_for_id(epub_path.stem)


def _normalize_for_id(text: str) -> str:
    """Normalize a string into a URL-safe identifier."""
    # NFKC normalize (collapses fullwidth chars, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric (keeping CJK) with hyphens
    text = re.sub(r"[^\w]", "-", text)
    # Collapse multiple hyphens
    text = re.sub(r"-{2,}", "-", text)
    # Strip leading/trailing hyphens
    text = text.strip("-")
    return text or "unknown"


_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


def _get_opf_dir(epub_path: Path) -> str:
    """Parse ``META-INF/container.xml`` to find the OPF directory.

    Returns the directory portion of the OPF full-path (e.g. ``"OEBPS"``).
    Returns ``""`` if the OPF is at the ZIP root.
    """
    with zipfile.ZipFile(epub_path, "r") as zf:
        container_xml = zf.read("META-INF/container.xml")
    tree = etree.fromstring(container_xml)
    rootfile = tree.find(
        f".//{{{_CONTAINER_NS}}}rootfile[@media-type='application/oebps-package+xml']"
    )
    if rootfile is None:
        logger.warning("Could not find rootfile in container.xml, assuming OPF at root")
        return ""
    opf_full_path = rootfile.get("full-path", "")
    return posixpath.dirname(opf_full_path)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_epub(
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
) -> Manifest:
    """Extract spine items and metadata from the source EPUB.

    Parameters
    ----------
    config:
        Application configuration.
    state:
        Pipeline state (mutated in place; persisted after each item).
    force:
        If *True*, re-extract even if the stage is already completed.

    Returns
    -------
    Manifest
        The populated manifest (also persisted to ``manifest.json``).
    """
    work_dir = config.work_dir_path

    if not force and is_stage_completed(state, "extract"):
        logger.info("Extract stage already completed — skipping (use --force to re-run)")
        # Load and return existing manifest.
        mp = manifest_path(work_dir)
        if mp.exists():
            import json

            return Manifest(**json.loads(mp.read_text(encoding="utf-8")))
        # If manifest is missing despite stage being complete, fall through.

    ensure_dirs(work_dir)

    if force:
        reset_stage(work_dir, state, "extract")

    mark_stage_started(work_dir, state, "extract")

    epub_path = config.source_epub_path
    logger.info("Reading EPUB: %s", epub_path)
    book = epub.read_epub(str(epub_path))

    # --- OPF directory ---
    opf_dir = _get_opf_dir(epub_path)
    logger.debug("OPF directory: %s", opf_dir or "(root)")

    # --- Metadata ---
    metadata: dict = {}
    for key in ("title", "language", "identifier", "creator", "publisher", "date"):
        values = book.get_metadata("DC", key)
        if values:
            # Store as single value if one, list if many.
            if len(values) == 1:
                metadata[key] = values[0][0]
            else:
                metadata[key] = [v[0] for v in values]

    book_id = _derive_book_id(book, epub_path)
    logger.info("Derived book_id: %s", book_id)

    # --- Spine items ---
    spine_items: list[ManifestItem] = []

    # Build a lookup from item id to item object.
    all_items = {item.get_id(): item for item in book.get_items()}

    # Track which document items are in the spine.
    spine_doc_ids: set[str] = set()

    for spine_index, (item_id, _linear) in enumerate(book.spine):
        item = all_items.get(item_id)
        if item is None:
            logger.warning("Spine references unknown item id: %s", item_id)
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            logger.debug(
                "Spine item %s is not a document (type=%s), skipping", item_id, item.get_type()
            )
            continue

        spine_doc_ids.add(item_id)
        padded = pad_spine(spine_index)
        rp = raw_path(work_dir, spine_index)
        href = item.get_name()

        mark_item_started(work_dir, state, "extract", padded)

        # Write raw XHTML content.
        content = item.get_content()

        # IMPORTANT: Data writes MUST happen before status writes.
        # If we crash after marking an item complete but before the data is
        # on disk, a resumed run will skip the item, leaving a gap.  By
        # writing data first, the worst case on crash is a completed data
        # file with an incomplete status — the item will simply be
        # re-extracted on resume (idempotent).
        assert isinstance(content, bytes), "EPUB item content must be bytes"
        rp.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(rp, content)

        manifest_item = ManifestItem(
            spine_index=spine_index,
            original_href=href,
            raw_path=str(rp.relative_to(work_dir)),
        )
        spine_items.append(manifest_item)

        mark_item_completed(work_dir, state, "extract", padded)
        logger.debug("Extracted spine %s: %s -> %s", padded, href, rp.name)

    # --- Images ---
    images: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        images.append(item.get_name())

    # --- Warn about documents not in spine ---
    for item_id, item in all_items.items():
        if (
            item.get_type() == ebooklib.ITEM_DOCUMENT
            and item_id not in spine_doc_ids
            # Navigation documents are expected to be outside the spine in many EPUBs.
            and not getattr(item, "is_nav", False)
            and "nav" not in item_id.lower()
        ):
            logger.warning(
                "Document item '%s' (%s) is not in the spine",
                item_id,
                item.get_name(),
            )

    # --- Build and persist manifest ---
    manifest = Manifest(
        source_epub_path=str(epub_path),
        book_id=book_id,
        opf_dir=opf_dir,
        spine=spine_items,
        images=images,
        metadata=metadata,
    )

    # IMPORTANT: Data writes (manifest) MUST happen before status writes
    # (stage completion).  See comment above on the per-item pattern.
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))

    mark_stage_completed(work_dir, state, "extract")
    logger.info(
        "Extraction complete: %d spine items, %d images",
        len(spine_items),
        len(images),
    )

    return manifest
