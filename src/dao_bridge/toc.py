"""Table-of-Contents translation and OPF metadata updates.

Handles both EPUB 2 (``toc.ncx``) and EPUB 3 (``nav.xhtml``) ToC formats.
Titles are extracted, deduplicated across both files, translated in a single
LLM call, and written back.  OPF metadata (language, title, description,
identifier) is updated for the target language.
"""

from __future__ import annotations

import functools
import json
import logging
import posixpath
import uuid
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

from dao_bridge.config import AppConfig
from dao_bridge.llm_client import LLMClient
from dao_bridge.schemas import Glossary, Manifest, TocTranslationResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML namespaces
# ---------------------------------------------------------------------------

_NS_OPF = "http://www.idpf.org/2007/opf"
_NS_DC = "http://purl.org/dc/elements/1.1/"
_NS_NCX = "http://www.daisy.org/z3986/2005/ncx/"
_NS_XHTML = "http://www.w3.org/1999/xhtml"
_NS_EPUB = "http://www.idpf.org/2007/ops"

_NCX_NSMAP = {"ncx": _NS_NCX}
_OPF_NSMAP = {"opf": _NS_OPF, "dc": _NS_DC}

# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parent / "prompts"


@functools.lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the ``prompts/`` directory.

    Cached -- template files are read once per process.

    Parameters
    ----------
    name:
        Template filename (e.g. ``"translate_toc.txt"``).
    """
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_zip_path(opf_dir: str, href: str) -> str:
    """Resolve an OPF-relative *href* to a ZIP-absolute path."""
    if not opf_dir:
        return posixpath.normpath(href)
    return posixpath.normpath(posixpath.join(opf_dir, href))


def _element_text(el: etree._Element) -> str:
    """Return the full text content of *el*, including tail of children."""
    return "".join(el.itertext()).strip()


def _read_zip_entry(epub_path: str, zip_path: str) -> bytes:
    """Read a single entry from an EPUB ZIP."""
    with zipfile.ZipFile(epub_path, "r") as zf:
        return zf.read(zip_path)


def _find_opf_zip_path(epub_path: str) -> str:
    """Return the ZIP-internal path of the OPF file."""
    container_xml = _read_zip_entry(epub_path, "META-INF/container.xml")
    tree = etree.fromstring(container_xml)
    ns = "urn:oasis:names:tc:opendocument:xmlns:container"
    rootfile = tree.find(f".//{{{ns}}}rootfile[@media-type='application/oebps-package+xml']")
    if rootfile is None:
        raise RuntimeError("Could not find rootfile in container.xml")
    return rootfile.get("full-path", "content.opf")


# ---------------------------------------------------------------------------
# Find ToC files
# ---------------------------------------------------------------------------


def find_toc_files(source_epub_path: str, opf_dir: str) -> tuple[str | None, str | None]:
    """Identify NCX and nav document ZIP paths from the OPF.

    Parameters
    ----------
    source_epub_path:
        Path to the source EPUB file on disk.
    opf_dir:
        Directory of the OPF file within the ZIP (e.g. ``"OEBPS"``).

    Returns
    -------
    tuple[str | None, str | None]
        ``(ncx_zip_path, nav_zip_path)`` -- either may be ``None``.
    """
    opf_zip_path = _find_opf_zip_path(source_epub_path)
    opf_bytes = _read_zip_entry(source_epub_path, opf_zip_path)
    opf = etree.fromstring(opf_bytes)

    ncx_zip_path: str | None = None
    nav_zip_path: str | None = None

    # --- NCX: <spine toc="ncx_id"> -> <item id="ncx_id" href="..."> ---
    spine_el = opf.find(f"{{{_NS_OPF}}}spine")
    if spine_el is not None:
        ncx_id = spine_el.get("toc")
        if ncx_id:
            # Find manifest item with this id.
            for item in opf.iter(f"{{{_NS_OPF}}}item"):
                if item.get("id") == ncx_id:
                    href = item.get("href", "")
                    ncx_zip_path = _resolve_zip_path(opf_dir, href)
                    break

    # --- Nav: <item properties="nav" ...> ---
    for item in opf.iter(f"{{{_NS_OPF}}}item"):
        props = item.get("properties", "")
        if "nav" in props.split():
            href = item.get("href", "")
            nav_zip_path = _resolve_zip_path(opf_dir, href)
            break

    return ncx_zip_path, nav_zip_path


# ---------------------------------------------------------------------------
# Extract ToC titles
# ---------------------------------------------------------------------------


def extract_toc_titles(toc_content: str, toc_type: str) -> list[str]:
    """Extract title strings from a ToC file in document order.

    Parameters
    ----------
    toc_content:
        The raw XML/XHTML content of the ToC file.
    toc_type:
        ``"ncx"`` or ``"nav"``.

    Returns
    -------
    list[str]
        Flat list of title strings in document order.  Nested entries are
        included at their natural position.
    """
    tree = etree.fromstring(
        toc_content.encode("utf-8") if isinstance(toc_content, str) else toc_content
    )

    if toc_type == "ncx":
        return _extract_ncx_titles(tree)
    elif toc_type == "nav":
        return _extract_nav_titles(tree)
    else:
        raise ValueError(f"Unknown toc_type: {toc_type!r}")


def _extract_ncx_titles(tree: etree._Element) -> list[str]:
    """Extract titles from an NCX ``<navMap>``."""
    titles: list[str] = []
    # Walk navPoints in document order and extract their navLabel/text.
    for nav_point in tree.iter(f"{{{_NS_NCX}}}navPoint"):
        label = nav_point.find(f"{{{_NS_NCX}}}navLabel")
        if label is not None:
            text_el = label.find(f"{{{_NS_NCX}}}text")
            if text_el is not None:
                titles.append(_element_text(text_el))
    return titles


def _extract_nav_titles(tree: etree._Element) -> list[str]:
    """Extract titles from a nav XHTML ``<nav epub:type="toc">``."""
    titles: list[str] = []

    # Find the toc nav element.  It may use the XHTML namespace.
    toc_nav = None
    for nav in tree.iter(f"{{{_NS_XHTML}}}nav", "nav"):
        # Check epub:type attribute (may be namespaced or bare).
        epub_type = nav.get(f"{{{_NS_EPUB}}}type", nav.get("epub:type", ""))
        if "toc" in epub_type.split():
            toc_nav = nav
            break

    if toc_nav is None:
        logger.warning("No <nav epub:type='toc'> found in nav document")
        return titles

    # Find all <a> elements inside the toc nav in document order.
    for a_el in toc_nav.iter(f"{{{_NS_XHTML}}}a", "a"):
        title = _element_text(a_el)
        titles.append(title)

    return titles


# ---------------------------------------------------------------------------
# Translate titles
# ---------------------------------------------------------------------------


def _render_glossary_for_toc(glossary: Glossary) -> str:
    """Render a compact glossary for ToC title translation.

    Only includes names and places (most relevant for chapter titles).
    """
    if not glossary.entries:
        return "(no glossary available)"
    lines: list[str] = []
    for entry in glossary.entries:
        if entry.category in ("character", "place", "organisation", "faction", "title"):
            src = entry.source_term or ""
            eng = entry.english
            if src:
                lines.append(f"  {src} -> {eng}")
            else:
                lines.append(f"  {eng}")
    return "\n".join(lines) if lines else "(no relevant glossary entries)"


def translate_titles(
    titles: list[str],
    glossary: Glossary,
    llm_client: LLMClient,
    config: AppConfig,
) -> list[str]:
    """Translate a list of ToC titles via one LLM call.

    Parameters
    ----------
    titles:
        Original title strings.
    glossary:
        Per-book glossary (names/places used for consistency).
    llm_client:
        LLM client to use for the translation call.
    config:
        Application config (languages, etc.).

    Returns
    -------
    list[str]
        Translated titles in the same order.

    Raises
    ------
    ValueError
        If the LLM returns a different number of titles than the input.
    """
    if not titles:
        return []

    from dao_bridge.config import resolve_language_name

    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    template = _load_prompt_template("translate_toc.txt")
    glossary_text = _render_glossary_for_toc(glossary)

    system_prompt = template.format(
        source_language=source_lang,
        target_language=target_lang,
        glossary=glossary_text,
        titles_json=json.dumps(titles, ensure_ascii=False, indent=2),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(titles, ensure_ascii=False)},
    ]

    response: TocTranslationResponse = llm_client.complete_json(
        messages,
        response_model=TocTranslationResponse,
    )

    if len(response.titles) != len(titles):
        raise ValueError(
            f"LLM returned {len(response.titles)} titles, expected {len(titles)}. "
            f"Input: {titles!r}, Output: {response.titles!r}"
        )

    return response.titles


# ---------------------------------------------------------------------------
# Apply translated titles
# ---------------------------------------------------------------------------


def apply_translated_titles(
    toc_content: str,
    toc_type: str,
    translated_titles: list[str],
) -> str:
    """Write translated titles back into a ToC file.

    Each title element's content is replaced with a single text node.
    All structural attributes (hrefs, nesting, IDs) are preserved.

    Parameters
    ----------
    toc_content:
        Original ToC file content.
    toc_type:
        ``"ncx"`` or ``"nav"``.
    translated_titles:
        Translated titles in the same order as extracted.

    Returns
    -------
    str
        Modified ToC file content.
    """
    tree = etree.fromstring(
        toc_content.encode("utf-8") if isinstance(toc_content, str) else toc_content
    )

    if toc_type == "ncx":
        _apply_ncx_titles(tree, translated_titles)
    elif toc_type == "nav":
        _apply_nav_titles(tree, translated_titles)
    else:
        raise ValueError(f"Unknown toc_type: {toc_type!r}")

    # Serialize.  Preserve XML declaration.
    return etree.tostring(
        tree,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    ).decode("utf-8")


def _apply_ncx_titles(tree: etree._Element, translated: list[str]) -> None:
    """Replace title text in NCX ``<navPoint>`` ``<navLabel><text>`` elements."""
    text_elements: list[etree._Element] = []
    for nav_point in tree.iter(f"{{{_NS_NCX}}}navPoint"):
        label = nav_point.find(f"{{{_NS_NCX}}}navLabel")
        if label is not None:
            text_el = label.find(f"{{{_NS_NCX}}}text")
            if text_el is not None:
                text_elements.append(text_el)
    if len(text_elements) != len(translated):
        raise ValueError(
            f"NCX has {len(text_elements)} title elements but got {len(translated)} translations"
        )
    for el, title in zip(text_elements, translated):
        # Clear any children and set as single text node.
        for child in list(el):
            el.remove(child)
        el.text = title


def _apply_nav_titles(tree: etree._Element, translated: list[str]) -> None:
    """Replace title text in nav ``<a>`` elements."""
    # Find the toc nav.
    toc_nav = None
    for nav in tree.iter(f"{{{_NS_XHTML}}}nav", "nav"):
        epub_type = nav.get(f"{{{_NS_EPUB}}}type", nav.get("epub:type", ""))
        if "toc" in epub_type.split():
            toc_nav = nav
            break

    if toc_nav is None:
        raise ValueError("No <nav epub:type='toc'> found")

    a_elements = list(toc_nav.iter(f"{{{_NS_XHTML}}}a", "a"))
    if len(a_elements) != len(translated):
        raise ValueError(
            f"Nav has {len(a_elements)} <a> elements but got {len(translated)} translations"
        )
    for el, title in zip(a_elements, translated):
        # Clear children (internal markup) and set as single text node.
        for child in list(el):
            el.remove(child)
        el.text = title


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def translate_toc(
    source_epub_path: str,
    manifest: Manifest,
    glossary: Glossary,
    llm_client: LLMClient,
    config: AppConfig,
) -> dict[str, bytes]:
    """Translate ToC entries and return modified files.

    Finds NCX and/or nav documents, extracts titles, deduplicates,
    translates in one LLM call, and writes translations back into both
    files.

    Parameters
    ----------
    source_epub_path:
        Path to the source EPUB file.
    manifest:
        Book manifest (used for ``opf_dir``).
    glossary:
        Per-book glossary.
    llm_client:
        LLM client for translation.
    config:
        Application config.

    Returns
    -------
    dict[str, bytes]
        Modified ToC files keyed by their ZIP-internal path.
        Empty dict if no ToC files found.
    """
    ncx_path, nav_path = find_toc_files(source_epub_path, manifest.opf_dir)

    if ncx_path is None and nav_path is None:
        logger.warning("No ToC files found (no NCX, no nav document). Skipping ToC translation.")
        return {}

    # Read ToC file contents.
    ncx_content: str | None = None
    nav_content: str | None = None
    ncx_titles: list[str] = []
    nav_titles: list[str] = []

    if ncx_path:
        ncx_bytes = _read_zip_entry(source_epub_path, ncx_path)
        ncx_content = ncx_bytes.decode("utf-8")
        ncx_titles = extract_toc_titles(ncx_content, "ncx")
        logger.info("NCX: extracted %d titles from %s", len(ncx_titles), ncx_path)

    if nav_path:
        nav_bytes = _read_zip_entry(source_epub_path, nav_path)
        nav_content = nav_bytes.decode("utf-8")
        nav_titles = extract_toc_titles(nav_content, "nav")
        logger.info("Nav: extracted %d titles from %s", len(nav_titles), nav_path)

    # Deduplicate titles for a single LLM call.
    seen: set[str] = set()
    unique_titles: list[str] = []
    for t in ncx_titles + nav_titles:
        if t not in seen:
            seen.add(t)
            unique_titles.append(t)

    if not unique_titles:
        logger.warning("No ToC titles found to translate.")
        return {}

    # One LLM call for all unique titles.
    logger.info("Translating %d unique ToC titles", len(unique_titles))
    translated_unique = translate_titles(unique_titles, glossary, llm_client, config)

    # Build lookup map.
    translation_map: dict[str, str] = dict(zip(unique_titles, translated_unique))

    # Apply translations back to each file.
    modified: dict[str, bytes] = {}

    if ncx_content and ncx_path:
        translated_ncx = [translation_map[t] for t in ncx_titles]
        new_ncx = apply_translated_titles(ncx_content, "ncx", translated_ncx)
        modified[ncx_path] = new_ncx.encode("utf-8")
        logger.info("NCX translated: %s", ncx_path)

    if nav_content and nav_path:
        translated_nav = [translation_map[t] for t in nav_titles]
        new_nav = apply_translated_titles(nav_content, "nav", translated_nav)
        modified[nav_path] = new_nav.encode("utf-8")
        logger.info("Nav translated: %s", nav_path)

    return modified


# ---------------------------------------------------------------------------
# OPF metadata updates
# ---------------------------------------------------------------------------


def update_opf_metadata(opf_content: str, config: AppConfig) -> str:
    """Update OPF metadata for the target language.

    Modifications:

    - ``<dc:language>``: set to ``config.languages.target``.
    - ``<dc:title>``: append ``config.output.title_suffix``.
    - ``<dc:description>``: add machine-translation note
      (if ``config.output.add_translation_note``).
    - ``<dc:identifier>``: replace with new UUID
      (if ``config.output.new_identifier``).

    Parameters
    ----------
    opf_content:
        Original OPF file content as a string.
    config:
        Application config.

    Returns
    -------
    str
        Modified OPF content.
    """
    tree = etree.fromstring(
        opf_content.encode("utf-8") if isinstance(opf_content, str) else opf_content
    )

    # --- Update language ---
    for lang_el in tree.iter(f"{{{_NS_DC}}}language"):
        lang_el.text = config.languages.target

    # --- Update title ---
    for title_el in tree.iter(f"{{{_NS_DC}}}title"):
        if title_el.text:
            title_el.text = title_el.text + config.output.title_suffix
        else:
            title_el.text = config.output.title_suffix.strip()
        break  # Only modify the first <dc:title>.

    # --- Add translation note ---
    if config.output.add_translation_note:
        model_name = config.models.translate.model or "unknown model"
        note = (
            f"Machine translated by dao-bridge-translator using {model_name}. "
            "Not professionally edited."
        )
        # Find or create <dc:description>.
        metadata_el = tree.find(f"{{{_NS_OPF}}}metadata")
        if metadata_el is None:
            # Very unusual but handle gracefully.
            logger.warning("No <metadata> element in OPF, skipping translation note")
        else:
            desc_el = metadata_el.find(f"{{{_NS_DC}}}description")
            if desc_el is None:
                desc_el = etree.SubElement(metadata_el, f"{{{_NS_DC}}}description")
            desc_el.text = note

    # --- Replace identifier ---
    if config.output.new_identifier:
        new_id = f"urn:uuid:{uuid.uuid4()}"
        for id_el in tree.iter(f"{{{_NS_DC}}}identifier"):
            id_el.text = new_id
            break  # Only replace the first identifier.

    return etree.tostring(
        tree,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    ).decode("utf-8")
