"""EPUB rebuild via modified copy of the source archive.

The output EPUB is constructed by copying the original source EPUB at the
ZIP level and replacing only the files that changed.  This preserves all
original structure: images, fonts, CSS, DRM metadata, Apple/Kindle-specific
files, custom OPF entries, archive compression settings, and everything
else we don't need to touch.

**Do NOT use ``ebooklib.write_epub()`` for output.**  It reconstructs the
EPUB from its internal model and may lose non-standard entries, custom
namespaces, or unusual archive structure.
"""

from __future__ import annotations

import copy
import json
import logging
import posixpath
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from lxml import etree

from dao_bridge.config import AppConfig
from dao_bridge.schemas import Glossary, Manifest
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    load_state,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_started,
    reset_stage,
)
from dao_bridge.toc import (
    _find_opf_zip_path,
    _read_zip_entry,
    translate_toc,
    update_opf_metadata,
)
from dao_bridge.workdir import (
    assembled_path,
    glossary_path,
    manifest_path,
    resolve_zip_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Templates directory
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Ruby-safe markdown -> HTML conversion
# ---------------------------------------------------------------------------

# Matches {kanji|reading} notation produced by the clean stage.
_RUBY_RE = re.compile(r"\{([^|{}]+)\|([^|{}]+)\}")

# Matches scene-break markers that markdown may wrap in <p> tags
# (fallback for non-standard markdown processors).
_SCENE_BREAK_P_RE = re.compile(r"<p>\s*\*\s*\*\s*\*\s*</p>")

# Matches <hr> / <hr /> / <hr/> variants for normalisation.
_HR_RE = re.compile(r"<hr\s*/?>")

# Matches non-self-closing <br> tags.
_BR_RE = re.compile(r"<br\s*/?>")


def markdown_to_html(md_text: str) -> str:
    """Convert markdown to HTML with a ruby-safe pipeline.

    1. Replace ``{kanji|reading}`` with unique placeholders to prevent
       the ``attr_list`` extension (included in ``extra``) from mangling
       the ``{...}`` syntax.
    2. Run ``markdown.markdown()`` with the ``extra`` extension.
    3. Normalise scene breaks: convert ``<p>* * *</p>`` and ``<hr />``
       variants to ``<hr/>``.
    4. Ensure ``<br>`` tags are self-closing (``<br/>``).
    5. Restore placeholders to ``<ruby>`` tags.

    Parameters
    ----------
    md_text:
        Markdown text (may contain ``{kanji|reading}`` notation).

    Returns
    -------
    str
        HTML string.
    """
    # Step 1: Replace ruby notation with placeholders.
    placeholder_map: dict[str, tuple[str, str]] = {}
    counter = 0

    def _make_placeholder(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        key = f"RUBYDBT{counter:04d}"
        placeholder_map[key] = (m.group(1), m.group(2))
        return key

    processed = _RUBY_RE.sub(_make_placeholder, md_text)

    # Step 2: Markdown -> HTML.
    html = markdown.markdown(processed, extensions=["extra"])

    # Step 3: Scene breaks.
    html = _SCENE_BREAK_P_RE.sub("<hr/>", html)
    html = _HR_RE.sub("<hr/>", html)

    # Step 4: Self-close <br> tags.
    html = _BR_RE.sub("<br/>", html)

    # Step 5: Restore ruby tags.
    html = restore_ruby_tags(html, placeholder_map)

    return html


def restore_ruby_tags(html: str, placeholder_map: dict[str, tuple[str, str]]) -> str:
    """Replace ``RUBYDBT_NNNN`` placeholders with ``<ruby>`` tags.

    Parameters
    ----------
    html:
        HTML string containing placeholders.
    placeholder_map:
        Mapping of placeholder keys to ``(kanji, reading)`` tuples.

    Returns
    -------
    str
        HTML with ``<ruby>kanji<rt>reading</rt></ruby>`` tags.
    """
    for key, (kanji, reading) in placeholder_map.items():
        html = html.replace(key, f"<ruby>{kanji}<rt>{reading}</rt></ruby>")
    return html


# ---------------------------------------------------------------------------
# XHTML body replacement
# ---------------------------------------------------------------------------


def replace_xhtml_body(original_xhtml: str, translated_markdown: str) -> str:
    """Replace the ``<body>`` children in *original_xhtml* with translated content.

    Preserves the original ``<head>`` entirely and the ``<body>`` tag's
    own attributes.  Only the body's children are replaced.

    Uses the ``lxml-xml`` parser (not ``lxml`` HTML mode) to preserve
    XML declarations, namespace declarations, and self-closing tag syntax
    required for valid XHTML in EPUBs.

    Parameters
    ----------
    original_xhtml:
        The original XHTML content from the source EPUB.
    translated_markdown:
        The translated markdown from the assembled stage.

    Returns
    -------
    str
        Modified XHTML string with translated body content.
    """
    # Parse original with XML parser to preserve structure.
    soup = BeautifulSoup(original_xhtml, "lxml-xml")
    body = soup.find("body")
    if body is None:
        raise ValueError("No <body> element found in original XHTML")

    # Convert markdown to HTML.
    translated_html = markdown_to_html(translated_markdown)

    # Parse the translated HTML as a fragment.
    # Wrap in a temporary body to get consistent structure.
    fragment_html = f"<html><body>{translated_html}</body></html>"
    fragment_soup = BeautifulSoup(fragment_html, "html.parser")
    fragment_body = fragment_soup.find("body")

    # Clear original body children but keep attributes.
    body.clear()

    # Transplant children from fragment into original body.
    if fragment_body:
        # Need to collect children first since we're modifying the tree.
        children = list(fragment_body.children)
        for child in children:
            body.append(child)

    return str(soup)


def inject_default_css_link(xhtml_str: str, css_href: str) -> str:
    """Add a ``<link>`` element for the default CSS to the ``<head>``.

    Parameters
    ----------
    xhtml_str:
        XHTML content string.
    css_href:
        Relative href to the CSS file from this XHTML file's location.

    Returns
    -------
    str
        Modified XHTML with the CSS link added.
    """
    soup = BeautifulSoup(xhtml_str, "lxml-xml")
    head = soup.find("head")
    if head is None:
        logger.warning("No <head> element found, cannot inject CSS link")
        return xhtml_str

    link = soup.new_tag("link", rel="stylesheet", type="text/css", href=css_href)
    head.append(link)
    return str(soup)


# ---------------------------------------------------------------------------
# Build modified files
# ---------------------------------------------------------------------------


def build_modified_files(
    manifest: Manifest,
    work_dir: Path,
    source_epub_path: str,
    config: AppConfig,
) -> dict[str, bytes]:
    """Build the dict of modified files for body-replaced spine items.

    For each spine item with ``chunk_count > 0`` (translated content):
    read the original XHTML from the source EPUB, replace its body with
    the assembled translated markdown converted to HTML.

    Parameters
    ----------
    manifest:
        Book manifest.
    work_dir:
        Working directory path.
    source_epub_path:
        Path to the source EPUB file.
    config:
        Application config.

    Returns
    -------
    dict[str, bytes]
        Mapping of ZIP-internal paths to new file content (bytes).
    """
    modified: dict[str, bytes] = {}
    sw = manifest.spine_padding_width

    # Determine default CSS zip path and href if needed.
    css_zip_path: str | None = None
    if config.output.css == "default":
        css_zip_path = _resolve_css_zip_path(manifest.opf_dir)

    with zipfile.ZipFile(source_epub_path, "r") as zf:
        for item in manifest.spine:
            if not item.chunk_count or item.chunk_count <= 0:
                continue  # Not translated; copies through unchanged.

            zip_path = resolve_zip_path(manifest.opf_dir, item.original_href)

            # Read original XHTML.
            try:
                original_bytes = zf.read(zip_path)
            except KeyError:
                logger.error("ZIP entry not found: %s (from href %s)", zip_path, item.original_href)
                raise

            original_xhtml = original_bytes.decode("utf-8")

            # Read assembled markdown.
            asm_path = assembled_path(work_dir, item.spine_index, sw)
            if not asm_path.exists():
                raise FileNotFoundError(
                    f"Assembled file missing for spine {item.spine_index}: {asm_path}"
                )
            assembled_md = asm_path.read_text(encoding="utf-8")

            # Replace body.
            new_xhtml = replace_xhtml_body(original_xhtml, assembled_md)

            # Inject default CSS link if configured.
            if config.output.css == "default" and css_zip_path:
                # Compute relative href from this XHTML's directory to the CSS file.
                xhtml_dir = posixpath.dirname(zip_path)
                css_href = posixpath.relpath(css_zip_path, xhtml_dir)
                new_xhtml = inject_default_css_link(new_xhtml, css_href)

            modified[zip_path] = new_xhtml.encode("utf-8")
            logger.info("Body replaced: %s (spine %d)", zip_path, item.spine_index)

    return modified


def _resolve_css_zip_path(opf_dir: str) -> str:
    """Determine the ZIP path for the injected default CSS."""
    if opf_dir:
        return posixpath.join(opf_dir, "dao_bridge_default.css")
    return "dao_bridge_default.css"


# ---------------------------------------------------------------------------
# Write EPUB (ZIP-level modified copy)
# ---------------------------------------------------------------------------


def write_epub_modified_copy(
    source_epub: str,
    output_epub: str,
    modified_files: dict[str, bytes],
) -> None:
    """Create an output EPUB by copying the source and replacing modified files.

    Preserves per-entry ``ZipInfo`` metadata (``compress_type``, etc.)
    from the source.  Ensures the ``mimetype`` entry is first in the
    archive, stored uncompressed with no extra field (EPUB spec
    requirement).

    Parameters
    ----------
    source_epub:
        Path to the source EPUB file.
    output_epub:
        Path for the output EPUB file.
    modified_files:
        Mapping of ZIP-internal paths to new content bytes.
        Entries not in the source ZIP are added as new files.
    """
    # Ensure output directory exists.
    Path(output_epub).parent.mkdir(parents=True, exist_ok=True)

    # Track which modified_files entries are new (not in source).
    source_paths: set[str] = set()

    with zipfile.ZipFile(source_epub, "r") as src:
        with zipfile.ZipFile(output_epub, "w") as dst:
            # Phase 1: Write mimetype first (EPUB spec requirement).
            mimetype_info = None
            for item in src.infolist():
                if item.filename == "mimetype":
                    mimetype_info = item
                    break

            if mimetype_info is not None:
                new_info = zipfile.ZipInfo("mimetype")
                new_info.compress_type = zipfile.ZIP_STORED
                new_info.extra = b""
                if "mimetype" in modified_files:
                    dst.writestr(new_info, modified_files["mimetype"])
                else:
                    dst.writestr(new_info, src.read("mimetype"))
                source_paths.add("mimetype")

            # Phase 2: Copy/replace all other entries.
            for item in src.infolist():
                source_paths.add(item.filename)
                if item.filename == "mimetype":
                    continue  # Already written.

                new_info = copy.copy(item)
                new_info.header_offset = 0  # Reset; written by ZipFile.

                if item.filename in modified_files:
                    # Content changed -- wipe extra in case any fields
                    # reference the old payload (e.g. compressed size hints).
                    new_info.extra = b""
                    dst.writestr(new_info, modified_files[item.filename])
                else:
                    # Byte-identical copy.
                    dst.writestr(new_info, src.read(item.filename))

            # Phase 3: Add new files not in source (e.g., default.css).
            for path, content in modified_files.items():
                if path not in source_paths:
                    info = zipfile.ZipInfo(path)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    dst.writestr(info, content)

    logger.info("Wrote output EPUB: %s", output_epub)


# ---------------------------------------------------------------------------
# Epubcheck validation
# ---------------------------------------------------------------------------


def validate_with_epubcheck(epub_path: str) -> bool:
    """Run ``epubcheck`` on the output EPUB if available.

    This is a non-blocking safety net -- validation failure does not
    prevent the EPUB from being written.

    Parameters
    ----------
    epub_path:
        Path to the EPUB file to validate.

    Returns
    -------
    bool
        ``True`` if validation passed or epubcheck was not found.
        ``False`` if validation failed.
    """
    epubcheck = shutil.which("epubcheck")
    if epubcheck is None:
        logger.warning("epubcheck not found on PATH, skipping validation")
        return True

    logger.info("Running epubcheck on %s", epub_path)
    try:
        result = subprocess.run(
            [epubcheck, epub_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.stdout:
            logger.info("epubcheck stdout:\n%s", result.stdout)
        if result.stderr:
            logger.info("epubcheck stderr:\n%s", result.stderr)

        if result.returncode == 0:
            logger.info("epubcheck: PASSED")
            return True
        else:
            logger.warning("epubcheck: FAILED (exit code %d)", result.returncode)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("epubcheck timed out after 300 seconds")
        return False
    except Exception as e:
        logger.warning("epubcheck failed to run: %s", e)
        return True  # Non-blocking.


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_rebuild_stage(
    work_dir: Path,
    config: AppConfig,
    force: bool = False,
) -> None:
    """Run the rebuild stage: produce the output EPUB.

    Validates that all translatable spine items have been assembled,
    then builds the output EPUB by copying the source and replacing
    translated XHTML bodies, ToC entries, and metadata.

    Parameters
    ----------
    work_dir:
        Working directory path.
    config:
        Application config.
    force:
        If ``True``, re-run even if the stage was already completed.
    """
    state = load_state(work_dir)

    if is_stage_completed(state, "rebuild") and not force:
        logger.info("Rebuild stage already completed, skipping (use --force to re-run)")
        return

    if force:
        reset_stage(work_dir, state, "rebuild")

    mark_stage_started(work_dir, state, "rebuild")

    try:
        _do_rebuild(work_dir, config, state)
        mark_stage_completed(work_dir, state, "rebuild")
    except Exception as e:
        mark_stage_failed(work_dir, state, "rebuild", str(e))
        raise


def _do_rebuild(work_dir: Path, config: AppConfig, state: PipelineState) -> None:
    """Core rebuild logic (called by the stage runner)."""
    # Load manifest.
    mp = manifest_path(work_dir)
    if not mp.exists():
        raise RuntimeError(f"Manifest not found at {mp}")
    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))

    # Validate source EPUB.
    source_epub = str(config.source_epub_path)
    if not Path(source_epub).exists():
        raise FileNotFoundError(f"Source EPUB not found: {source_epub}")

    # Validate all translatable items have assembled files.
    sw = manifest.spine_padding_width
    missing: list[str] = []
    for item in manifest.spine:
        if item.chunk_count and item.chunk_count > 0:
            asm = assembled_path(work_dir, item.spine_index, sw)
            if not asm.exists():
                missing.append(f"  spine {item.spine_index}: {asm}")
    if missing:
        raise FileNotFoundError(
            f"Missing assembled files for {len(missing)} spine item(s):\n" + "\n".join(missing)
        )

    # --- Build modified files (body replacement) ---
    logger.info("Replacing body content for translated spine items")
    modified = build_modified_files(manifest, work_dir, source_epub, config)
    logger.info("Body replacement complete: %d files modified", len(modified))

    # --- Translate ToC ---
    from dao_bridge.llm_client import LLMClient

    llm_client = LLMClient(config.models.translate, config.llm)
    gp = glossary_path(work_dir)
    glossary = Glossary(
        created_at="2000-01-01T00:00:00Z",
        updated_at="2000-01-01T00:00:00Z",
    )
    if gp.exists():
        glossary = Glossary(**json.loads(gp.read_text(encoding="utf-8")))

    toc_modified = translate_toc(source_epub, manifest, glossary, llm_client, config)
    modified.update(toc_modified)
    logger.info("ToC translation: %d files modified", len(toc_modified))

    # --- Update OPF metadata ---
    opf_zip_path = _find_opf_zip_path(source_epub)
    opf_content = _read_zip_entry(source_epub, opf_zip_path).decode("utf-8")
    new_opf = update_opf_metadata(opf_content, config)

    # If default CSS, add it to the OPF manifest and the modified files.
    if config.output.css == "default":
        css_zip_path = _resolve_css_zip_path(manifest.opf_dir)
        css_content = (_TEMPLATES_DIR / "default.css").read_text(encoding="utf-8")
        modified[css_zip_path] = css_content.encode("utf-8")
        new_opf = _add_css_to_opf(new_opf, manifest.opf_dir, css_zip_path)
        logger.info("Default CSS added: %s", css_zip_path)

    modified[opf_zip_path] = new_opf.encode("utf-8")
    logger.info("OPF metadata updated: %s", opf_zip_path)

    # --- Resolve output path ---
    output_path = _resolve_output_path(work_dir, config)

    # --- Write output EPUB ---
    write_epub_modified_copy(source_epub, str(output_path), modified)
    logger.info("Output EPUB written: %s", output_path)

    # --- Optional validation ---
    if config.output.validate_epub:
        passed = validate_with_epubcheck(str(output_path))
        if not passed:
            logger.warning("EPUB validation failed (output still written)")


def _resolve_output_path(work_dir: Path, config: AppConfig) -> Path:
    """Resolve the output EPUB path.

    If ``config.output.epub_path`` is relative, resolve it relative to
    the work directory's parent (so ``./book.en.epub`` produces the
    output alongside the work directory, not inside it).
    """
    p = Path(config.output.epub_path)
    if p.is_absolute():
        return p
    return (work_dir.parent / p).resolve()


def _add_css_to_opf(opf_content: str, opf_dir: str, css_zip_path: str) -> str:
    """Add a ``<item>`` entry for the default CSS to the OPF manifest."""
    tree = etree.fromstring(
        opf_content.encode("utf-8") if isinstance(opf_content, str) else opf_content
    )

    manifest_el = tree.find(f"{{{_NS_OPF}}}manifest")
    if manifest_el is None:
        logger.warning("No <manifest> element in OPF, cannot add CSS item")
        return opf_content

    # Compute href relative to OPF directory.
    if opf_dir:
        href = posixpath.relpath(css_zip_path, opf_dir)
    else:
        href = css_zip_path

    item = etree.SubElement(manifest_el, f"{{{_NS_OPF}}}item")
    item.set("id", "dao-bridge-default-css")
    item.set("href", href)
    item.set("media-type", "text/css")

    return etree.tostring(
        tree,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    ).decode("utf-8")


# Re-use namespace constant from toc module.
_NS_OPF = "http://www.idpf.org/2007/opf"
