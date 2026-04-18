"""HTML cleaning and markdown conversion pipeline.

For each ``raw/NNN.xhtml`` file, this module:

1. Parses with BeautifulSoup (lxml).
2. Pre-processes ruby annotations into ``{base|reading}`` notation.
3. Strips scripts, styles, and purely presentational elements.
4. Converts to markdown with ``markdownify``.
5. Normalises whitespace.
6. Writes to ``clean/NNN.md``.
7. Counts paragraphs and tokens for the manifest.
"""

from __future__ import annotations

import logging
import re
import warnings

import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from markdownify import MarkdownConverter

from dao_bridge.config import AppConfig
from dao_bridge.schemas import Manifest
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
    clean_path,
    manifest_path,
    pad_spine,
    raw_path,
)

# We intentionally use the HTML parser ("lxml") for XHTML content to handle
# messy real-world EPUB markup (Calibre, Kobo, etc.).
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger("dao_bridge")

# Cache the tokeniser so we don't reload it per file.
_tokeniser: tiktoken.Encoding | None = None


def _get_tokeniser() -> tiktoken.Encoding:
    global _tokeniser
    if _tokeniser is None:
        _tokeniser = tiktoken.get_encoding("cl100k_base")
    return _tokeniser


# ---------------------------------------------------------------------------
# Custom markdownify converter
# ---------------------------------------------------------------------------


class _Converter(MarkdownConverter):
    """Customised converter that preserves line-breaks and scene-break markers."""

    def convert_br(self, el: Tag, text: str, **kwargs) -> str:  # noqa: ARG002
        return "  \n"

    def convert_hr(self, el: Tag, text: str, **kwargs) -> str:  # noqa: ARG002
        return "\n\n* * *\n\n"

    def convert_img(self, el: Tag, text: str, **kwargs) -> str:  # noqa: ARG002
        alt = el.get("alt", "")
        src = el.get("src", "")
        return f"![{alt}]({src})"


def _md(html: str) -> str:
    """Convert an HTML string to markdown using the custom converter."""
    return _Converter(
        heading_style="atx",
        strip=["script", "style"],
    ).convert(html)


# ---------------------------------------------------------------------------
# Ruby pre-processing
# ---------------------------------------------------------------------------


def _process_ruby(soup: BeautifulSoup) -> None:
    """Replace ``<ruby>`` elements with ``{base|reading}`` plain text.

    Handles:
    - Simple: ``<ruby>漢字<rt>かんじ</rt></ruby>``
    - With ``<rb>``: ``<ruby><rb>漢字</rb><rt>かんじ</rt></ruby>``
    - With ``<rp>`` fallback parens (stripped).
    - Kobo-injected ``<span class="koboSpan">`` wrappers inside ruby elements.
    - Multi-segment ruby: ``<ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>``
    """
    for ruby in soup.find_all("ruby"):
        # Collect base/reading pairs from this ruby element.
        pairs = _extract_ruby_pairs(ruby)
        if pairs:
            # Build combined {base|reading} text.
            parts = []
            for base_text, reading_text in pairs:
                if reading_text:
                    parts.append(f"{{{base_text}|{reading_text}}}")
                else:
                    parts.append(base_text)
            replacement = "".join(parts)
        else:
            # Fallback: just extract text.
            replacement = ruby.get_text()

        ruby.replace_with(replacement)


def _extract_ruby_pairs(ruby: Tag) -> list[tuple[str, str]]:
    """Extract ``(base, reading)`` pairs from a ``<ruby>`` element."""
    pairs: list[tuple[str, str]] = []
    current_base_parts: list[str] = []

    for child in ruby.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                current_base_parts.append(text)
        elif isinstance(child, Tag):
            tag_name = child.name.lower() if child.name else ""
            if tag_name == "rp":
                # Skip fallback parentheses.
                continue
            elif tag_name == "rt":
                reading = child.get_text().strip()
                base = "".join(current_base_parts).strip()
                if base or reading:
                    pairs.append((base, reading))
                current_base_parts = []
            elif tag_name == "rb":
                # Explicit base element.
                current_base_parts.append(child.get_text().strip())
            else:
                # Other elements (e.g. koboSpan) — extract their text as base.
                current_base_parts.append(child.get_text().strip())

    # If there's trailing base text with no reading, include it.
    trailing = "".join(current_base_parts).strip()
    if trailing:
        pairs.append((trailing, ""))

    return pairs


# ---------------------------------------------------------------------------
# KoboSpan stripping
# ---------------------------------------------------------------------------


def _strip_kobo_spans(soup: BeautifulSoup) -> None:
    """Unwrap ``<span class="koboSpan">`` elements, keeping their text content.

    Kobo injects these spans into EPUB content for reading position tracking.
    They must be removed before further processing but their text kept.
    This handles koboSpan elements both inside and outside ruby elements.
    """
    for span in soup.find_all("span", class_="koboSpan"):
        span.unwrap()

    # After unwrapping, adjacent NavigableStrings may need merging.
    # BeautifulSoup handles this on next traversal, but we can force it
    # with smooth() if available (BS4 >= 4.12.3).
    if hasattr(soup, "smooth"):
        soup.smooth()


# ---------------------------------------------------------------------------
# Presentational element stripping
# ---------------------------------------------------------------------------


def _strip_presentational(soup: BeautifulSoup) -> None:
    """Remove purely presentational elements.

    Strips:
    - ``<style>`` and ``<script>`` tags entirely.
    - Empty ``<div>`` and ``<span>`` elements (no text content).
    - Elements with only ``class``/``id`` attributes and no semantic tag
      name — these are unwrapped (children kept, wrapper removed).
    """
    # Remove style/script.
    for tag in soup.find_all(["style", "script"]):
        tag.decompose()

    # Remove xmlns namespace declarations injected by Kobo.
    for tag in soup.find_all(True):
        if tag.attrs:
            tag.attrs = {k: v for k, v in tag.attrs.items() if k != "xmlns"}

    # Unwrap purely presentational divs/spans.
    _PRESENTATIONAL_TAGS = {"div", "span"}
    _PRESENTATIONAL_ATTRS = {"class", "id", "style"}

    # Multiple passes since unwrapping can reveal new empty wrappers.
    for _ in range(3):
        changed = False
        for tag in soup.find_all(_PRESENTATIONAL_TAGS):
            # Skip if it has meaningful attributes beyond class/id/style.
            attr_keys = set(tag.attrs.keys()) if tag.attrs else set()
            has_semantic_attrs = bool(attr_keys - _PRESENTATIONAL_ATTRS)
            if has_semantic_attrs:
                continue

            text = tag.get_text(strip=True)
            if not text and not tag.find(["img", "svg", "image"]):
                # Truly empty — remove entirely.
                tag.decompose()
                changed = True
            elif tag.name == "div" and attr_keys and not has_semantic_attrs:
                # Div with only class/id — unwrap to flatten.
                # But only if it doesn't add semantic structure (e.g. not the
                # only child of body).
                if tag.parent and tag.parent.name != "body":
                    tag.unwrap()
                    changed = True

        if not changed:
            break


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

# Matches 3 or more consecutive newlines (possibly with whitespace-only lines).
_MULTI_BLANK_RE = re.compile(r"\n\s*\n\s*\n(\s*\n)*")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to exactly two newlines, strip trailing whitespace."""
    # Collapse 3+ blank lines to 2 (one blank line between paragraphs).
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    # Strip trailing whitespace from each line.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Strip leading/trailing whitespace from the whole document.
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Paragraph / token counting
# ---------------------------------------------------------------------------


def _count_paragraphs(md_text: str) -> int:
    """Count paragraphs (blocks separated by blank lines)."""
    blocks = [b.strip() for b in md_text.split("\n\n") if b.strip()]
    return len(blocks)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    enc = _get_tokeniser()
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------


def clean_spine_item(raw_xhtml: str) -> str:
    """Clean a single raw XHTML string and return markdown.

    This is the core cleaning function that can be used independently of
    the pipeline orchestration for testing.
    """
    soup = BeautifulSoup(raw_xhtml, "lxml")

    # 1. Strip Kobo-injected spans (must happen before ruby processing
    #    since koboSpans appear inside <ruby> elements).
    _strip_kobo_spans(soup)

    # 2. Process ruby annotations.
    _process_ruby(soup)

    # 3. Strip presentational elements, scripts, styles.
    _strip_presentational(soup)

    # 4. Convert to markdown.
    # Extract just the body content if present.
    body = soup.find("body")
    html_str = str(body) if body else str(soup)
    md = _md(html_str)

    # 5. Normalise whitespace.
    md = _normalize_whitespace(md)

    return md


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def clean_all(
    config: AppConfig,
    manifest: Manifest,
    state: PipelineState,
    *,
    force: bool = False,
) -> Manifest:
    """Clean all extracted spine items.

    Parameters
    ----------
    config:
        Application configuration.
    manifest:
        The manifest (mutated in place with clean paths and counts).
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, re-clean even if the stage is already completed.

    Returns
    -------
    Manifest
        The updated manifest.
    """
    work_dir = config.work_dir_path
    sw = manifest.spine_padding_width

    if not force and is_stage_completed(state, "clean"):
        logger.info("Clean stage already completed — skipping (use --force to re-run)")
        return manifest

    if force:
        reset_stage(work_dir, state, "clean")

    mark_stage_started(work_dir, state, "clean")

    for item in manifest.spine:
        padded = pad_spine(item.spine_index, sw)
        mark_item_started(work_dir, state, "clean", padded)

        rp = raw_path(work_dir, item.spine_index, sw)
        if not rp.exists():
            logger.warning("Raw file missing for spine %s: %s", padded, rp)
            continue

        raw_xhtml = rp.read_text(encoding="utf-8")
        md = clean_spine_item(raw_xhtml)

        cp = clean_path(work_dir, item.spine_index, sw)
        cp.parent.mkdir(parents=True, exist_ok=True)

        # Count before writing.
        para_count = _count_paragraphs(md)
        token_count = _count_tokens(md)

        # IMPORTANT: Data writes (clean file) MUST happen before status
        # writes (item completion).  If we crash between the two, the
        # item will simply be re-cleaned on resume (idempotent).  The
        # reverse — marking complete before the file is written — would
        # leave a gap that a resumed run silently skips.
        atomic_write(cp, md)

        # Update manifest item.
        item.clean_path = str(cp.relative_to(work_dir))
        item.paragraph_count = para_count
        item.token_count = token_count

        mark_item_completed(work_dir, state, "clean", padded)
        logger.debug(
            "Cleaned spine %s: %d paragraphs, %d tokens",
            padded,
            para_count,
            token_count,
        )

    # Persist updated manifest.
    # IMPORTANT: Manifest data write before stage-completion status write.
    mp = manifest_path(work_dir)
    atomic_write(mp, manifest.model_dump_json(indent=2))

    mark_stage_completed(work_dir, state, "clean")
    logger.info("Cleaning complete: %d items processed", len(manifest.spine))

    return manifest
