"""Spine item classification with structural hints and LLM fallback.

Three-layer classification strategy:

1. **Structural hints** — deterministic XHTML inspection, no LLM needed.
2. **LLM classification** — sends excerpts to an LLM for items not resolved
   by layer 1.
3. **Manual override** — users can edit ``manifest.json`` directly; existing
   non-null classifications are preserved unless ``--force`` is used.

Classification values (from :data:`~dao_bridge.schemas.Classification`):
``chapter``, ``frontmatter``, ``backmatter``, ``toc_auto``,
``toc_authored``, ``illustration``, ``unknown``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from dao_bridge.config import AppConfig, resolve_language_name
from dao_bridge.llm_client import LLMClient, LLMStructuredOutputError
from dao_bridge.schemas import ClassificationResponse, Manifest, ManifestItem
from dao_bridge.state import (
    PipelineState,
    is_stage_completed,
    iter_pending_items,
    mark_item_completed,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
    reset_stage,
)
from dao_bridge.workdir import atomic_write, manifest_path, pad_spine

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_EXCERPT_CHARS = 500
_CLEAN_EXCERPT_TOKENS = 1500  # truncate clean excerpt to ~1500 tokens

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "classify.txt"


# Threshold for the illustration structural hint (token count).
_ILLUSTRATION_MAX_TOKENS = 30

# Threshold for the frontmatter/title-only structural hint.
# Non-heading visible text must be under this many words.
_FRONTMATTER_MAX_NON_HEADING_WORDS = 10

# ---------------------------------------------------------------------------
# Tokeniser (cached)
# ---------------------------------------------------------------------------

_tokeniser: tiktoken.Encoding | None = None


def _get_tokeniser() -> tiktoken.Encoding:
    global _tokeniser  # noqa: PLW0603
    if _tokeniser is None:
        _tokeniser = tiktoken.get_encoding("cl100k_base")
    return _tokeniser


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_get_tokeniser().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens."""
    enc = _get_tokeniser()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Result of classifying a single spine item."""

    classification: str  # one of the 7 Classification literal values
    title: str | None
    confidence: str  # "high" | "medium" | "low"
    reasoning: str
    source: str  # "structural" | "llm"


# ---------------------------------------------------------------------------
# HTML text extraction helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(
    r"<(h[1-6])\b[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_tags(html: str) -> str:
    """Remove all HTML/XML tags, returning visible text only."""
    return _TAG_RE.sub("", html)


def _extract_headings(html: str) -> list[str]:
    """Extract text content of all ``<h1>``–``<h6>`` elements."""
    results = []
    for match in _HEADING_RE.finditer(html):
        text = _strip_tags(match.group(2)).strip()
        if text:
            results.append(text)
    return results


def _visible_text(html: str) -> str:
    """Return visible text from HTML after stripping tags.

    Also collapses runs of whitespace into single spaces.
    """
    text = _strip_tags(html)
    return re.sub(r"\s+", " ", text).strip()


def _non_heading_text(html: str) -> str:
    """Return visible text with all heading elements removed first."""
    no_headings = _HEADING_RE.sub("", html)
    return _visible_text(no_headings)


# ---------------------------------------------------------------------------
# Layer 1: Structural hints
# ---------------------------------------------------------------------------


def apply_structural_hints(
    raw_xhtml: str,
    clean_markdown: str,
) -> ClassificationResult | None:
    """Inspect raw XHTML for deterministic classification signals.

    Returns a :class:`ClassificationResult` if a hint matches, or ``None``
    if the item needs LLM classification.

    **Hints checked (in order):**

    1. ``epub:type="toc"`` or ``<nav epub:type="toc">`` → ``toc_auto``
    2. Visible text < 30 words AND contains ``<img`` → ``illustration``
    3. Only a heading with < 10 words of non-heading text → ``frontmatter``
    """
    # --- Hint 1: ToC nav ---
    if re.search(r'epub:type\s*=\s*["\']toc["\']', raw_xhtml, re.IGNORECASE):
        logger.info("Structural hint: epub:type='toc' detected → toc_auto")
        return ClassificationResult(
            classification="toc_auto",
            title="Table of Contents",
            confidence="high",
            reasoning="epub:type='toc' attribute found in XHTML",
            source="structural",
        )

    # --- Hint 2: Illustration ---
    visible = _visible_text(raw_xhtml)
    token_count = _count_tokens(visible) if visible else 0

    if token_count < _ILLUSTRATION_MAX_TOKENS and re.search(r"<img\b", raw_xhtml, re.IGNORECASE):
        logger.info(
            "Structural hint: image with < %d tokens → illustration", _ILLUSTRATION_MAX_TOKENS
        )
        return ClassificationResult(
            classification="illustration",
            title=None,
            confidence="high",
            reasoning=(
                f"File has only {token_count} tokens of visible text and contains an <img> tag"
            ),
            source="structural",
        )

    # --- Hint 3: Frontmatter / title-only page ---
    headings = _extract_headings(raw_xhtml)
    non_heading = _non_heading_text(raw_xhtml)
    non_heading_words = len(non_heading.split()) if non_heading else 0

    if headings and non_heading_words < _FRONTMATTER_MAX_NON_HEADING_WORDS:
        title = headings[0]
        logger.info(
            "Structural hint: heading-only page ('%s', %d non-heading words) → frontmatter",
            title,
            non_heading_words,
        )
        return ClassificationResult(
            classification="frontmatter",
            title=title,
            confidence="high",
            reasoning=(
                f"Page contains heading '{title}' with only "
                f"{non_heading_words} words of non-heading text"
            ),
            source="structural",
        )

    return None


# ---------------------------------------------------------------------------
# Layer 2: LLM classification
# ---------------------------------------------------------------------------


def _load_prompt_template() -> str:
    """Load the classify prompt template from disk."""
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def llm_classify(
    raw_excerpt: str,
    clean_excerpt: str,
    position: tuple[int, int],
    config: AppConfig,
    llm_client: LLMClient,
) -> ClassificationResult:
    """Classify a spine item using the LLM.

    Parameters
    ----------
    raw_excerpt:
        First ~500 characters of the raw XHTML.
    clean_excerpt:
        First ~6000 characters (~1500 tokens) of the cleaned markdown.
    position:
        ``(spine_index, total_spine_items)`` for positional context.
    config:
        Application configuration (used for language info).
    llm_client:
        Pre-initialised LLM client for the classify model.

    Returns
    -------
    ClassificationResult
        The LLM's classification with ``source="llm"``.
    """
    template = _load_prompt_template()
    source_lang = resolve_language_name(config.languages.source)

    prompt = template.format(
        source_language=source_lang,
        raw_excerpt=raw_excerpt,
        clean_excerpt=clean_excerpt,
        spine_position=position[0],
        total_spine_items=position[1],
    )

    messages = [{"role": "user", "content": prompt}]
    response: ClassificationResponse = llm_client.complete_json(
        messages,
        response_model=ClassificationResponse,
    )

    return ClassificationResult(
        classification=response.classification,
        title=response.title,
        confidence=response.confidence,
        reasoning=response.reasoning,
        source="llm",
    )


# ---------------------------------------------------------------------------
# Per-item classification
# ---------------------------------------------------------------------------


def classify_item(
    item: ManifestItem,
    raw_xhtml: str,
    clean_markdown: str,
    position: tuple[int, int],
    config: AppConfig,
    llm_client: LLMClient,
) -> ClassificationResult:
    """Classify a single spine item using structural hints then LLM fallback.

    Parameters
    ----------
    item:
        The manifest item being classified.
    raw_xhtml:
        Full raw XHTML content of the spine item.
    clean_markdown:
        Full cleaned markdown content (may be empty if clean stage skipped).
    position:
        ``(spine_index, total_spine_items)``.
    config:
        Application configuration.
    llm_client:
        LLM client (only called if structural hints don't match).

    Returns
    -------
    ClassificationResult
        Classification result from either structural hints or LLM.
    """
    # Layer 1: structural hints.
    hint = apply_structural_hints(raw_xhtml, clean_markdown)
    if hint is not None:
        return hint

    # Layer 2: LLM classification.
    raw_excerpt = raw_xhtml[:_RAW_EXCERPT_CHARS]
    clean_excerpt = _truncate_to_tokens(clean_markdown, _CLEAN_EXCERPT_TOKENS)

    result = llm_classify(raw_excerpt, clean_excerpt, position, config, llm_client)

    if result.confidence == "low":
        logger.warning(
            "Spine %d classified as '%s' with LOW confidence — review recommended. Reason: %s",
            item.spine_index,
            result.classification,
            result.reasoning,
        )

    return result


# ---------------------------------------------------------------------------
# Stage orchestrator
# ---------------------------------------------------------------------------


def run_classify_stage(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    *,
    force: bool = False,
    spine_filter: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Manifest:
    """Run the classification stage over all (or filtered) spine items.

    Iterates the manifest, classifies each item that needs it, and
    updates the manifest atomically.

    Parameters
    ----------
    work_dir:
        Resolved work directory path.
    config:
        Application configuration.
    state:
        Pipeline state (mutated in place).
    force:
        If *True*, reclassify all targeted items regardless of existing
        classification or state.
    spine_filter:
        If set, classify only this spine index.
    on_progress:
        Optional callback invoked with the padded spine ID after each
        item is processed.

    Returns
    -------
    Manifest
        The updated manifest with classifications set.
    """
    # Load manifest.
    mp = manifest_path(work_dir)
    manifest = Manifest(**json.loads(mp.read_text(encoding="utf-8")))
    sw = manifest.spine_padding_width

    # Determine which items to process.
    if spine_filter is not None:
        items = [item for item in manifest.spine if item.spine_index == spine_filter]
        if not items:
            raise ValueError(f"Spine index {spine_filter} not found in manifest")
    else:
        items = list(manifest.spine)

    # Handle --force.
    if force:
        reset_stage(work_dir, state, "classify")
        for item in items:
            item.classification = None
            item.title = None

    # Skip if already completed.
    if not force and is_stage_completed(state, "classify") and spine_filter is None:
        logger.info("Classify stage already completed — skipping (use --force to re-run)")
        return manifest

    mark_stage_started(work_dir, state, "classify")

    # Build pending item list.
    item_ids = [pad_spine(item.spine_index, sw) for item in items]
    pending = set(iter_pending_items(state, "classify", item_ids))

    # Create LLM client lazily.
    _llm_client: LLMClient | None = None

    def _get_llm_client() -> LLMClient:
        nonlocal _llm_client
        if _llm_client is None:
            _llm_client = LLMClient(config.models.classify, config.llm)
        return _llm_client

    total_items = len(items)
    classified_count = 0
    skipped_count = 0
    structural_count = 0
    llm_count = 0
    low_confidence_items: list[tuple[str, str, str]] = []  # (padded_id, classification, reasoning)

    for item in items:
        padded = pad_spine(item.spine_index, sw)

        # Skip if already completed in state (unless force).
        if not force and padded not in pending:
            if on_progress:
                on_progress(padded)
            continue

        # Manual override: preserve existing classification.
        if item.classification is not None and not force:
            logger.info(
                "Spine %s: preserving existing classification '%s' (manual override)",
                padded,
                item.classification,
            )
            mark_item_started(work_dir, state, "classify", padded)
            mark_item_completed(work_dir, state, "classify", padded)
            skipped_count += 1
            if on_progress:
                on_progress(padded)
            continue

        mark_item_started(work_dir, state, "classify", padded)

        # Read raw XHTML.
        raw_file = work_dir / item.raw_path
        if not raw_file.exists():
            logger.error("Spine %s: raw file missing at %s", padded, raw_file)
            mark_item_failed(work_dir, state, "classify", padded, "raw file missing")
            item.classification = "unknown"
            if on_progress:
                on_progress(padded)
            continue

        raw_xhtml = raw_file.read_text(encoding="utf-8")

        # Read clean markdown (may not exist yet).
        clean_markdown = ""
        if item.clean_path:
            clean_file = work_dir / item.clean_path
            if clean_file.exists():
                clean_markdown = clean_file.read_text(encoding="utf-8")

        # Classify.
        position = (item.spine_index, total_items)
        try:
            result = classify_item(
                item, raw_xhtml, clean_markdown, position, config, _get_llm_client()
            )
            item.classification = result.classification  # type: ignore[assignment]
            item.title = result.title
            classified_count += 1

            if result.source == "structural":
                structural_count += 1
            else:
                llm_count += 1

            if result.confidence == "low":
                low_confidence_items.append((padded, result.classification, result.reasoning))

            logger.info(
                "Spine %s: classified as '%s' (source=%s, confidence=%s)",
                padded,
                result.classification,
                result.source,
                result.confidence,
            )
            mark_item_completed(work_dir, state, "classify", padded)

        except LLMStructuredOutputError as exc:
            logger.warning(
                "Spine %s: LLM classification failed after retries — marking as 'unknown'. "
                "Error: %s",
                padded,
                exc,
            )
            item.classification = "unknown"
            item.title = None
            mark_item_failed(work_dir, state, "classify", padded, str(exc))

        except Exception as exc:
            logger.warning(
                "Spine %s: unexpected error during classification — marking as 'unknown'. "
                "Error: %s",
                padded,
                exc,
            )
            item.classification = "unknown"
            item.title = None
            mark_item_failed(work_dir, state, "classify", padded, str(exc))

        if on_progress:
            on_progress(padded)

    # Persist manifest atomically.
    atomic_write(mp, manifest.model_dump_json(indent=2))

    # Mark stage completed if all items are done (no spine filter).
    if spine_filter is None:
        all_ids = [pad_spine(item.spine_index, sw) for item in manifest.spine]
        remaining = iter_pending_items(state, "classify", all_ids)
        if not remaining:
            mark_stage_completed(work_dir, state, "classify")

    # Log summary.
    logger.info(
        "Classification complete: %d classified (%d structural, %d LLM), %d skipped",
        classified_count,
        structural_count,
        llm_count,
        skipped_count,
    )

    if low_confidence_items:
        logger.warning("Items with LOW confidence (review recommended):")
        for padded_id, cls, reasoning in low_confidence_items:
            logger.warning("  Spine %s → %s: %s", padded_id, cls, reasoning)

    return manifest
