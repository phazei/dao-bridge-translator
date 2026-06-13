"""Translation stage — per-chunk LLM translation with glossary context.

Translates chunked source text using a multi-pass LLM pipeline:

1. **Pass 1** — initial translation with glossary, overlap, and rolling
   summary context.
2. **Pass 2** (optional) — revision pass comparing draft against the
   original source.
3. **QA assessment** (optional) — programmatic sanity checks followed by
   an LLM judge that evaluates accuracy, completeness, and glossary
   compliance.
4. **Rolling summary** (optional) — generates a short narrative summary
   appended to a sliding-window context file.

Each pass uses a fresh conversation with only the context it needs,
keeping token usage lean.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from dao_bridge.chunk import count_tokens
from dao_bridge.config import AppConfig, resolve_language_name
from dao_bridge.glossary import load_glossary
from dao_bridge.llm_client import LLMClient, LLMStructuredOutputError
from dao_bridge.schemas import (
    Chunk,
    Glossary,
    GlossaryEntity,
    Manifest,
    ManifestItem,
    TranslatedChunk,
)
from dao_bridge.state import (
    PipelineState,
    iter_pending_items,
    mark_item_completed,
    mark_item_failed,
    mark_item_started,
    mark_stage_completed,
    mark_stage_started,
    reopen_stage,
    reset_stage,
    reset_stage_items,
)
from dao_bridge.workdir import (
    atomic_write,
    chunk_path,
    failed_translation_path,
    format_chunk_id,
    glossary_path,
    next_failed_attempt,
    parse_chunk_id,
    summary_path,
    translation_dir,
    translation_path,
)

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parent / "prompts"
_STAGE: Literal["translate"] = "translate"

# Warn if rendered glossary exceeds this many tokens (in "all" mode).
_GLOSSARY_TOKEN_WARNING_THRESHOLD = 5000

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    """Result of a QA assessment (programmatic or LLM)."""

    result: Literal["pass", "fail"]
    issues: list[str] = field(default_factory=list)
    source: Literal["programmatic", "llm"] = "llm"


class QAIssue(BaseModel):
    """A single QA issue with a severity rating.

    ``severity`` gates whether the issue forces a failure:

    - ``"high"`` — a real defect that breaks the translation (dropped or
      skipped content, refusal, repetition loop, reversed/garbled meaning,
      or untranslated author/publisher notes leaking into the output).
    - ``"low"`` — a stylistic or nuance observation that should NOT fail the
      chunk (word choice, phrasing, minor imagery differences).
    """

    severity: Literal["high", "low"] = "low"
    issue: str


class QAResponse(BaseModel):
    """Pydantic model for the LLM QA assessment JSON response."""

    result: Literal["pass", "fail"]
    issues: list[QAIssue] = []


@dataclass
class TranslationProgress:
    """Progress information passed to the CLI callback."""

    chunk_id: str
    pass_name: str  # "Pass 1", "Pass 2", "QA", "Summary"
    tokens_so_far: int
    chunks_completed: int
    chunks_total: int


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the ``prompts/`` directory.

    Cached — template files are read once per process.

    Parameters
    ----------
    name:
        Template filename (e.g. ``"translate_pass1.txt"``).
    """
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Analysis stripping
# ---------------------------------------------------------------------------

_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>\s*", re.DOTALL)


def _extract_analysis(text: str) -> str | None:
    """Extract the ``<analysis>...</analysis>`` block from Pass 1 LLM output.

    Returns the full tagged block (including ``<analysis>`` tags), or
    ``None`` if no analysis block is found.
    """
    match = _ANALYSIS_RE.search(text)
    if match:
        return match.group(0).strip()
    return None


def _strip_analysis(text: str) -> str:
    """Remove the ``<analysis>...</analysis>`` block from Pass 1 LLM output."""
    return _ANALYSIS_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Glossary rendering
# ---------------------------------------------------------------------------


def find_relevant_entities(chunk_text: str, glossary: Glossary) -> list[GlossaryEntity]:
    """Return entities that have at least one surface form present in *chunk_text*.

    Each entity appears at most once, even if multiple surface forms match.

    Parameters
    ----------
    chunk_text:
        Source text of the current chunk.
    glossary:
        The per-book glossary.

    Returns
    -------
    list[GlossaryEntity]
        Entities with at least one matching surface form.
    """
    relevant: list[GlossaryEntity] = []
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.source and sf.source in chunk_text:
                relevant.append(entity)
                break  # One match is enough; don't duplicate
    return relevant


def render_glossary(glossary: Glossary, chunk_text: str, mode: str) -> str:
    """Render glossary entities for prompt injection.

    Parameters
    ----------
    glossary:
        The per-book glossary.
    chunk_text:
        Source text of the current chunk (used for "relevant" filtering).
    mode:
        ``"relevant"`` — include only entities with a surface form that
        appears in *chunk_text*.  ``"all"`` — include every entity.

    Returns
    -------
    str
        Formatted glossary block ready for insertion into a system prompt.
    """
    if mode == "relevant":
        entities = find_relevant_entities(chunk_text, glossary)
    else:
        entities = list(glossary.entities)

    if not entities:
        return ""

    # Group by category.
    groups: dict[str, list[GlossaryEntity]] = {}
    for entity in entities:
        cat = entity.category.capitalize()
        groups.setdefault(cat, []).append(entity)

    lines = ["\nGLOSSARY (terms in this section)"]

    for category, group_entities in groups.items():
        lines.append("")
        lines.append(f"{category}:")
        for entity in group_entities:
            # Header: canonical name.
            header = f"- [{entity.canonical_name}]"
            if entity.summary:
                header += f" — {entity.summary}"
            elif entity.notes:
                header += f" — {entity.notes}"
            lines.append(header)

            # Surface forms with per-form translations.
            for sf in entity.surface_forms:
                lines.append(f"  {sf.source} -> {sf.translation}")

            # Character-specific fields.
            if category.lower() == "character":
                if entity.speech_style:
                    lines.append(f"  Speech: {entity.speech_style}")
                if entity.nicknames:
                    nick_parts = [
                        f'{speaker} calls them "{nick}"'
                        for speaker, nick in entity.nicknames.items()
                    ]
                    lines.append(f"  Nicknames: {'; '.join(nick_parts)}.")

    rendered = "\n".join(lines)

    # Warn if glossary is very large.
    if mode == "all":
        token_count = count_tokens(rendered)
        if token_count > _GLOSSARY_TOKEN_WARNING_THRESHOLD:
            logger.warning(
                "Glossary exceeds %d tokens (%d tokens). Consider using "
                "glossary_injection: relevant.",
                _GLOSSARY_TOKEN_WARNING_THRESHOLD,
                token_count,
            )

    return rendered


# ---------------------------------------------------------------------------
# Rolling summary rendering
# ---------------------------------------------------------------------------


def render_rolling_summary(summaries: list[dict], max_tokens: int) -> str:
    """Render the sliding window of rolling summaries for prompt injection.

    Iterates from newest to oldest, accumulating entries until the token
    budget is exhausted.  The result is returned in chronological order.

    Parameters
    ----------
    summaries:
        List of ``{"chunk_id": str, "summary": str}`` dicts.
    max_tokens:
        Maximum total tokens to include.

    Returns
    -------
    str
        Formatted summary block, or empty string if no summaries.
    """
    if not summaries:
        return ""

    # Walk backwards from newest, accumulating until budget exhausted.
    selected: list[dict] = []
    running_tokens = 0

    for entry in reversed(summaries):
        entry_text = f"[{entry['chunk_id']}] {entry['summary']}"
        entry_tokens = count_tokens(entry_text)
        if running_tokens + entry_tokens > max_tokens and selected:
            break
        selected.append(entry)
        running_tokens += entry_tokens

    # Restore chronological order.
    selected.reverse()

    lines = ["\nSTORY SO FAR (rolling summary)", ""]
    for entry in selected:
        lines.append(f"[{entry['chunk_id']}] {entry['summary']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overlap loading
# ---------------------------------------------------------------------------


def load_overlap(
    chunk: Chunk,
    manifest: Manifest,
    config: AppConfig,
) -> TranslatedChunk | None:
    """Find and load the overlap chunk for continuity context.

    Parameters
    ----------
    chunk:
        The chunk about to be translated.
    manifest:
        Book manifest (for cross-spine lookups).
    config:
        Application config (for overlap settings and work_dir).

    Returns
    -------
    TranslatedChunk | None
        The previous chunk's translation, or ``None`` if no overlap applies.

    Raises
    ------
    RuntimeError
        If the required overlap chunk has not yet been translated.
    """
    tp = config.translation_phase
    work_dir = config.work_dir_path
    sw = manifest.spine_padding_width

    if tp.overlap_chunks == 0:
        return None

    spine_index, chunk_index = parse_chunk_id(chunk.chunk_id)

    if chunk_index > 1:
        # Same-spine overlap: previous chunk in the same spine.
        overlap_id = format_chunk_id(spine_index, chunk_index - 1, sw)
        return _load_translated_chunk(work_dir, overlap_id, sw)

    # chunk_index == 1: first chunk of this spine.
    # Find the previous spine with chunks.
    if not tp.cross_spine_overlap:
        return None

    prev_spine = _find_previous_chunked_spine(spine_index, manifest)
    if prev_spine is None:
        # This is the very first chunk of the book.
        return None

    # Load the last chunk of the previous spine.
    overlap_id = format_chunk_id(
        prev_spine.spine_index,
        prev_spine.chunk_count,
        sw,  # type: ignore[arg-type]
    )
    return _load_translated_chunk(work_dir, overlap_id, sw)


def _find_previous_chunked_spine(
    current_spine_index: int, manifest: Manifest
) -> ManifestItem | None:
    """Find the nearest previous spine item that has chunks."""
    prev: ManifestItem | None = None
    for item in manifest.spine:
        if item.spine_index >= current_spine_index:
            break
        if item.chunk_count and item.chunk_count > 0:
            prev = item
    return prev


def _load_translated_chunk(work_dir: Path, chunk_id: str, spine_width: int) -> TranslatedChunk:
    """Load a translated chunk from disk.

    Raises
    ------
    RuntimeError
        If the translation file does not exist.
    """
    path = translation_path(work_dir, chunk_id, spine_width)
    if not path.exists():
        raise RuntimeError(
            f"Chunk {chunk_id} depends on overlap chunk which has not been "
            f"translated. Expected: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return TranslatedChunk(**data)


# ---------------------------------------------------------------------------
# Programmatic QA check
# ---------------------------------------------------------------------------


def programmatic_qa_check(
    source_text: str,
    translated_text: str,
    config: AppConfig,
) -> QAResult | None:
    """Run programmatic sanity checks on the translation.

    Returns a :class:`QAResult` on failure, or ``None`` if checks pass
    (proceed to LLM judge).
    """
    tp = config.translation_phase
    source_tokens = count_tokens(source_text)
    if source_tokens == 0:
        return None

    translation_tokens = count_tokens(translated_text)
    ratio = translation_tokens / source_tokens

    if ratio < tp.min_length_ratio:
        return QAResult(
            result="fail",
            issues=[
                "output suspiciously short — possible refusal or truncation "
                f"(ratio {ratio:.2f} < {tp.min_length_ratio})"
            ],
            source="programmatic",
        )

    if ratio > tp.max_length_ratio:
        return QAResult(
            result="fail",
            issues=[
                "output suspiciously long — possible repetition loop "
                f"(ratio {ratio:.2f} > {tp.max_length_ratio})"
            ],
            source="programmatic",
        )

    return None


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_pass1_messages(
    chunk: Chunk,
    glossary: Glossary,
    overlap: TranslatedChunk | None,
    rolling_summaries: list[dict],
    config: AppConfig,
) -> list[dict]:
    """Construct the message list for Pass 1 (initial translation).

    Parameters
    ----------
    chunk:
        The chunk to translate.
    glossary:
        Per-book glossary.
    overlap:
        Previous chunk's translation for continuity, or ``None``.
    rolling_summaries:
        List of ``{chunk_id, summary}`` dicts.
    config:
        Application config.

    Returns
    -------
    list[dict]
        OpenAI chat-format message list.
    """
    tp = config.translation_phase

    # Render glossary.
    glossary_text = render_glossary(glossary, chunk.text, tp.glossary_injection)

    # Render rolling summary.
    if tp.rolling_summary and rolling_summaries:
        summary_text = render_rolling_summary(rolling_summaries, tp.summary_max_tokens)
    else:
        summary_text = ""

    # Load and format system prompt (stable — no dynamic content).
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)
    template = _load_prompt_template("translate_pass1.txt")
    system_content = template.format(
        source_language=source_lang,
        target_language=target_lang,
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Glossary context (skip if no matching entries).
    if glossary_text:
        messages.append({"role": "user", "content": glossary_text})
        messages.append({"role": "assistant", "content": "Understood."})

    # Rolling summary context (skip if no summaries).
    if summary_text:
        messages.append({"role": "user", "content": summary_text})
        messages.append({"role": "assistant", "content": "Understood."})

    # Overlap context (skip if no previous chunk).
    if overlap is not None:
        overlap_content = (
            f"Here is the preceding section and its {target_lang} translation, "
            "for style and voice continuity. Do not retranslate this.\n\n"
            f"{source_lang.upper()}:\n{overlap.source_text}\n\n"
            f"{target_lang.upper()}:\n{overlap.translated_text}"
        )
        messages.append({"role": "user", "content": overlap_content})
        messages.append({"role": "assistant", "content": "Understood."})

    # Source text to translate.
    source_content = f"Translate the following {source_lang} text to {target_lang}:\n\n{chunk.text}"
    messages.append({"role": "user", "content": source_content})

    return messages


def build_pass2_messages(
    source_text: str,
    pass1_response: str,
    config: AppConfig,
) -> list[dict]:
    """Build a fresh message list for Pass 2 (polish).

    Contains only the source text and the Pass 1 draft — no overlap,
    glossary, or rolling summary.  These are unnecessary for polishing
    and would bloat the context.

    Parameters
    ----------
    source_text:
        Original source text for the chunk.
    pass1_response:
        The Pass 1 translation output.
    config:
        Application config.

    Returns
    -------
    list[dict]
        OpenAI chat-format message list (system + user).
    """
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)
    system_template = _load_prompt_template("translate_pass2.txt")
    system_content = system_template.format(
        source_language=source_lang,
        target_language=target_lang,
    )

    user_template = _load_prompt_template("translate_pass2_user.txt")
    user_content = user_template.format(
        source_language=source_lang,
        target_language=target_lang,
        source_text=source_text,
        draft_translation=pass1_response,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_qa_messages(
    source_text: str,
    final_translation: str,
    glossary: Glossary,
    config: AppConfig,
) -> list[dict]:
    """Build a fresh message list for QA assessment.

    Contains the source text, the final translation, and the glossary terms
    relevant to this chunk.  The glossary lets the judge recognise canonical
    names (e.g. a deliberately chosen nickname rendering) so it does not
    mistake them for hallucinated proper nouns.  No overlap or rolling summary
    is included — the judge only compares source against output.

    Parameters
    ----------
    source_text:
        Original source text for the chunk.
    final_translation:
        The final translation to assess (Pass 2 or Pass 1).
    glossary:
        Per-book glossary; relevant entries are injected as reference.
    config:
        Application config.

    Returns
    -------
    list[dict]
        OpenAI chat-format message list (system + optional glossary + user).
    """
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    system_template = _load_prompt_template("translate_qa.txt")
    system_content = system_template.format(
        source_language=source_lang,
        target_language=target_lang,
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject the glossary terms relevant to this chunk (filtered by source
    # text).  These names are authoritative — never flag them as errors.
    glossary_text = render_glossary(glossary, source_text, "relevant")
    if glossary_text:
        glossary_content = (
            "The following glossary terms are the APPROVED translations for "
            "this book. Treat every name and rendering below as correct by "
            "definition. Never flag a translation that matches this glossary "
            "as a hallucination, misspelling, or wrong name.\n"
            f"{glossary_text}"
        )
        messages.append({"role": "user", "content": glossary_content})
        messages.append({"role": "assistant", "content": "Understood."})

    user_template = _load_prompt_template("translate_qa_user.txt")
    user_content = user_template.format(
        source_language=source_lang,
        target_language=target_lang,
        source_text=source_text,
        translated_text=final_translation,
    )
    messages.append({"role": "user", "content": user_content})

    return messages


def build_qa_fix_messages(
    source_text: str,
    current_translation: str,
    issues: list[str],
    glossary: Glossary,
    config: AppConfig,
) -> list[dict]:
    """Build the message list for a targeted QA-fix pass.

    Unlike Pass 1 (which translates from scratch) or Pass 2 (which polishes
    on the assumption the text is already correct), this pass is given the
    existing translation plus the specific high-severity defects the QA judge
    found.  It operates in one of two modes depending on the defect:

    - **Surgical** (boilerplate leak, repetition): minimal edits to the prose.
    - **Regenerate** (refusal, missing content): translate from source, since
      there is no good prose to preserve.

    The glossary is injected (with a "these names are approved" guard) so that
    when the pass regenerates content it uses the correct canonical names and
    never "corrects" a name that is already right.  Rolling summary and overlap
    are deliberately omitted — continuity is irrelevant to a corrective edit.

    Parameters
    ----------
    source_text:
        Original source text for the chunk.
    current_translation:
        The translation that failed QA (the text the judge assessed).
    issues:
        The high-severity issue descriptions to fix.
    glossary:
        Per-book glossary; relevant entries are injected as approved names.
    config:
        Application config.

    Returns
    -------
    list[dict]
        OpenAI chat-format message list (system + optional glossary + user).
    """
    source_lang = resolve_language_name(config.languages.source)
    target_lang = resolve_language_name(config.languages.target)

    system_template = _load_prompt_template("translate_qa_fix.txt")
    system_content = system_template.format(
        source_language=source_lang,
        target_language=target_lang,
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject the glossary terms relevant to this chunk so regenerated content
    # uses the approved names and existing correct names are left untouched.
    glossary_text = render_glossary(glossary, source_text, "relevant")
    if glossary_text:
        glossary_content = (
            "The following glossary terms are the APPROVED translations for "
            "this book. Use these exact renderings for any name or term you "
            "translate or correct, and never change a name that already "
            "matches this glossary.\n"
            f"{glossary_text}"
        )
        messages.append({"role": "user", "content": glossary_content})
        messages.append({"role": "assistant", "content": "Understood."})

    issues_block = "\n".join(f"- {issue}" for issue in issues)
    user_template = _load_prompt_template("translate_qa_fix_user.txt")
    user_content = user_template.format(
        source_language=source_lang,
        target_language=target_lang,
        source_text=source_text,
        current_translation=current_translation,
        issues=issues_block,
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Rolling summary I/O
# ---------------------------------------------------------------------------


def _load_rolling_summaries(work_dir: Path) -> list[dict]:
    """Load the rolling summary file, returning ``[]`` if absent."""
    path = summary_path(work_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_rolling_summaries(work_dir: Path, summaries: list[dict]) -> None:
    """Atomically write the rolling summary file."""
    path = summary_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(summaries, indent=2, ensure_ascii=False))


def _update_rolling_summary(summaries: list[dict], chunk_id: str, summary: str) -> list[dict]:
    """Add or overwrite a summary entry for *chunk_id*."""
    # Overwrite if exists.
    for i, entry in enumerate(summaries):
        if entry["chunk_id"] == chunk_id:
            summaries[i] = {"chunk_id": chunk_id, "summary": summary}
            return summaries
    # Append.
    summaries.append({"chunk_id": chunk_id, "summary": summary})
    return summaries


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(
    translated_text: str,
    chunk_id: str,
    rolling_summaries: list[dict],
    llm_client: LLMClient,
    config: AppConfig,
) -> str:
    """Generate a rolling summary for one translated chunk.

    Parameters
    ----------
    translated_text:
        The final translated text for the chunk.
    chunk_id:
        The chunk ID (e.g. ``"0003.015"``).
    rolling_summaries:
        Existing summaries (for prior-context injection).
    llm_client:
        LLM client configured for the summarize model.
    config:
        Application config.

    Returns
    -------
    str
        The generated summary text.
    """
    tp = config.translation_phase

    # Render prior summaries with the same sliding window.
    if rolling_summaries:
        prior_text = render_rolling_summary(rolling_summaries, tp.summary_max_tokens)
    else:
        prior_text = ""

    template = _load_prompt_template("translate_summary.txt")
    system_content = template.format(prior_summaries=prior_text)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": translated_text},
    ]

    result = llm_client.complete(messages)
    return result.text.strip()


# ---------------------------------------------------------------------------
# Per-chunk translation
# ---------------------------------------------------------------------------


def _merge_token_usage(base: dict, addition: dict) -> dict:
    """Accumulate token usage dictionaries."""
    merged = dict(base)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = merged.get(key, 0) + addition.get(key, 0)
    return merged


def _run_qa(
    chunk_id: str,
    source_text: str,
    final_text: str,
    glossary: Glossary,
    llm_client: LLMClient,
    config: AppConfig,
) -> tuple[Literal["pass", "fail"] | None, list[str]]:
    """Assess a translation and return ``(qa_result, high_severity_issues)``.

    Programmatic checks run first.  If they pass, the LLM judge is consulted.
    Only HIGH-severity issues force a ``"fail"``; low-severity observations are
    logged but pass.  Returns the list of high-severity issue strings (empty on
    pass).
    """
    tp = config.translation_phase

    prog_result = programmatic_qa_check(source_text, final_text, config)
    if prog_result is not None:
        return prog_result.result, list(prog_result.issues)

    messages = build_qa_messages(source_text, final_text, glossary, config)
    try:
        qa_resp: QAResponse = llm_client.complete_json(  # type: ignore[assignment]
            messages,
            QAResponse,
            max_retries=3,
            temperature=tp.qa_temperature,
            context_label=f"{chunk_id}:qa",
        )
    except LLMStructuredOutputError:
        return "fail", ["QA assessment returned unparseable JSON after retries"]

    high_issues = [i for i in qa_resp.issues if i.severity == "high"]
    low_issues = [i for i in qa_resp.issues if i.severity != "high"]
    if low_issues:
        logger.info(
            "Chunk %s QA: %d high / %d low-severity issue(s). Low notes: %s",
            chunk_id,
            len(high_issues),
            len(low_issues),
            "; ".join(i.issue for i in low_issues),
        )
    if high_issues:
        return "fail", [i.issue for i in high_issues]
    return "pass", []


def translate_chunk(
    chunk: Chunk,
    config: AppConfig,
    glossary: Glossary,
    overlap: TranslatedChunk | None,
    rolling_summaries: list[dict],
    llm_client: LLMClient,
    summary_client: LLMClient | None = None,
) -> TranslatedChunk:
    """Translate a single chunk through the full pipeline.

    Performs one translation attempt (Pass 1, optionally Pass 2, optionally
    QA, optionally summary).  Does **not** handle retries — the caller
    (:func:`run_translate_stage`) manages retry logic, running a targeted
    QA-fix pass (:func:`qa_fix_chunk`) when QA reports high-severity issues.

    Parameters
    ----------
    chunk:
        The source chunk to translate.
    config:
        Application config.
    glossary:
        Per-book glossary.
    overlap:
        Previous chunk's translation for continuity, or ``None``.
    rolling_summaries:
        List of ``{chunk_id, summary}`` dicts.
    llm_client:
        LLM client for translation (and QA).
    summary_client:
        LLM client for summary generation.  Falls back to *llm_client*
        if ``None``.

    Returns
    -------
    TranslatedChunk
        The translation result.  Inspect ``qa_result`` to determine
        whether the chunk passed QA.
    """
    tp = config.translation_phase
    start_time = time.monotonic()
    model_used = ""

    # Reset cumulative counters so we capture exactly this chunk's usage.
    llm_client.reset_token_usage()
    s_client = summary_client or llm_client
    if s_client is not llm_client:
        s_client.reset_token_usage()

    # --- Pass 1 ---
    messages = build_pass1_messages(chunk, glossary, overlap, rolling_summaries, config)
    result1 = llm_client.complete(messages)
    pass1_raw = result1.text
    pass1_analysis = _extract_analysis(pass1_raw)
    pass1_text = _strip_analysis(pass1_raw)
    model_used = result1.model or llm_client.config.model

    # --- Pass 2 (revision) ---
    pass_count = 1
    final_text = pass1_text

    if tp.double_pass:
        messages = build_pass2_messages(chunk.text, pass1_text, config)
        result2 = llm_client.complete(messages)
        final_text = result2.text.strip()
        pass_count = 2

    # --- QA assessment ---
    qa_result_val: Literal["pass", "fail"] | None = None
    qa_issues: list[str] = []

    if tp.qa_check:
        qa_result_val, qa_issues = _run_qa(
            chunk.chunk_id, chunk.text, final_text, glossary, llm_client, config
        )

    # --- Rolling summary ---
    summary_text: str | None = None
    if tp.rolling_summary and qa_result_val != "fail":
        summary_text = generate_summary(
            final_text, chunk.chunk_id, rolling_summaries, s_client, config
        )

    # Collect token usage from all clients used for this chunk.
    token_usage = llm_client.total_token_usage
    if s_client is not llm_client:
        token_usage = _merge_token_usage(token_usage, s_client.total_token_usage)

    duration = time.monotonic() - start_time

    return TranslatedChunk(
        chunk_id=chunk.chunk_id,
        source_text=chunk.text,
        pass1_translation=pass1_text,
        pass1_analysis=pass1_analysis,
        translated_text=final_text,
        pass_count=pass_count,
        qa_result=qa_result_val,
        qa_issues=qa_issues,
        total_attempts=1,  # caller increments across retries
        overlap_chunk_id=overlap.chunk_id if overlap else None,
        summary_generated=summary_text,
        token_usage=token_usage,
        model_used=model_used,
        duration_seconds=round(duration, 2),
    )


def qa_fix_chunk(
    chunk: Chunk,
    prior: TranslatedChunk,
    issues: list[str],
    config: AppConfig,
    glossary: Glossary,
    overlap: TranslatedChunk | None,
    rolling_summaries: list[dict],
    llm_client: LLMClient,
    summary_client: LLMClient | None = None,
) -> TranslatedChunk:
    """Run a targeted QA-fix pass on a translation that failed QA.

    Unlike :func:`translate_chunk` (which retranslates from scratch), this
    takes the *existing* translation plus the specific high-severity issues and
    asks the model to surgically correct only those defects.  The corrected
    text is then re-assessed by QA.  Returns a fresh, internally-consistent
    :class:`TranslatedChunk` whose every field derives from this fix attempt.
    """
    tp = config.translation_phase
    start_time = time.monotonic()

    llm_client.reset_token_usage()
    s_client = summary_client or llm_client
    if s_client is not llm_client:
        s_client.reset_token_usage()

    # --- QA-fix pass ---
    messages = build_qa_fix_messages(
        chunk.text, prior.translated_text, issues, glossary, config
    )
    result = llm_client.complete(messages)
    fixed_text = result.text.strip()
    model_used = result.model or llm_client.config.model

    # --- Re-assess ---
    qa_result_val: Literal["pass", "fail"] | None = None
    qa_issues: list[str] = []
    if tp.qa_check:
        qa_result_val, qa_issues = _run_qa(
            chunk.chunk_id, chunk.text, fixed_text, glossary, llm_client, config
        )

    # --- Rolling summary (only when the fixed text passes) ---
    summary_text: str | None = None
    if tp.rolling_summary and qa_result_val != "fail":
        summary_text = generate_summary(
            fixed_text, chunk.chunk_id, rolling_summaries, s_client, config
        )

    token_usage = llm_client.total_token_usage
    if s_client is not llm_client:
        token_usage = _merge_token_usage(token_usage, s_client.total_token_usage)

    duration = time.monotonic() - start_time

    return TranslatedChunk(
        chunk_id=chunk.chunk_id,
        source_text=chunk.text,
        # Carry the prior attempt's Pass 1 text so the record stays meaningful;
        # the fix operates on the final text, not Pass 1.
        pass1_translation=prior.pass1_translation,
        pass1_analysis=prior.pass1_analysis,
        translated_text=fixed_text,
        pass_count=prior.pass_count,
        qa_result=qa_result_val,
        qa_issues=qa_issues,
        total_attempts=1,  # caller sets the real attempt number
        overlap_chunk_id=overlap.chunk_id if overlap else None,
        summary_generated=summary_text,
        token_usage=token_usage,
        model_used=model_used,
        duration_seconds=round(duration, 2),
    )


# ---------------------------------------------------------------------------
# Chunk enumeration helpers
# ---------------------------------------------------------------------------


def _enumerate_chunk_ids(manifest: Manifest) -> list[str]:
    """Return all chunk IDs in spine order, then chunk order."""
    sw = manifest.spine_padding_width
    ids: list[str] = []
    for item in manifest.spine:
        if not item.chunk_count or item.chunk_count == 0:
            continue
        for ci in range(1, item.chunk_count + 1):
            ids.append(format_chunk_id(item.spine_index, ci, sw))
    return ids


def _filter_chunk_range(
    chunk_ids: list[str],
    from_chunk: str | None,
    to_chunk: str | None,
) -> list[str]:
    """Filter chunk IDs to the specified range (inclusive, string comparison)."""
    if from_chunk is None and to_chunk is None:
        return chunk_ids
    return [
        cid
        for cid in chunk_ids
        if (from_chunk is None or cid >= from_chunk) and (to_chunk is None or cid <= to_chunk)
    ]


def _spine_range_for_filter(spine_index: int, manifest: Manifest) -> tuple[str, str]:
    """Return (from_chunk, to_chunk) covering all chunks in a spine."""
    sw = manifest.spine_padding_width
    item = next((i for i in manifest.spine if i.spine_index == spine_index), None)
    if item is None or not item.chunk_count:
        raise ValueError(f"Spine {spine_index} not found or has no chunks.")
    from_id = format_chunk_id(spine_index, 1, sw)
    to_id = format_chunk_id(spine_index, item.chunk_count, sw)
    return from_id, to_id


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def _load_chunk(work_dir: Path, chunk_id: str, spine_width: int) -> Chunk:
    """Load a Chunk JSON from disk."""
    path = chunk_path(work_dir, chunk_id, spine_width)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Chunk(**data)


def _is_previous_chunk_completed(
    chunk_id: str,
    all_chunk_ids: list[str],
    state: PipelineState,
) -> bool:
    """Check whether the chunk immediately before *chunk_id* is completed.

    Returns ``True`` if there is no previous chunk (first chunk of book)
    or if the previous chunk is in ``completed`` state.
    """
    idx = all_chunk_ids.index(chunk_id)
    if idx == 0:
        return True
    prev_id = all_chunk_ids[idx - 1]
    key = f"{_STAGE}:{prev_id}"
    item = state.items.get(key)
    return item is not None and item.status == "completed"


def run_translate_stage(
    work_dir: Path,
    config: AppConfig,
    state: PipelineState,
    manifest: Manifest,
    force: bool = False,
    retry_failed: bool = False,
    from_chunk: str | None = None,
    to_chunk: str | None = None,
    on_progress: Callable[[TranslationProgress], None] | None = None,
) -> dict:
    """Run the translate stage across all eligible chunks.

    Parameters
    ----------
    work_dir:
        Working directory.
    config:
        Application config.
    state:
        Pipeline state (mutated in place).
    manifest:
        Book manifest.
    force:
        If ``True``, retranslate even completed chunks.
    retry_failed:
        If ``True``, re-enter a completed stage to retry only failed
        chunks.  Preserves completed chunk state (unlike ``force``).
    from_chunk:
        Start translating from this chunk ID (inclusive).
    to_chunk:
        Stop translating after this chunk ID (inclusive).
    on_progress:
        Optional callback ``(TranslationProgress) -> None`` for UI updates.

    Returns
    -------
    dict
        Summary with keys: ``completed``, ``total_tokens``,
        ``failed_chunk``, ``error``, ``avg_time``, ``total_chunks``.
    """
    tp = config.translation_phase
    sw = manifest.spine_padding_width

    # Load glossary (validates categories against config).
    gp = glossary_path(work_dir)
    if not gp.exists():
        raise RuntimeError(
            f"Glossary not found at {gp}. "
            "Run 'dao-bridge glossary-build' and 'glossary-reconcile' first."
        )
    glossary = load_glossary(work_dir, config)

    # Create LLM clients.
    llm_config = config.llm
    translate_client = LLMClient(config.models.translate, llm_config)
    summarize_cfg = config.summarize_model()
    summary_client: LLMClient | None = None
    if summarize_cfg is not config.models.translate:
        summary_client = LLMClient(summarize_cfg, llm_config)

    # Enumerate all chunk IDs.
    all_chunk_ids = _enumerate_chunk_ids(manifest)
    target_ids = _filter_chunk_range(all_chunk_ids, from_chunk, to_chunk)

    # Targeted range (--chunk, --spine, --from/--to) implies force for
    # those chunks.
    targeted = from_chunk is not None or to_chunk is not None
    if targeted:
        reset_stage_items(work_dir, state, _STAGE, target_ids)
    elif force:
        reset_stage(work_dir, state, _STAGE)
    elif retry_failed:
        reopen_stage(work_dir, state, _STAGE)

    # Determine which chunks need work.
    target_ids = iter_pending_items(state, _STAGE, target_ids)

    mark_stage_started(work_dir, state, _STAGE)

    # Stats tracking.
    completed_count = 0
    total_tokens = 0
    total_duration = 0.0

    progress_cb = on_progress

    for chunk_id in target_ids:
        # Sequential enforcement when overlap is enabled.
        if tp.overlap_chunks > 0:
            if not _is_previous_chunk_completed(chunk_id, all_chunk_ids, state):
                prev_idx = all_chunk_ids.index(chunk_id)
                prev_id = all_chunk_ids[prev_idx - 1] if prev_idx > 0 else "N/A"
                error = f"Chunk {chunk_id} depends on {prev_id} which has not been translated."
                mark_item_failed(work_dir, state, _STAGE, chunk_id, error)
                return {
                    "completed": completed_count,
                    "total_tokens": total_tokens,
                    "failed_chunk": chunk_id,
                    "error": error,
                    "avg_time": (total_duration / completed_count if completed_count > 0 else 0),
                    "total_chunks": len(target_ids),
                }

        mark_item_started(work_dir, state, _STAGE, chunk_id)

        try:
            chunk = _load_chunk(work_dir, chunk_id, sw)
        except Exception as exc:
            error = f"Failed to load chunk: {exc}"
            mark_item_failed(work_dir, state, _STAGE, chunk_id, error)
            return {
                "completed": completed_count,
                "total_tokens": total_tokens,
                "failed_chunk": chunk_id,
                "error": error,
                "avg_time": (total_duration / completed_count if completed_count > 0 else 0),
                "total_chunks": len(target_ids),
            }

        # Load overlap.
        try:
            overlap = load_overlap(chunk, manifest, config)
        except RuntimeError as exc:
            error = str(exc)
            mark_item_failed(work_dir, state, _STAGE, chunk_id, error)
            return {
                "completed": completed_count,
                "total_tokens": total_tokens,
                "failed_chunk": chunk_id,
                "error": error,
                "avg_time": (total_duration / completed_count if completed_count > 0 else 0),
                "total_chunks": len(target_ids),
            }

        rolling_summaries = _load_rolling_summaries(work_dir)

        # Attempt 1 is a full translation; subsequent attempts are targeted
        # QA-fix passes that correct the high-severity issues in the best
        # translation so far.  max_attempts = 1 + qa_max_retries (QA on).
        max_attempts = 1 + (tp.qa_max_retries if tp.qa_check else 0)

        best_tc: TranslatedChunk | None = None  # fewest high-severity issues so far
        last_tc: TranslatedChunk | None = None  # most recent attempt (for QA-fix input)

        for attempt in range(1, max_attempts + 1):
            if progress_cb is not None:
                if attempt == 1:
                    pass_name = "Pass 1"
                else:
                    pass_name = f"QA-fix (attempt {attempt})"
                progress_cb(
                    TranslationProgress(
                        chunk_id=chunk_id,
                        pass_name=pass_name,
                        tokens_so_far=total_tokens,
                        chunks_completed=completed_count,
                        chunks_total=len(target_ids),
                    )
                )

            try:
                if attempt == 1:
                    tc = translate_chunk(
                        chunk=chunk,
                        config=config,
                        glossary=glossary,
                        overlap=overlap,
                        rolling_summaries=rolling_summaries,
                        llm_client=translate_client,
                        summary_client=summary_client,
                    )
                else:
                    assert last_tc is not None
                    tc = qa_fix_chunk(
                        chunk=chunk,
                        prior=last_tc,
                        issues=last_tc.qa_issues,
                        config=config,
                        glossary=glossary,
                        overlap=overlap,
                        rolling_summaries=rolling_summaries,
                        llm_client=translate_client,
                        summary_client=summary_client,
                    )
            except Exception as exc:
                # Infrastructure error.
                error = f"Translation failed: {exc}"
                logger.error("Chunk %s failed: %s", chunk_id, error)
                mark_item_failed(work_dir, state, _STAGE, chunk_id, error)
                return {
                    "completed": completed_count,
                    "total_tokens": total_tokens,
                    "failed_chunk": chunk_id,
                    "error": error,
                    "avg_time": (total_duration / completed_count if completed_count > 0 else 0),
                    "total_chunks": len(target_ids),
                }

            tc = TranslatedChunk(
                **{**tc.model_dump(), "total_attempts": attempt, "selected_attempt": attempt}
            )
            last_tc = tc

            # Keep-best: an attempt with zero high-severity issues is a pass —
            # adopt it and stop early.  Otherwise retain the attempt with the
            # fewest high-severity issues (ties resolved by the latest attempt).
            if tc.qa_result != "fail":
                best_tc = tc
                break

            if best_tc is None or len(tc.qa_issues) <= len(best_tc.qa_issues):
                best_tc = tc

            # Save the failed attempt as an audit artifact.
            fail_num = next_failed_attempt(work_dir, chunk_id, sw)
            fail_path = failed_translation_path(work_dir, chunk_id, fail_num, sw)
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(fail_path, tc.model_dump_json(indent=2))

            logger.warning(
                "Chunk %s QA failed (attempt %d/%d, high:%d): %s — saved %s",
                chunk_id,
                attempt,
                max_attempts,
                len(tc.qa_issues),
                "; ".join(tc.qa_issues),
                fail_path.name,
            )

        assert best_tc is not None
        final_tc = best_tc

        # Save translation.
        t_dir = translation_dir(work_dir, chunk.spine_index, sw)
        t_dir.mkdir(parents=True, exist_ok=True)
        t_path = translation_path(work_dir, chunk_id, sw)
        atomic_write(t_path, final_tc.model_dump_json(indent=2))

        # Update rolling summary.
        if final_tc.summary_generated:
            rolling_summaries = _update_rolling_summary(
                rolling_summaries, chunk_id, final_tc.summary_generated
            )
            _save_rolling_summaries(work_dir, rolling_summaries)

        # Handle final QA result.  QA is advisory and NON-halting: if no attempt
        # cleared all high-severity issues, we keep the best attempt (fewest
        # high-severity issues), log a warning, and continue the run.  The
        # per-attempt failed artifacts remain on disk for later review.
        if final_tc.qa_result == "fail":
            error_msg = "; ".join(final_tc.qa_issues)
            logger.warning(
                "Chunk %s unresolved after %d attempt(s) — keeping best (attempt %d, "
                "high:%d) and continuing. Issues: %s",
                chunk_id,
                last_tc.total_attempts if last_tc else 0,
                final_tc.selected_attempt,
                len(final_tc.qa_issues),
                error_msg,
            )

        # Accept the chunk (passed, or QA-failed-but-kept).
        mark_item_completed(work_dir, state, _STAGE, chunk_id)
        completed_count += 1
        chunk_tokens = final_tc.token_usage.get("total_tokens", 0)
        total_tokens += chunk_tokens
        total_duration += final_tc.duration_seconds

        logger.info(
            "Chunk %s translated (pass %d, %s tokens, %.1fs)",
            chunk_id,
            final_tc.pass_count,
            f"{chunk_tokens:,}",
            final_tc.duration_seconds,
        )

        # Log per-spine summary when all chunks for a spine item are done.
        spine_prefix = chunk_id.rsplit(".", 1)[0]
        spine_chunk_ids = [cid for cid in all_chunk_ids if cid.startswith(spine_prefix + ".")]
        if chunk_id == spine_chunk_ids[-1]:
            # Last chunk in this spine item.
            spine_item = next(
                (si for si in manifest.spine if f"{si.spine_index:0{sw}d}" == spine_prefix),
                None,
            )
            n_chunks = len(spine_chunk_ids)
            label = spine_item.classification if spine_item else "unknown"
            logger.info(
                "Spine %s: translation complete (%d chunk%s, %s)",
                spine_prefix,
                n_chunks,
                "s" if n_chunks != 1 else "",
                label,
            )

        if progress_cb is not None:
            progress_cb(
                TranslationProgress(
                    chunk_id=chunk_id,
                    pass_name="Done",
                    tokens_so_far=total_tokens,
                    chunks_completed=completed_count,
                    chunks_total=len(target_ids),
                )
            )

    # All chunks completed.
    # Only mark stage completed if we processed the full book (no range filter).
    if not targeted:
        remaining = iter_pending_items(state, _STAGE, all_chunk_ids)
        if not remaining:
            mark_stage_completed(work_dir, state, _STAGE)

    return {
        "completed": completed_count,
        "total_tokens": total_tokens,
        "failed_chunk": None,
        "error": None,
        "avg_time": (total_duration / completed_count if completed_count > 0 else 0),
        "total_chunks": len(target_ids),
    }
