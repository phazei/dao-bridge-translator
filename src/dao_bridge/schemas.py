"""Pydantic schemas for manifest, chunks, glossary, and translations.

These are the complete schemas for the entire pipeline.  Later prompts will
use them as-is.  Fields not populated by early stages are nullable and default
to ``None``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Classification literal
# ---------------------------------------------------------------------------

Classification = Literal[
    "chapter",
    "frontmatter",
    "backmatter",
    "toc_auto",
    "toc_authored",
    "illustration",
    "unknown",
]


# ---------------------------------------------------------------------------
# Classification response (LLM structured output)
# ---------------------------------------------------------------------------


class ClassificationResponse(BaseModel):
    """Pydantic model for LLM structured output via ``complete_json``."""

    classification: Classification
    title: str | None = None
    confidence: Literal["high", "medium", "low"]
    reasoning: str


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class ManifestItem(BaseModel):
    """A single spine item tracked through the pipeline."""

    spine_index: int
    original_href: str
    raw_path: str
    clean_path: str | None = None
    classification: Classification | None = None
    title: str | None = None
    token_count: int | None = None
    paragraph_count: int | None = None
    chunk_count: int | None = None  # set by chunker later


class Manifest(BaseModel):
    """Book-level manifest persisted to ``manifest.json``."""

    source_epub_path: str
    book_id: str
    opf_dir: str = ""  # OPF directory within the EPUB ZIP (e.g. "OEBPS")
    spine_padding_width: int = 4  # computed at extract time as max(4, len(str(spine_count)))
    spine: list[ManifestItem] = []
    images: list[str] = []
    metadata: dict = {}


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A single chunk of a spine item, produced by the chunker."""

    chunk_id: str  # "NNN.MMM"
    spine_index: int
    chunk_index: int  # per-spine, starting at 1
    source_file: str  # path to clean markdown file
    block_range: tuple[int, int]  # inclusive (start, end) of block indices
    token_count: int
    extended_for_remainder: bool = False
    text: str
    ends_at_scene_break: bool = False


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

GlossarySource = Literal["seed", "extracted", "user", "master"]


class GlossaryEntry(BaseModel):
    """A single glossary entry for term consistency."""

    source_term: str | None = None  # None for target-language-reference-only imports
    reading: str | None = None  # from furigana
    english: str
    category: str  # validated against config.glossary.categories at load time
    first_seen_chunk: str | None = None
    aliases: list[str] = []
    nicknames: dict[str, str] = {}  # {speaker_english_name: nickname_english}
    speech_style: str | None = None  # prose description, characters only
    notes: str | None = None
    source: GlossarySource
    source_books: list[str] = []  # populated for master-glossary entries


class Glossary(BaseModel):
    """Per-book or master glossary."""

    entries: list[GlossaryEntry] = []
    version: int = 1
    book_id: str | None = None  # None for master glossary
    book_metadata: dict = {}  # title, author, volume; per-book only
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Glossary LLM response models
# ---------------------------------------------------------------------------


class GlossaryExtractionEntry(BaseModel):
    """A single entry from the glossary extraction LLM response."""

    source_term: str
    reading: str | None = None
    english_proposed: str
    category: str
    aliases: list[str] = []
    nicknames: dict[str, str] = {}
    speech_style: str | None = None
    notes: str | None = None


class GlossaryCorrectionEntry(BaseModel):
    """A correction proposed by the LLM during glossary extraction."""

    existing_english: str
    source_term: str
    corrected_english: str
    reason: str


class GlossaryExtractionResponse(BaseModel):
    """Top-level LLM response for glossary extraction."""

    entries: list[GlossaryExtractionEntry] = []
    corrections: list[GlossaryCorrectionEntry] = []


class GlossaryReconcileResponse(BaseModel):
    """LLM response for resolving a term conflict."""

    chosen_english: str
    reasoning: str


class GlossarySpeechMergeResponse(BaseModel):
    """LLM response for consolidating speech-style observations."""

    consolidated_speech_style: str


# ---------------------------------------------------------------------------
# TranslatedChunk
# ---------------------------------------------------------------------------


class TranslatedChunk(BaseModel):
    """Output of the translation stage for one chunk."""

    chunk_id: str
    source_text: str  # copy of Japanese source
    pass1_translation: str  # Pass 1 output, kept for debugging
    translated_text: str  # final: Pass 2 if double_pass, else Pass 1
    pass_count: int  # 1 or 2
    qa_result: Literal["pass", "fail"] | None = None  # None if QA disabled
    qa_issues: list[str] = []
    total_attempts: int  # count of full chunk translation attempts
    overlap_chunk_id: str | None = None
    summary_generated: str | None = None
    token_usage: dict = {}  # {prompt_tokens, completion_tokens, total_tokens}
    model_used: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
