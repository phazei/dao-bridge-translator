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


class SurfaceForm(BaseModel):
    """A source-language text form that refers to an entity.

    Each surface form carries its own English rendering.  For example,
    ``アベル → Abel`` and ``ヴィンセント・ヴォラキア皇帝 → Emperor Vincent
    Volakia`` may both belong to the same :class:`GlossaryEntity`, but
    they translate differently depending on which form appears in the
    source text.
    """

    source: str  # The source-language string, e.g. "アベル"
    reading: str | None = None  # From furigana, if available
    english: str  # English rendering for THIS specific form
    context_hints: list[str] = Field(default_factory=list)
    notes: str | None = None
    first_seen_chunk: str | None = None
    occurrence_count: int = 1


class GlossaryEntity(BaseModel):
    """A single entity in the glossary.

    An entity owns a pool of :class:`SurfaceForm` objects — all the
    source-language strings that refer to this person, place, item, or
    concept.  The *canonical_english* is the primary name used in
    reports and logs; individual surface forms carry their own per-form
    English renderings for translation.
    """

    entity_id: str  # Stable slug, e.g. "character_000012"
    category: str  # Validated against config.glossary.categories
    canonical_english: str  # Primary English name, e.g. "Abel"
    summary: str | None = None  # Accumulated understanding of this entity
    surface_forms: list[SurfaceForm] = Field(default_factory=list)

    # Carried from previous schema
    aliases: list[str] = Field(default_factory=list)
    nicknames: dict[str, str] = Field(default_factory=dict)
    speech_style: str | None = None  # Prose description, characters only
    notes: str | None = None
    source: GlossarySource = "extracted"
    source_books: list[str] = Field(default_factory=list)

    # Temporal tracking
    first_seen_chunk: str | None = None
    latest_evidence_chunk: str | None = None


class Glossary(BaseModel):
    """Per-book or master glossary (entity-centric, v2)."""

    entities: list[GlossaryEntity] = Field(default_factory=list)
    version: int = 2
    book_id: str | None = None  # None for master glossary
    book_metadata: dict = {}  # title, author, volume; per-book only
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Glossary LLM response models
# ---------------------------------------------------------------------------


class ExtractedMention(BaseModel):
    """A raw mention observed by the extraction LLM in a chunk/batch.

    This is a temporary observation — build code decides whether it
    attaches to an existing entity or creates a new one.
    """

    source: str  # Exact source-language term as written
    reading: str | None = None  # Pronunciation from furigana, else null
    english: str  # Proposed English rendering for this form
    category: str  # One of the allowed categories
    summary_update: str | None = None  # Concise sentence about what this entity appears to be
    context_hint: str | None = None  # Low-confidence hint, e.g. "same person as アベル"
    notes: str | None = None
    aliases: list[str] = Field(default_factory=list)
    nicknames: dict[str, str] = Field(default_factory=dict)
    speech_style: str | None = None


class GlossaryCorrectionEntry(BaseModel):
    """A correction proposed by the LLM during glossary extraction."""

    existing_english: str
    source_term: str
    corrected_english: str
    reason: str


class GlossaryExtractionResponse(BaseModel):
    """Top-level LLM response for glossary extraction."""

    mentions: list[ExtractedMention] = Field(default_factory=list)
    corrections: list[GlossaryCorrectionEntry] = Field(default_factory=list)


class GlossaryReconcileResponse(BaseModel):
    """LLM response for resolving a term conflict."""

    chosen_english: str
    reasoning: str


class GlossarySpeechMergeResponse(BaseModel):
    """LLM response for consolidating speech-style observations."""

    consolidated_speech_style: str


class TocTranslationResponse(BaseModel):
    """LLM response for ToC title translation."""

    titles: list[str]


# ---------------------------------------------------------------------------
# TranslatedChunk
# ---------------------------------------------------------------------------


class TranslatedChunk(BaseModel):
    """Output of the translation stage for one chunk."""

    chunk_id: str
    source_text: str  # copy of Japanese source
    pass1_translation: str  # Pass 1 output, kept for debugging
    pass1_analysis: str | None = None  # <analysis> block from Pass 1 (stripped from translation)
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
