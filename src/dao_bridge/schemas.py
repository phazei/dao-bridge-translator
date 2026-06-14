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


class SummaryObservation(BaseModel):
    """A single raw summary observation accumulated during glossary build.

    Each observation is one concise sentence (the ``summary_update`` emitted
    by the extraction LLM for a mention) tagged with the chunk it came from.
    During build these accumulate on :class:`GlossaryEntity.summary_observations`
    and are later compressed (Phase 2B) into the scalar ``summary``.

    The ``chunk_id`` provenance is retained on disk so the versioned-summary
    stage (Phase 2C) can reconstruct a chronological timeline without
    re-running build.
    """

    chunk_id: str  # Originating chunk, e.g. "0003.012"
    text: str  # The observation sentence


class SurfaceForm(BaseModel):
    """A source-language text form that refers to an entity.

    Each surface form carries its own target-language translation.  For
    example, ``アベル → Abel`` and ``ヴィンセント・ヴォラキア皇帝 → Emperor
    Vincent Volakia`` may both belong to the same :class:`GlossaryEntity`,
    but they translate differently depending on which form appears in the
    source text.
    """

    source: str  # The source-language string, e.g. "アベル"
    reading: str | None = None  # From furigana, if available
    translation: str  # Target-language rendering for THIS specific form
    translation_variants: list[str] = Field(default_factory=list)
    # Alternate translations discovered during clustering merges.
    # Reconcile inspects these to resolve translation conflicts.
    context_hints: list[str] = Field(default_factory=list)
    notes: str | None = None
    first_seen_chunk: str | None = None
    occurrence_count: int = 1


class GlossaryEntity(BaseModel):
    """A single entity in the glossary.

    An entity owns a pool of :class:`SurfaceForm` objects — all the
    source-language strings that refer to this person, place, item, or
    concept.  The *canonical_name* is the primary name used in
    reports and logs; individual surface forms carry their own per-form
    translations.
    """

    entity_id: str  # Stable slug, e.g. "character_000012"
    category: str  # Validated against config.glossary.categories
    canonical_name: str  # Primary target-language name, e.g. "Abel"
    summary: str | None = None  # Accumulated understanding of this entity
    summary_observations: list[SummaryObservation] = Field(default_factory=list)
    # Raw per-chunk summary observations accumulated during build (Phase 2B).
    # Compressed into ``summary`` by the build-tail compression pass and
    # retained on disk for the versioned-summary stage (Phase 2C). Additive:
    # never replaces the scalar ``summary``.
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
    translation: str  # Proposed target-language rendering for this form
    category: str  # One of the allowed categories
    summary_update: str | None = None  # Concise sentence about what this entity appears to be
    context_hint: str | None = None  # Low-confidence hint, e.g. "same person as アベル"
    notes: str | None = None
    aliases: list[str] = Field(default_factory=list)
    nicknames: dict[str, str] = Field(default_factory=dict)
    speech_style: str | None = None


class GlossaryCorrectionEntry(BaseModel):
    """A correction proposed by the LLM during glossary extraction."""

    existing_translation: str
    source_term: str
    corrected_translation: str
    reason: str


class GlossaryExtractionResponse(BaseModel):
    """Top-level LLM response for glossary extraction."""

    mentions: list[ExtractedMention] = Field(default_factory=list)
    corrections: list[GlossaryCorrectionEntry] = Field(default_factory=list)


class GlossaryReconcileResponse(BaseModel):
    """LLM response for resolving a term conflict."""

    chosen_translation: str
    reasoning: str


class GlossarySpeechMergeResponse(BaseModel):
    """LLM response for consolidating speech-style observations."""

    consolidated_speech_style: str


class GlossarySummaryCompressResponse(BaseModel):
    """LLM response for entity summary compression (Phase 2B)."""

    summary: str


class GlossaryClusterDecision(BaseModel):
    """LLM decision for a single candidate entity pair during clustering."""

    entity_id_a: str
    entity_id_b: str
    same_entity: bool
    preferred_entity_id: str | None = None
    preferred_canonical_name: str | None = None
    reasoning: str


class GlossaryClusterResponse(BaseModel):
    """LLM response for a batch of clustering candidate pairs."""

    decisions: list[GlossaryClusterDecision] = Field(default_factory=list)


class TocTranslationResponse(BaseModel):
    """LLM response for ToC title translation."""

    titles: list[str]


# ---------------------------------------------------------------------------
# TranslatedChunk
# ---------------------------------------------------------------------------


class TranslatedChunk(BaseModel):
    """Output of the translation stage for one chunk."""

    chunk_id: str
    source_text: str  # copy of source-language text
    pass1_translation: str  # Pass 1 output, kept for debugging
    pass1_analysis: str | None = None  # <analysis> block from Pass 1 (stripped from translation)
    translated_text: str  # final: Pass 2 if double_pass, else Pass 1
    pass_count: int  # 1 or 2
    qa_result: Literal["pass", "fail"] | None = None  # None if QA disabled
    qa_issues: list[str] = []
    total_attempts: int  # count of full chunk translation attempts
    selected_attempt: int = 1  # which attempt this saved record came from
    overlap_chunk_id: str | None = None
    summary_generated: str | None = None
    token_usage: dict = {}  # {prompt_tokens, completion_tokens, total_tokens}
    model_used: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
