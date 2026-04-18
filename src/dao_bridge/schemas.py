"""Pydantic schemas for manifest, chunks, glossary, and translations.

These are the complete schemas for the entire pipeline.  Later prompts will
use them as-is.  Fields not populated by early stages are nullable and default
to ``None``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def padded_id(self) -> str:
        """Zero-padded 3-digit spine index string."""
        return f"{self.spine_index:03d}"


class Manifest(BaseModel):
    """Book-level manifest persisted to ``manifest.json``."""

    source_epub_path: str
    book_id: str
    opf_dir: str = ""  # OPF directory within the EPUB ZIP (e.g. "OEBPS")
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

    japanese: str | None = None  # None for English-reference-only imports
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
