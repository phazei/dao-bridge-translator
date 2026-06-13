"""Configuration loading and validation.

The full pipeline config is loaded from a YAML file and validated into nested
Pydantic models.  Sections for stages not yet implemented are loaded and
validated but not acted on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("dao_bridge")

# ---------------------------------------------------------------------------
# Language name resolution
# ---------------------------------------------------------------------------

_LANG_NAMES_PATH = Path(__file__).parent / "lang_names.json"
_lang_names_cache: dict[str, str] | None = None


def _load_lang_names() -> dict[str, str]:
    """Load the language code-to-name mapping from ``lang_names.json``."""
    global _lang_names_cache  # noqa: PLW0603
    if _lang_names_cache is not None:
        return _lang_names_cache
    try:
        _lang_names_cache = json.loads(_LANG_NAMES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load lang_names.json — falling back to raw codes")
        _lang_names_cache = {}
    return _lang_names_cache


def resolve_language_name(code: str) -> str:
    """Return the human-readable language name for *code*.

    Falls back to the raw code string if not found in ``lang_names.json``.
    """
    names = _load_lang_names()
    return names.get(code, code)


# ---------------------------------------------------------------------------
# Model config (per-task LLM endpoint)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "default"
    temperature: float | None = None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class ChunkingConfig(BaseModel):
    """Chunking parameters."""

    target_tokens: int = 2000
    max_tokens: int = 2400
    min_chunk_tokens: int = 400
    flex_window_ratio: float = 0.2
    scene_break_patterns: list[str] = Field(
        default_factory=lambda: [
            r"^\s*[\*]{3,}\s*$",
            r"^\s*[◇]{3,}\s*$",
            r"^\s*[＊]{3,}\s*$",
            r"^\s*[・]{3,}\s*$",
            r"^\s*[×](\s*[×])+\s*$",
            r"^\s*[─]{4,}\s*$",
            r"^\s*\*\s+\*\s+\*\s*$",
        ]
    )
    normalize_scene_breaks: str | None = "* * *"
    chunkable_classifications: list[str] = Field(
        default_factory=lambda: [
            "chapter",
            "frontmatter",
            "backmatter",
            "toc_authored",
            "toc_auto",
        ]
    )


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


class GlossaryClusterConfig(BaseModel):
    """Glossary clustering sub-config."""

    max_iterations: int = 3
    """Maximum candidate-generation / merge iterations before stopping."""
    jw_threshold: float = 0.75
    """Jaro-Winkler similarity threshold for candidate pair generation."""
    batch_size: int = 10
    """Number of candidate pairs per LLM confirmation call."""
    auto_merge_enabled: bool = False
    """When True, high-confidence candidate pairs (multiple agreeing heuristics
    plus same category) are merged deterministically without an LLM call. When
    False (the default), every candidate goes to the LLM regardless of
    confidence.

    Defaults to False because the current string-only confidence scorer
    produces false auto-merges on real data: containment + Jaro-Winkler fire
    identically for "qualifier means the same entity" and "qualifier means a
    distinct rank" (see score_candidate_confidence and the addendum in
    build_phases/glossary-cluster-evidence-and-auto-merge.md). Enabling auto-merge
    is only recommended once the scorer gains a corroborating embedding signal."""


class GlossaryCrosscheckConfig(BaseModel):
    """Glossary crosscheck sub-config."""

    enabled: bool = False
    llm_assist: bool = False
    on_conflict: Literal["prefer_master", "prefer_book", "flag_only"] = "prefer_master"


class GlossaryConfig(BaseModel):
    """Glossary parameters."""

    categories: list[str] = Field(
        default_factory=lambda: [
            "character",
            "place",
            "ability",
            "title",
            "term",
            "item",
            "species",
            "clan",
            "organization",
            "other",
        ]
    )
    category_hints: dict[str, str] = Field(
        default_factory=lambda: {
            "character": "Named individuals including full names, given names, family names",
            "clan": (
                "Family names, noble houses, tribes, or group identities "
                "multiple characters belong to"
            ),
            "organization": "Formal groups, guilds, military units, institutions",
        }
    )
    toc_categories: list[str] = Field(
        default_factory=list,
        description=(
            "Glossary categories included in ToC title translation context. "
            "Empty list (default) falls back to the main 'categories' list."
        ),
    )
    master_glossary_path: str | None = None
    cluster: GlossaryClusterConfig = Field(default_factory=GlossaryClusterConfig)
    crosscheck: GlossaryCrosscheckConfig = Field(default_factory=GlossaryCrosscheckConfig)
    promote_on_complete: bool = False


# ---------------------------------------------------------------------------
# Glossary phase
# ---------------------------------------------------------------------------


class GlossaryPhaseConfig(BaseModel):
    """Glossary extraction phase parameters."""

    target_tokens_per_call: int = 8000
    overlap_chunks: int = 0
    min_batch_tokens: int = 1000
    """If the final sub-batch for a spine item has fewer tokens than this,
    absorb it into the previous sub-batch."""
    redistribute_threshold: float = 0.4
    """If the final sub-batch is between *min_batch_tokens* and
    *target_tokens_per_call * redistribute_threshold*, redistribute tokens
    evenly across the last two sub-batches instead of leaving a runt."""


# ---------------------------------------------------------------------------
# Translation phase
# ---------------------------------------------------------------------------


class TranslationPhaseConfig(BaseModel):
    """Translation phase parameters."""

    chunks_per_call: int = 1
    overlap_chunks: int = 1
    cross_spine_overlap: bool = True
    double_pass: bool = True
    rolling_summary: bool = True
    summary_max_tokens: int = 2000
    glossary_injection: Literal["relevant", "all"] = "relevant"
    qa_check: bool = True
    qa_temperature: float | None = 0.1
    qa_max_retries: int = 2
    min_length_ratio: float = 0.3
    max_length_ratio: float = 2.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """EPUB output parameters."""

    epub_path: str = "./book.en.epub"
    title_suffix: str | None = None  # Auto-generated from target language when None
    new_identifier: bool = False
    css: Literal["original", "default"] = "original"
    add_translation_note: bool = True
    run_epubcheck: bool = False


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


class LanguagesConfig(BaseModel):
    """Source and target language identifiers."""

    source: str = "ja"
    target: str = "en"


# ---------------------------------------------------------------------------
# LLM (global retry / timeout settings)
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Global LLM retry and timeout settings."""

    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 300.0


# ---------------------------------------------------------------------------
# Models collection
# ---------------------------------------------------------------------------


class ModelsConfig(BaseModel):
    """Per-task model endpoints.  ``summarize`` falls back to ``translate``."""

    classify: ModelConfig = Field(default_factory=ModelConfig)
    glossary: ModelConfig = Field(default_factory=ModelConfig)
    translate: ModelConfig = Field(default_factory=ModelConfig)
    summarize: ModelConfig | None = None  # falls back to translate if absent


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root configuration loaded from ``config.yaml``."""

    source_epub: str
    work_dir: str = "./work"

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    glossary: GlossaryConfig = Field(default_factory=GlossaryConfig)
    glossary_phase: GlossaryPhaseConfig = Field(default_factory=GlossaryPhaseConfig)
    translation_phase: TranslationPhaseConfig = Field(default_factory=TranslationPhaseConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @property
    def work_dir_path(self) -> Path:
        """Resolved work directory as a :class:`~pathlib.Path`."""
        return Path(self.work_dir).resolve()

    @property
    def source_epub_path(self) -> Path:
        """Resolved source EPUB path."""
        return Path(self.source_epub).resolve()

    def summarize_model(self) -> ModelConfig:
        """Return the summarize model config, falling back to translate."""
        return self.models.summarize or self.models.translate


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the ``config.yaml`` file.

    Returns
    -------
    AppConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pydantic.ValidationError
        If the YAML content fails validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return AppConfig(**raw)
