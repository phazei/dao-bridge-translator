"""Embedding-based candidate generation for glossary clustering.

Optional: only imported when GlossaryClusterConfig.embedding_enabled is True.
All sentence-transformers usage is contained in this module so the rest of the
pipeline has no hard dependency on it.
"""

from __future__ import annotations

import logging

from dao_bridge.schemas import Glossary, GlossaryEntity

logger = logging.getLogger("dao_bridge")

_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    """Lazily load and cache a SentenceTransformer model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding clustering requires sentence-transformers. "
                "Install with: pip install dao-bridge-translator[embeddings]"
            ) from exc
        logger.info("Loading embedding model %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def entity_embedding_text(entity: GlossaryEntity) -> str:
    """Build the enriched text used to embed an entity.

    Combines category, canonical name, all surface form sources + translations,
    the summary, and any context hints. Empty parts are dropped so the joined
    string has no dangling separators.

    The summary and context hints are deliberately included: this is what lets
    adjacent-but-distinct entities separate (e.g. ``准仙帝``/``仙帝`` whose
    summaries describe adjacent-but-distinct cultivation realms) and genuine
    aliases converge (e.g. ``アベル``/``ヴィンセント`` once their summaries both
    accrue "emperor" semantics). Better summaries (Phase 2B) therefore improve
    this signal retroactively.
    """
    sources = ", ".join(sf.source for sf in entity.surface_forms if sf.source)
    translations = ", ".join(
        sf.translation for sf in entity.surface_forms if sf.translation
    )
    hints = " ".join(
        h for sf in entity.surface_forms for h in sf.context_hints if h
    )
    parts = [
        entity.category,
        entity.canonical_name,
        sources,
        translations,
        entity.summary or "",
        hints,
    ]
    return ". ".join(p for p in parts if p)


def compute_entity_embeddings(
    glossary: Glossary,
    model_name: str,
) -> tuple[list[str], object]:
    """Return ``(entity_ids, embedding_matrix)`` aligned by index.

    The matrix is a normalized float array ``(N, dim)``. Caller computes cosine
    via matrix multiply (see :func:`cosine_matrix`).
    """
    model = _load_model(model_name)
    entity_ids = [e.entity_id for e in glossary.entities]
    texts = [entity_embedding_text(e) for e in glossary.entities]
    # normalize_embeddings=True so cosine == dot product.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return entity_ids, embeddings


def cosine_matrix(embeddings) -> object:
    """Full pairwise cosine matrix.

    Inputs are normalized embeddings, so cosine similarity is simply
    ``matrix @ matrix.T``. Glossary-sized collections (hundreds, low thousands)
    make the full pairwise matrix trivial — no FAISS, no vector store.
    """
    return embeddings @ embeddings.T
