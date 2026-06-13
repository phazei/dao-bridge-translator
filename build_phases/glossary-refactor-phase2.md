# Glossary Phase 2: Semantic Clustering, Summary Compression, Versioned Memory

## Overview

Phase 1 delivered the entity-centric glossary (surface form pools), the
clustering stage (heuristic candidate generation + LLM confirmation), and the
dormant high-confidence auto-merge machinery. Phase 2 adds the semantic and
temporal layers that the Phase 1 structure was explicitly designed to receive.

Three sub-phases, in dependency order:

- **2A — Embeddings**: semantic candidate generation + the corroborating
  `Evidence.cosine` signal that makes high-confidence auto-merge safe.
- **2B — Summary Compressor**: replace naive summary concatenation with an
  LLM-driven update, producing bounded, higher-quality summaries (which in turn
  improve 2A's embeddings).
- **2C — Versioned Summaries**: temporal summary tracking so translation can ask
  "what was known at chunk N" instead of leaking future/reveal knowledge.

Each sub-phase is independently shippable. 2A delivers the largest capability
gain and should ship first. 2B improves 2A retroactively. 2C builds on 2B's
infrastructure.

The codebase already anticipates this work. Do NOT redesign these seams:

- `glossary_clustering.py` — `Evidence` dataclass has a reserved `cosine: float
  | None` field; `Candidates = dict[CandidatePair, Evidence]`;
  `ClusterConfidence.LOW` is reserved for embedding auto-reject;
  `score_candidate_confidence(evidence)` reads evidence only and performs no
  entity comparison.
- `config.py` — `GlossaryClusterConfig` holds the cluster knobs;
  `auto_merge_enabled` defaults False with a docstring explaining it should flip
  once an embedding signal exists.
- `glossary.py` — `merge_entity_summary()` is a standalone function (the 2B
  replacement point); `_MAX_SUMMARY_LENGTH` constant already exists.
- `schemas.py` — `GlossaryEntity` has `summary`, `first_seen_chunk`,
  `latest_evidence_chunk`; `SurfaceForm` has `translation_variants`. (2B adds a
  transient `summary_observations` accumulator; 2C adds `summary_versions` — both
  additive, neither replaces the scalar `summary`.)

> **Code references in this doc use symbol names, not line numbers.** The three
> sub-phases ship sequentially and 2A/2B edit `glossary.py` substantially, so any
> line number cited here would be stale by the time a later sub-phase is
> implemented. Locate the named function/constant/class (e.g. `grep` for
> `merge_entity_summary`, `find_relevant_entities`, `render_glossary`) rather than
> trusting a position.

---

## Reference Material

### Reference codebase: `context-aware-translation`

Forked at `D:\AITools\context-aware-translation\`. Patterns relevant to 2B/2C:

- `core/term_memory.py` — `TermMemoryVersion` dataclass: `term`,
  `effective_start_chunk`, `latest_evidence_chunk`, `summary_text`, `kind`,
  `source_count`, `created_at`. This is the data shape for versioned summaries
  (2C).
- `core/term_memory_builder.py` — bootstrap + incremental update orchestration.
- `llm/summarizor.py` — the actual prompts: a bootstrap prompt (first summary
  from N observations) and an incremental update prompt that returns either "no
  change" or a revised summary. The update prompt explicitly frames the current
  summary as a tentative hypothesis, which encourages the model to revise rather
  than rubber-stamp. Mirror this framing in 2B.

### Academic grounding

- **LINK-KG** (https://arxiv.org/pdf/2510.26486), Section III-A — the
  type-specific prompt cache: alias→canonical mappings updated context-
  sensitively per chunk. Our entity + surface-form pool is the same idea; 2A's
  embedding step is how we generate cross-form merge candidates the string
  heuristics cannot.
- **LlmLink** (https://aclanthology.org/2025.coling-main.751.pdf) — the
  collaborative memorisation scheme for cross-chunk entity linking; conceptual
  basis for 2C's temporal memory.

### Embedding model

`paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`. ~420MB,
CPU-friendly, strong multilingual (CJK + Latin) performance. Loaded lazily, only
when the embedding heuristic is enabled.

---

## Phase 2A — Embeddings

### Goal

Generate clustering candidate pairs from semantic similarity, catching merges no
string heuristic can reach (`アベル`/`ヴィンセント`, masked-man/emperor,
title/name). Populate `Evidence.cosine` so `score_candidate_confidence` gains a
corroborating signal that distinguishes the false-merge cases that defeated the
string-only scorer (`准仙帝`/`仙帝`).

### Dependency

Add `sentence-transformers` as an optional dependency group, e.g.
`[project.optional-dependencies] embeddings = ["sentence-transformers>=3.0"]`.
The clustering code must degrade gracefully (clear error message) if the
heuristic is enabled but the package is missing.

### Config additions

Extend `GlossaryClusterConfig` in `config.py`:

```python
embedding_enabled: bool = False
"""When True, an embedding heuristic generates additional candidate pairs from
semantic similarity, and Evidence.cosine is populated for all candidate pairs
(including those found by string heuristics) to inform confidence scoring."""

embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
"""SentenceTransformers model name for the embedding heuristic."""

embedding_candidate_threshold: float = 0.72
"""Cosine similarity at/above which a pair becomes an embedding-sourced
candidate. Tuned conservatively; embedding candidates still go to the LLM unless
corroborated by string evidence."""

embedding_auto_merge_min_cosine: float = 0.86
"""Minimum cosine for an embedding signal to corroborate a HIGH (auto-merge)
decision. Pairs below this cannot reach HIGH on string evidence alone once
embeddings are enabled."""

embedding_low_confidence_max_cosine: float = 0.55
"""Embedding-only candidate pairs (no string heuristic fired) with cosine below
this are tiered LOW and auto-rejected without an LLM call. Prevents the wide
embedding net from inflating LLM batch volume with weak semantic neighbours."""
```

These four cosine thresholds are the main tuning surface of 2A, and the defaults
above are starting points, not validated constants. Apply the project's
"measure first" principle (same as the Phase 1 auto-merge decision): run 2A on at
least one real book with `auto_merge_enabled=False` first, inspect the cluster
report's candidate counts and which pairs land in each tier, and adjust before
trusting `auto_merge_enabled=True`. In particular confirm the `准仙帝`/`仙帝`-class
pairs actually fall below `embedding_auto_merge_min_cosine` on the chosen model.

> **Acceptance gate, not just tuning.** The whole auto-merge-safety case rests on
> the adjacent-realm pair landing *below* `embedding_auto_merge_min_cosine`.
> Short, lexically-overlapping CJK strings (`准仙帝` vs `仙帝` share 2 of 3 chars)
> can score high cosine on multilingual models, so this separation is an
> assumption that must be **verified on the chosen model**, not assumed: 2A is not
> "done" until a fixture demonstrates `准仙帝`/`仙帝`-class pairs score below the
> auto-merge cosine. If they do not separate, do NOT ship the auto-merge-flip
> guidance for that model. The separation depends heavily on summary quality —
> the embedding text must carry the distinguishing realm descriptions — so signal
> that distinction clearly in the summaries (this is also why 2B improves 2A). A
> sufficiently capable embedding model that reads the enriched
> `entity_embedding_text` (category + names + summary + hints) will separate
> adjacent-but-distinct realms from genuine aliases when the summaries make the
> distinction explicit.

Add a new strong-signal constant in `glossary_clustering.py` alongside the
existing `HEURISTIC_*` names:

```python
HEURISTIC_EMBEDDING = "embedding"
```

### New module: `glossary_embeddings.py`

Isolate all `sentence-transformers` usage here so the rest of the pipeline has
no hard dependency on it.

```python
"""Embedding-based candidate generation for glossary clustering.

Optional: only imported when GlossaryClusterConfig.embedding_enabled is True.
All sentence-transformers usage is contained in this module.
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
) -> tuple[list[str], "object"]:
    """Return (entity_ids, embedding_matrix) aligned by index.

    The matrix is a normalized float array (N, dim). Caller computes cosine
    via matrix multiply.
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


def cosine_matrix(embeddings) -> "object":
    """Full pairwise cosine matrix (normalized embeddings → matrix @ matrix.T)."""
    return embeddings @ embeddings.T
```

Rationale for the design:

- Glossary-sized collections (hundreds, low thousands) make the full pairwise
  matrix trivial. No FAISS, no vector store.
- `entity_embedding_text` deliberately includes the summary and context hints —
  this is what lets `准仙帝`/`仙帝` separate (their summaries describe adjacent-
  but-distinct realms) and `アベル`/`ヴィンセント` converge (shared "emperor"
  semantics once summaries accumulate). This is also why 2B (better summaries)
  improves 2A retroactively.

#### Recompute-per-iteration is intentional (do not hoist embeddings out of the loop)

Clustering runs **iteratively**: candidate generation re-runs after each merge
round. Embeddings are therefore recomputed each iteration — and this is
**correct, not just a simplification.** After a merge, the surviving entity's
`entity_embedding_text` legitimately changes (surface forms unioned, summary
merged), so its vector shifts. That shift is exactly what surfaces *transitive*
semantic merges: merging A+B can move the combined entity close enough to C to
generate a new candidate that neither A nor B reached alone. Computing embeddings
once before the loop would freeze pre-merge vectors and silently break transitive
clustering. Keep `compute_entity_embeddings` inside the per-iteration candidate
generation.

#### Future optimization: embedding cache (not required for Phase 2)

`compute_entity_embeddings` re-embeds every entity on each call. At a few hundred
entities this is negligible. At a few thousand the 3–4× per-invocation cost
becomes noticeable.

When this matters, cache vectors keyed by `hash(entity_embedding_text(entity))`.
That text is exactly what changes when a merge alters an entity, so entities
untouched by a merge round produce an identical hash and reuse their cached
vector; only changed/new entities are recomputed — which preserves the
correctness property above (changed entities still get fresh vectors) while
skipping the unchanged majority. Acknowledged here for design stability — **not
required for Phase 2 implementation.**

### Integration into `generate_cluster_candidates`

`generate_cluster_candidates(glossary, config)` currently returns a `Candidates`
dict built purely from string heuristics. Extend it (do NOT rewrite the existing
heuristic tagging):

1. Run all existing string heuristics and tag evidence exactly as today.
2. If `config.embedding_enabled`:
   a. Compute embeddings + cosine matrix once.
   b. For every entity pair with cosine >= `embedding_candidate_threshold`, add
      the pair to the evidence dict and tag `HEURISTIC_EMBEDDING`.
   c. For EVERY pair already in the evidence dict (string-sourced or embedding-
      sourced), populate `ev.cosine` from the matrix. This is the key step: the
      cosine score corroborates or contradicts string evidence during scoring,
      even for pairs that strings found.

Sketch (candidate extraction is **vectorized** — `cos` is already a numpy array,
so do not iterate it pair-by-pair in pure Python; that would be O(n²) Python-level
work paid 3–4× per clustering invocation, see the iteration note above):

```python
if config.embedding_enabled:
    import numpy as np
    from dao_bridge.glossary_embeddings import (
        compute_entity_embeddings,
        cosine_matrix,
    )
    entity_ids, emb = compute_entity_embeddings(glossary, config.embedding_model)
    cos = cosine_matrix(emb)
    index_of = {eid: i for i, eid in enumerate(entity_ids)}

    # New embedding-sourced candidates — vectorized upper-triangle threshold.
    iu, ju = np.triu_indices(len(entity_ids), k=1)
    mask = cos[iu, ju] >= config.embedding_candidate_threshold
    for i, j in zip(iu[mask], ju[mask]):
        pair = _ordered_pair(entity_ids[i], entity_ids[j])
        evidence[pair].heuristics.add(HEURISTIC_EMBEDDING)

    # Populate cosine for ALL candidate pairs (string- or embedding-sourced).
    # Only the pairs already in `evidence` are looked up, so this loop is over
    # the candidate set (small), not the full O(n²) matrix.
    for (id_a, id_b), ev in evidence.items():
        ia, ib = index_of.get(id_a), index_of.get(id_b)
        if ia is not None and ib is not None:
            ev.cosine = float(cos[ia][ib])
```

(The `same_category` / `jw_score` / containment annotation loop that follows is
unchanged.)

### Confidence scoring changes

`score_candidate_confidence` keeps its evidence-only contract. Extend it to weigh
`cosine`, but **all cosine-dependent logic is gated on `embedding_enabled`**.

**Hard invariant: with `embedding_enabled=False`, scoring is byte-for-byte Phase 1
behavior.** When embeddings are off, no pair is ever tagged `HEURISTIC_EMBEDDING`
and every `ev.cosine` is `None`. The function must take the same branch it does
today (the "2+ strong string signals AND same category" rule) and never consult
cosine. This is the backward-compatibility contract the OFF-path regression test
asserts.

**`None`-safety:** when embeddings are off (or a pair has no embedding row),
`ev.cosine is None`. A bare `None >= float` raises `TypeError` in Python 3, so
every cosine comparison MUST be guarded by both `embedding_enabled` and
`ev.cosine is not None`. Never compare `cosine` unguarded.

The rules, in order:

0. **Embeddings disabled → Phase 1 path.** If `not embedding_enabled`, evaluate
   the string-only rule exactly as Phase 1 (`2+ strong string signals AND
   same_category → HIGH`, else `MEDIUM`). Return immediately. Rules 1–3 below
   apply ONLY when `embedding_enabled` is True. `LOW` is never returned on this
   path.

1. **Embedding-only weak pairs → LOW (auto-reject).** (embeddings on) If the only
   heuristic is `HEURISTIC_EMBEDDING` (no string signal), `ev.cosine is not None`,
   and `ev.cosine < embedding_low_confidence_max_cosine`, return
   `ClusterConfidence.LOW`. These are weak semantic neighbours; rejecting them
   without an LLM call keeps batch volume sane. (See "Partition interaction"
   below — LOW pairs are dropped before partitioning, never sent to the LLM.)

2. **HIGH requires embedding corroboration.** (embeddings on) The existing "2+
   strong signals AND same category" rule still applies, but additionally require
   `ev.cosine is not None and ev.cosine >= embedding_auto_merge_min_cosine`. This
   is the fix for the `准仙帝`/`仙帝` false merge: it has containment + high JW +
   same category (HIGH under the string-only scorer), but its cosine is depressed
   by the adjacent-realm summaries, so it drops to MEDIUM and goes to the LLM. A
   pair that would have been HIGH on strings but has `cosine is None` or low
   cosine is demoted to MEDIUM (not LOW — string evidence still merits LLM review).

3. **Embedding as a strong signal.** (embeddings on) A `cosine >=
   embedding_auto_merge_min_cosine` counts as one strong signal toward the "2+"
   tally (alongside the existing string strong signals), so a strong embedding +
   one strong string signal + same category can reach HIGH (subject to rule 2,
   which it satisfies by construction).

4. **Everything else → MEDIUM** (unchanged default).

Because `score_candidate_confidence` must remain evidence-only and stateless, it
needs the thresholds and the `embedding_enabled` flag. Pass them in rather than
reading globals — change the signature to `score_candidate_confidence(evidence,
config)` (or pass a small thresholds dataclass that includes `embedding_enabled`).

> **Make the new param OPTIONAL.** The function has more than one caller: the
> `glossary_cluster` partition loop in `glossary.py` *and* the Phase 1 unit tests
> (`TestScoreCandidateConfidence`, `TestScorerKnownFalsePositiveBaseline`), which
> call it with a single argument. Default the new param to `None` and treat `None`
> as the Phase 1 path. This keeps every existing call site (and the byte-for-byte
> OFF-path invariant) working without touching the Phase 1 tests, and the
> production call site simply passes `config`.

The function's docstring currently documents the string-only limitation and the
`准仙帝`/`仙帝` failure. Update it to describe the corroborated behavior, the
`embedding_enabled` gate, and keep the historical note as the reason the embedding
signal exists.

### Partition interaction (LOW handling, batch volume, auto-merge value)

2A increases candidate volume by design — that is the point. The new candidates
flow into the Phase 1 partition (`high_pairs` / `llm_pairs`) and the LLM batch
loop in `glossary_cluster`. Three consequences to handle:

- **LOW pairs are dropped before partitioning.** In the partition loop, a pair
  scored `ClusterConfidence.LOW` is discarded outright — it goes into neither
  `high_pairs` nor `llm_pairs` and is never sent to the LLM. Add this as a third
  branch alongside the existing HIGH/else split. (LOW only occurs with embeddings
  on; with embeddings off the scorer never returns LOW, so the partition loop is
  unchanged on that path.)

- **MEDIUM volume rises.** Embedding-sourced pairs that are not weak enough for
  LOW nor corroborated enough for HIGH land in `llm_pairs`. This is where the
  Phase 1 `batch_size` knob earns its keep; expect more LLM batches per iteration
  than the string-only era. The `embedding_candidate_threshold` and
  `embedding_low_confidence_max_cosine` together throttle how many semantic pairs
  reach the LLM — tune them so the LLM call count stays bounded (see tuning note
  below).

- **Auto-merge value, not just safety.** Phase 1 connected auto-merge *safety* to
  embeddings (cosine corroboration). 2A also restores its *value*: with embeddings
  flooding in more candidates, auto-merging the well-corroborated HIGH pairs is
  what keeps the LLM batch count manageable. This is why `embedding_enabled=True`
  + `auto_merge_enabled=True` is the intended production pairing — embeddings make
  auto-merge both trustworthy and worthwhile.

`id_map` remapping is unchanged: embedding candidates are ordinary
`(entity_id_a, entity_id_b)` pairs keyed into the same `Candidates` dict, so the
auto-merge phase and the LLM batch loop resolve them through the existing
per-iteration `id_map` exactly as Phase 1 does. No new remap logic is needed.

### Auto-merge flip

**Auto-merge is not production-safe without `embedding_enabled=True`.** This is a
codified lesson, not a preference: the string-only scorer produced a 50%
false-merge rate on real data (4 auto-merges, 2 wrong — the `准仙帝`/`仙帝`
adjacent-realm case and the `Huang`/`Da Zhuang` coincidental-JW case). Embedding
corroboration is what makes the HIGH tier trustworthy, because cosine distance
separates the qualifier-means-same-entity case from the qualifier-means-distinct-
thing case that fires identical string evidence.

`auto_merge_enabled` stays user-controlled and its default stays False. Do not
silently change the default; flipping it is a user decision. But the guidance is
unambiguous: enabling `auto_merge_enabled=True` without `embedding_enabled=True`
is unsupported. The recommended (and only production-safe) combination is
`embedding_enabled=True` + `auto_merge_enabled=True`.

Update the `auto_merge_enabled` docstring in `config.py` to state this directly:
auto-merge is not production-safe without embeddings; cite the 50% false-merge
rate and the `准仙帝`/`仙帝` case as the reason. The docstring is where someone
actually decides to flip the flag, so the warning has to live there, not only in
this spec.

### Regression test flip

Phase 1 left an annotated regression test asserting that a "containment + JW +
same category" pair scores HIGH under the string-only scorer. In 2A:

- With embeddings OFF: that test still asserts HIGH (string-only behavior
  unchanged — backward compatible).
- With embeddings ON and a low-cosine fixture for the same pair: assert it now
  scores MEDIUM. This is the visible, intended behavior change that proves the
  corroboration works. Use the `准仙帝`/`仙帝` shapes as the fixture: same string
  evidence, depressed cosine → MEDIUM.

### 2A tests

- `entity_embedding_text` drops empty parts, includes summary + hints.
- `compute_entity_embeddings` returns aligned ids/matrix; normalized rows.
- Embedding candidate generation adds pairs above threshold, populates `cosine`
  on all pairs.
- Scoring: embedding-only + low cosine → LOW.
- Scoring: containment + JW + same category + low cosine → MEDIUM (the fix).
- Scoring: containment + JW + same category + high cosine → HIGH.
- Scoring: strong embedding + one strong string signal + same category → HIGH.
- Graceful error when `embedding_enabled=True` but package missing.
- Embeddings OFF → `cosine` stays None, behavior identical to Phase 1.
- Scoring with embeddings OFF never returns LOW and never touches `cosine` (the
  `None`-safety / backward-compat invariant).

**Test dependency note:** `sentence-transformers` pulls in `torch` (large) and is
an optional group. Tests that actually load a model must `pytest.importorskip(
"sentence_transformers")` (or be marked and skipped) so the default test run and
CI without the extra installed stay green. Pure-logic tests (scoring rules,
threshold gating, `entity_embedding_text` string assembly) should NOT require the
package — keep them importable by not importing `sentence_transformers` at module
top level (the lazy `_load_model` pattern already ensures this for the source;
mirror it in tests).

### 2A implementation results — model bake-off & threshold tuning

The acceptance gate was run via `scripts/probe_embeddings.py`, which embeds four
representative entity pairs — two genuine aliases that SHOULD merge
(`ペテルギウス`/`ペテルギウス` romanisation variants; `アベル`/`ヴィンセント` cross-name
identity reveal) and two adjacent-but-distinct pairs that must NOT auto-merge
(`准仙帝`/`仙帝` adjacent realms; `Huang`/`Da Zhuang` coincidental-JW) — at two
summary densities ("thin" stress-test summaries and "dense" post-reveal converged
summaries representing the full-book + 2B production input).

**Model comparison (dense / production summaries, CPU):**

| Model | Disk | Gate | Distinct pairs < auto floor? | Alias recall | Verdict |
|-------|------|------|------------------------------|--------------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~0.5GB | FAIL | No — `准仙帝/仙帝` 0.889 | poor | too small |
| `intfloat/multilingual-e5-large` | ~2.2GB | FAIL | No — 0.943 & 0.877 | overlap | high-band cosines; merge & distinct overlap |
| `BAAI/bge-m3` | ~2.3GB | PASS | Yes — 0.792 & 0.603 | misses `アベル` (0.690) | viable alternate |
| **`Qwen/Qwen3-Embedding-0.6B`** | **~1.2GB** | **PASS** | **Yes — 0.768 & 0.495** | **catches `アベル` (0.595)** | **selected default** |
| `lainlives/Qwen3-Embedding-4B-bnb-4bit` | ~2.7GB | FAIL | No — 0.956 & 0.892 | overlap | **4-bit quant collapse** — all CJK crushed to 0.89–0.98 |
| `Octen/Octen-Embedding-4B-INT8` | ~4.4GB | FAIL (×0.006) | `Huang` 0.514 OK; `准仙帝/仙帝` 0.826 just over 0.82 | catches `アベル` (0.643) | near-miss; INT8 does NOT collapse but 4.4GB & worse on the hardest distinct pair |

Qwen3-Embedding-0.6B gave the cleanest separation, the smallest download, and runs
full-precision (no bitsandbytes/CUDA hard requirement, so it stays CPU-portable).
Its dense cosines: `Petelgeuse` alias 0.923; `准仙帝/仙帝` distinct 0.768;
`アベル/ヴィンセント` alias 0.595; `Huang/Da Zhuang` distinct 0.495.

**Quantization finding:** the 4B-**4bit** model collapsed all same-language CJK
pairs into a high, overlapping band (0.89–0.98), destroying separation. The 4B-
**INT8** model did NOT collapse — its spread mirrors the 0.6B's shape (`Huang`
0.514, `アベル` 0.643) — proving the collapse was specifically the 4-bit
quantization, not the 4B model. INT8 still narrowly failed the gate (`准仙帝/仙帝`
0.826 vs the 0.82 floor) and is a 4.4GB download, so the small full-precision
0.6B remains the better pick. Takeaway: avoid 4-bit for embedding models; prefer
a smaller full-precision model over a larger heavily-quantized one.

> **Note on test hardware:** the project `.venv` carries a CUDA-enabled torch
> (`+cu130`); encode times observed were ~30–100 ms/text on GPU. A CPU-only torch
> build also works (the 0.6B is CPU-friendly by design) and remains the portable
> baseline — embeddings are an opt-in `[embeddings]` extra and never require a GPU.

**Tuned thresholds (Qwen3 scale, now the config defaults):**

- `embedding_auto_merge_min_cosine = 0.82` — the gate-critical knob: above the
  highest distinct pair (0.768) with margin, below genuine aliases (0.923), so
  `准仙帝/仙帝` is demoted to MEDIUM (LLM review) instead of auto-merged.
- `embedding_candidate_threshold = 0.55` — catches the hardest genuine alias
  (`アベル/ヴィンセント` 0.595) while leaving the coincidental distinct pair
  (`Huang/Da Zhuang` 0.495) below threshold (never surfaced).
- `embedding_low_confidence_max_cosine = 0.50` — kept at/just below the candidate
  floor so it never auto-rejects a genuine low-scoring cross-name alias.

**Key findings / limitations:**

- **Cosine scales are NOT portable across models.** The defaults that pass on
  MiniLM (per this doc's original guidance) fail on e5 and need different values
  on Qwen3/bge-m3. Changing `embedding_model` REQUIRES re-running the probe and
  re-tuning. This is now stated in the config docstrings.
- **Auto-merge safety is achieved** (distinct pairs stay off the auto tier) — so
  `embedding_enabled=True` + `auto_merge_enabled=True` is now defensible on Qwen3.
- **Cross-name identity-reveal recall remains inherently limited.** Both passing
  models rank `アベル/ヴィンセント` (zero shared surface text; the link is a plot
  reveal) BELOW the adjacent-realm distinct pair. Embeddings surface it as an LLM
  *candidate* but never as a high-confidence auto-merge — the string heuristics +
  LLM cluster pass remain the backstop for those. Richer summaries (2B) raise this
  case's cosine (thin 0.372 → dense 0.595 on Qwen3), confirming the 2B dependency.
- The probe corpus is only 4 pairs — thresholds are mechanism-justified, not
  statistically fit. Expand the corpus before treating these as final constants.

---

## Phase 2B — Summary Compressor

### Goal

Replace the naive concatenation in `merge_entity_summary` (`glossary.py`) with
an LLM-driven update that keeps summaries bounded, deduplicated, and focused on
translation-relevant facts (identity, role, category, relationships,
distinguishing features) — not plot events. Better summaries → better embeddings
(2A) and better translation context.

### Current behavior (to replace)

`merge_entity_summary(entity, summary_update, chunk_id)` appends the new
observation to `entity.summary` with substring-dedup and truncates at
`_MAX_SUMMARY_LENGTH`. It always sets `latest_evidence_chunk`.

### New behavior

Two-mode compressor mirroring the reference repo's `summarizor.py`:

- **Bootstrap**: when `entity.summary` is empty, a single observation becomes the
  summary directly (no LLM call needed for one sentence); multiple accumulated
  observations are compressed into a first summary in one LLM call.
- **Incremental update**: when `entity.summary` exists and a new
  `summary_update` arrives, call the LLM with the current summary + the new
  observation. The model returns either "no change" or a revised summary.

> **The bootstrap-vs-incremental distinction is a strategy-2 concern.** Under
> deferred compression (strategy 1, recommended) there is no per-observation
> processing during build — the whole accumulated list is compressed in one pass
> per entity, so every entity is effectively bootstrapped exactly once and the
> incremental-update mode never runs. The two modes only both apply to strategy 2
> (inline/threshold compression), where an existing summary is revised by a new
> observation mid-build.

To avoid an LLM call on every single mention during build (expensive on long
books), batch this. Two acceptable strategies — pick one and document it:

1. **Deferred compression (recommended).** During build, accumulate raw
   observations cheaply into a **dedicated transient field** on `GlossaryEntity`,
   recording the originating chunk id with each observation so 2C can build a
   correct timeline without re-running build (e.g.
   `summary_observations: list[SummaryObservation]` where each carries
   `chunk_id` + `text`, or `list[tuple[str, str]]`) — **do NOT overload the
   published `summary` field as a scratch buffer.** `summary` is read by exports,
   reports, and (2A) the embedding text; if a build crashes before the deferred
   compression pass runs, an overloaded `summary` would persist raw
   newline-joined observations and feed that junk straight into embeddings and
   exports. Keeping observations in a separate field means a half-built glossary
   still has a clean (possibly empty) `summary`. Then run a dedicated compression
   pass at the end of build (or start of cluster) that compresses each entity's
   `summary_observations` in one LLM call per entity and writes the result to
   `summary`. This bounds LLM calls to O(entities), not O(mentions).
2. **Threshold compression.** Compress inline but only when accumulated raw text
   exceeds a length/observation-count threshold, so most mentions stay cheap.

Recommended: strategy 1. It's predictable, keeps build fast, keeps `summary`
crash-clean, and the compression pass slots naturally before clustering (so
clusters and embeddings see compressed summaries).

**`latest_evidence_chunk` ownership.** Today `merge_entity_summary` always sets
`entity.latest_evidence_chunk` inline (in `merge_entity_summary`). Under deferred
compression that responsibility moves: the build-time observation accumulator
sets `latest_evidence_chunk` as each observation is recorded (cheap, no LLM), and
the compression pass leaves it untouched (compression rewrites text, not
provenance). When `summary_compress_enabled=False`, the inline naive path keeps
its current behavior unchanged. State this explicitly so neither path leaves the
field stale.

### New prompt: `prompts/glossary_summary_compress.txt`

Model the framing on the reference repo's update prompt:

- State the entity's category and canonical name for context.
- Provide the accumulated observations (or current summary + new observation).
- Instruct: produce ONE concise summary capturing only translation-relevant
  facts — identity, role/category, key relationships, aliases/identity reveals,
  distinguishing traits. Drop plot events, one-off actions, scene details.
- Frame the existing summary as a tentative hypothesis that should be revised if
  new evidence contradicts or refines it (this encourages real revision over
  rubber-stamping — directly from the reference repo's approach).
- The summary must NOT restate `context_hints`. Hints already feed the embedding
  text separately (2A), so duplicating them in the summary pollutes both the
  summary and the embedding. The summary captures stable identity; hints carry
  low-confidence per-form observations.
- Bound the output length explicitly.

Include a concrete before/observation/after example in the prompt as a few-shot
(local models benefit measurably from a worked example):

```text
Current summary:  A fugitive traveling under the name Abel.
New observation:  Several characters imply Abel is connected to the emperor.
Revised summary:  A fugitive traveling under the name Abel with apparent ties to
                  Vollachian royalty.
```

Add a response model in `schemas.py`. Use the minimal no-change pattern — on the
unchanged path the model returns only `changed: false` and omits the summary,
so it never has to re-emit the unchanged text:

```python
class GlossarySummaryCompressResponse(BaseModel):
    """LLM response for summary compression."""
    changed: bool
    summary: str | None = None  # present only when changed is True
```

When `changed` is False, skip the write entirely (summary untouched).

### Config additions

Extend `GlossaryConfig` (or a new `GlossarySummaryConfig` sub-config):

```python
summary_compress_enabled: bool = False
"""When True, entity summaries are compressed by an LLM pass rather than naive
concatenation."""

summary_max_length: int = 500
"""Target maximum character length for compressed summaries."""
```

Keep `summary_compress_enabled=False` as default so build behavior is unchanged
until opted in. When False, `merge_entity_summary` uses the existing
concatenation path.

### Wiring

- If deferred compression: add `glossary_summary_compress(work_dir, config,
  state, ...)` as either a small stage between build and cluster, or a step at
  the tail of build. Track state per entity batch for resumability, following
  existing stage patterns. Read from build output, write compressed summaries
  back. Cluster/embeddings then consume the compressed glossary.
- Reuse the `LLMClient(config.models.glossary, config.llm)` pattern already used
  by build/reconcile/cluster.

### 2B tests

- Bootstrap: empty summary + first observation → summary set to observation, no
  LLM call.
- Incremental: existing summary + new observation → LLM called, summary updated.
- No-change path (if implemented): LLM says unchanged → summary untouched.
- Deferred pass: O(entities) LLM calls, not O(mentions).
- `summary_compress_enabled=False` → identical to Phase 1 concatenation.
- Compressed summary respects `summary_max_length`.

---

## Phase 2C — Versioned Summaries

### Goal

Track summaries as chronological versions so translation-time injection can use
the summary that was true *as of the chunk being translated*, preventing
identity-reveal leakage and retcon knowledge. Translating chapter 3 should not
know that Abel is Vincent Volakia if that reveal happens in chapter 15.

### Data shape

Mirror the reference repo's `TermMemoryVersion`. Add to `schemas.py`:

```python
class SummaryVersion(BaseModel):
    """A chronological version of an entity's summary."""
    summary_text: str
    effective_start_chunk: str  # first chunk this version is valid from
    latest_evidence_chunk: str  # most recent chunk contributing to this version
    source_count: int = 1       # number of observations folded into this version
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# On GlossaryEntity, add:
    summary_versions: list[SummaryVersion] = Field(default_factory=list)
```

Keep the scalar `summary` field as the latest/global summary for backward compat
and for reports/exports that don't care about chunk position. `summary_versions`
is additive.

### Building versions

This builds on 2B's compressor. Each time the compressor produces a materially
revised summary (not a no-change), append a new `SummaryVersion` with
`effective_start_chunk` = the chunk that triggered the revision. Versions form a
timeline: version N is valid from its `effective_start_chunk` until the next
version's `effective_start_chunk`.

The compression must process observations in **chunk order** for the timeline to
be correct. The deferred-compression pass (2B strategy 1) must therefore sort
each entity's `summary_observations` by chunk id before compressing. This implies
2B's accumulator should record the originating chunk id alongside each
observation (e.g. `list[tuple[chunk_id, observation]]`, or a small record type)
so 2C can sort and assign `effective_start_chunk` correctly. If 2B ships before
2C, prefer storing the chunk id with each observation from the start so 2C does
not require re-running build.

### Translation-time lookup

Add a helper used by the translation glossary-injection path:

```python
def summary_as_of(entity: GlossaryEntity, chunk_id: str) -> str | None:
    """Return the summary version effective at chunk_id.

    Semantics:
      - No versions at all: fall back to the scalar `summary`.
      - chunk_id is at/after the earliest version's effective_start_chunk:
        return the latest version whose effective_start_chunk <= chunk_id.
      - chunk_id is BEFORE the earliest version's effective_start_chunk
        (the entity had not yet been "introduced" as of this chunk): return
        None. Do NOT leak the earliest version backwards in time.
    """
```

The pre-first-version case returning `None` is deliberate and safe: it does not
suppress the entity from injection. `find_relevant_entities` (in `translate.py`)
already gates injection on a surface form appearing in the chunk text, so an
entity that is textually present but not yet summarized still gets injected — just
with no summary line. That is correct: the name should be translated
consistently, but no pre-introduction lore is leaked. Chunk-id comparison is
lexical (see "Important constraints" — zero-padded sortable ids).

Wire this into wherever entities are rendered for the translation prompt
(`render_glossary` / its per-entity rendering in `translate.py`). The
translation stage knows the current chunk id; it should request the
chunk-appropriate summary rather than the global one. Surface forms,
translations, speech style, etc. remain as-is — only the summary becomes
temporal.

### Important constraints

- **Clustering and embeddings should use the LATEST (global) summary**, not a
  versioned one — clustering is a whole-book reconciliation operation where
  knowing the full identity picture is correct and desirable. Only
  translation-time injection uses `summary_as_of`. Make this explicit in code
  comments so a later change doesn't accidentally feed versioned summaries into
  clustering and re-fragment entities.
- **Guiding principle: temporal correctness is more important than aggressive
  memory consolidation.** Entity merges must not collapse or discard version
  history to simplify storage. If a tradeoff arises between a tidier single
  summary and preserving the per-chunk timeline, preserve the timeline.
- Chunk ids are zero-padded sortable strings (e.g. `0012.010`), so string
  comparison gives correct chronological order. Confirm this holds for the id
  format in use and use string comparison consistently.

### Config additions

```python
summary_versioning_enabled: bool = False
"""When True, entity summaries are tracked as chronological versions and
translation injection uses the chunk-appropriate version."""
```

Default False. When False, translation uses the scalar `summary` (Phase 1/2B
behavior).

### 2C tests

- Versions appended in chunk order; timeline boundaries correct.
- `summary_as_of` returns the right version for a chunk before/after a reveal.
- `summary_as_of` with no versions falls back to scalar summary.
- Clustering/embeddings use global summary, not versioned (guard test).
- Pre-reveal chunk does not see post-reveal identity in the rendered prompt.
- `summary_versioning_enabled=False` → translation uses scalar summary.

---

## Implementation Order & Shipping

1. **2A first.** Largest capability gain; the seams already exist. Ships as:
   `glossary_embeddings.py`, config additions, `generate_cluster_candidates`
   extension, `score_candidate_confidence` corroboration, regression-test flip.
   Only after 2A is `auto_merge_enabled=True` production-safe, and only in
   combination with `embedding_enabled=True`.
2. **2B second.** Improves 2A's embeddings retroactively (better summaries embed
   better). Ships as: compress prompt + response model, `merge_entity_summary`
   replacement / deferred compression pass, config additions.
3. **2C last.** Builds on 2B's chunk-ordered compression. Ships as:
   `SummaryVersion` schema, versioned build, `summary_as_of`, translation wiring.

Each sub-phase keeps its feature flag defaulted OFF so the pipeline's default
behavior is unchanged until explicitly opted in.

## Non-Goals for Phase 2

- No FAISS / external vector store (glossary is small; full matrix is fine).
- No relationship graph or faction modeling (summaries stay free-text).
- No external NER / coreference models — the LLM + embeddings do the work.
- No master-glossary redesign (cross-book entity matching is a later phase).
- No re-architecture of the Phase 1 clustering control flow — 2A plugs into the
  existing `Evidence`/`Candidates`/scorer seams.