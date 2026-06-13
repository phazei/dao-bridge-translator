# dao-bridge-translator: Clustering Candidate Evidence + Confidence-Tiered Auto-Merge

## Status

Design / planning. Not yet implemented.

This doc supersedes the loose "Confidence-Tiered Auto-Merge" notes that assumed
candidate pairs already carried a score. They do not — see "Root Problem" below.
This spec fixes that first, then layers auto-merge on top.

Embeddings are **out of scope** here. This work only *prepares* the ground for
embeddings (see "Embeddings Readiness"); it does not add an embedding heuristic,
a model dependency, or any config for one.

---

## Mental Model

```text
Candidate generation:  Which entity pairs might be duplicates, and WHY (which
                       heuristics fired, how strong)?
Confidence scoring:    Given the recorded evidence, is this pair obviously the
                       same entity, or genuinely ambiguous?
Auto-merge:            Obvious pairs merge deterministically (no LLM call).
LLM confirmation:      Ambiguous pairs go to the LLM, exactly as today.
```

The single new idea: **candidate generation must keep the evidence it discovers**,
instead of collapsing everything into an anonymous set of ID pairs. Once evidence
travels with each pair, both confidence-tiered auto-merge (this doc) and future
embedding candidates (later) become clean additions instead of hacks.

---

## Root Problem

`generate_cluster_candidates` (`src/dao_bridge/glossary_clustering.py:227`) runs
five heuristics and unions their results:

```python
candidates: set[CandidatePair] = set()
candidates |= _source_substring_candidates(glossary)
candidates |= _translation_containment_candidates(glossary)
candidates |= _shared_reading_candidates(glossary)
candidates |= _alias_overlap_candidates(glossary)
candidates |= _jw_similarity_candidates(glossary, threshold=config.jw_threshold)
```

`CandidatePair = tuple[str, str]` (`glossary_clustering.py:33`) — just
`(entity_id_a, entity_id_b)`. The set-union is the lossy step. When two heuristics
flag the same pair, the set holds one identical tuple, and we lose:

- **which** heuristics fired (one weak signal vs. several strong ones agreeing)
- **how strong** the Jaro-Winkler signal was (`_jw_any_match` returns `bool`, not
  the score — `glossary_clustering.py:197`)
- **direction** of containment (A ⊂ B vs. B ⊂ A)

Because that information is discarded, any confidence scorer is forced to
**recompute all five heuristics from scratch** for each pair. Recomputation is the
wrong approach: it duplicates logic and risks **drift** — the inline reimplementation
can subtly disagree with the canonical predicate (e.g. the `len <= 1` skip rules at
`glossary_clustering.py:69` and `:111`), so a pair could be a *candidate* under one
rule and score differently under a near-identical inline rule.

**Fix: keep the evidence with the pair.** Compute once, carry it forward.

---

## Blast Radius (why this is a small change)

The `set[CandidatePair]` return shape is **confined to the clustering module**. It is
never persisted, never serialized, and never reaches reconcile / export / translate.

The only production consumer is the cluster loop in
`src/dao_bridge/glossary.py`:

- `candidates = generate_cluster_candidates(...)` (`glossary.py:1732`)
- `if not candidates:` — truthiness (`:1734`)
- `candidate_list = sorted(candidates)` (`:1742`)
- batching + iterating `(eid_a, eid_b)` pairs (`:1745-1747`, `:1769`)

The candidate set does **not** flow into `GlossaryClusterDecision` /
`GlossaryClusterResponse` (built from LLM output, keyed by entity_id —
`schemas.py:217`), `merge_entities` (takes entities), `write_cluster_report` /
`merge_log` (built from merge results), `cluster_meta` / state (stores counts and
merge log), config, or the CLI.

Returning a `dict` keyed by the same pairs preserves the consumer's behavior:
- `if not candidates:` — empty dict is falsy. Unchanged.
- `sorted(candidates)` over a dict yields **sorted keys** = the pairs. Unchanged.
- The batch loop still iterates `(eid_a, eid_b)`. Unchanged.

Only the *new* scoring code reads `candidates[pair]` for evidence.

---

## Chosen Approach: Option (a) — Wrapper

Two options were considered:

- **(a) Wrapper (chosen):** leave the five heuristic functions returning
  `set[CandidatePair]` exactly as they are, so their existing per-heuristic unit
  tests do not change. Only `generate_cluster_candidates` changes: it calls each
  heuristic, tags the returned pairs with the heuristic name, and merges into an
  evidence dict. The one signal not available from a boolean heuristic — the JW
  *magnitude* — is captured by re-deriving JW **only for the pairs JW already
  flagged** (a bounded, cheap pass), or by a tiny targeted change to
  `_jw_similarity_candidates` (see below).
- **(b) Thorough:** rewrite all five functions to return evidence directly. More
  honest (JW magnitude captured at source, zero recompute anywhere) but rewrites
  the functions and their per-function tests.

We choose **(a)**: smallest diff, keeps the well-tested heuristics frozen, and is
fully sufficient to feed both auto-merge now and embeddings later.

Note: option (a) still re-derives JW for JW-flagged pairs. This is **not** the
drift-prone recomputation we are eliminating — we are not re-evaluating *whether*
a pair qualifies (the heuristic already decided that); we are only reading back a
magnitude for pairs that already passed. The set of pairs is authoritative; the
score is a lookup-only annotation.

---

## Data Shapes

### `Evidence`

A small dataclass recording what candidate generation discovered for one pair.

```python
from dataclasses import dataclass, field

# Heuristic name constants — single source of truth for tagging + scoring.
HEURISTIC_SOURCE_SUBSTRING = "source_substring"
HEURISTIC_TRANSLATION_CONTAINMENT = "translation_containment"
HEURISTIC_SHARED_READING = "shared_reading"
HEURISTIC_ALIAS_OVERLAP = "alias_overlap"
HEURISTIC_JW = "jaro_winkler"


@dataclass
class Evidence:
    """Why a candidate pair was generated, and how strong the signals were.

    Carried per pair from candidate generation through to confidence scoring.
    Never persisted; lives only for the duration of a clustering iteration.
    """

    heuristics: set[str] = field(default_factory=set)
    """Names of heuristics that flagged this pair (see HEURISTIC_* constants)."""

    jw_score: float | None = None
    """Best bi-directional Jaro-Winkler score observed for this pair, if the
    JW heuristic flagged it. None when JW did not fire."""

    same_category: bool = False
    """Whether the two entities share a category. A supporting signal, never
    sufficient alone."""

    # Direction of containment, when known. Useful for the auto-merge canonical
    # picker (the container/longer form is usually the more specific name) and
    # for future debugging/report detail. Optional; None when not a containment
    # match or direction is ambiguous (both directions held).
    source_contains: str | None = None
    """For source-substring matches: the entity_id whose source form CONTAINS
    the other's (the longer/more specific form). None if not applicable or
    bidirectional."""

    translation_contains: str | None = None
    """For translation-containment matches: the entity_id whose translation
    CONTAINS the other's. None if not applicable or bidirectional."""

    # Reserved for embeddings (NOT populated by this phase). Documented here so
    # the shape is stable when embeddings land.
    cosine: float | None = None
    """Cosine similarity from an embedding heuristic. Always None in this phase."""
```

### Candidate-generation return type

```python
Candidates = dict[CandidatePair, Evidence]
```

`generate_cluster_candidates` returns `Candidates`. Keys are the same canonically
ordered `(a, b)` pairs as today (via `_ordered_pair`, `glossary_clustering.py:37`).

### `ClusterConfidence`

```python
from enum import Enum


class ClusterConfidence(Enum):
    HIGH = "high"      # Auto-merge without LLM.
    MEDIUM = "medium"  # Send to LLM for confirmation (today's default path).
    LOW = "low"        # Reserved for embeddings: weak semantic-only pairs to
                       # auto-reject. Not produced in this phase.
```

`LOW` is defined but **not returned** by this phase's scorer. It exists so the tier
vocabulary is stable when embeddings introduce weak semantic-only candidates that
should be dropped before the LLM. This phase's scorer returns only `HIGH` or
`MEDIUM`.

---

## Candidate Generation Changes (`glossary_clustering.py`)

Only `generate_cluster_candidates` changes. The five `_*_candidates` functions keep
their current signatures and bodies (and their tests).

```python
def generate_cluster_candidates(
    glossary: Glossary,
    config: GlossaryClusterConfig,
) -> Candidates:
    """Run all heuristics and return candidate pairs WITH evidence.

    Each heuristic still returns a set of pairs; this function tags those pairs
    with the heuristic that produced them and accumulates them into per-pair
    Evidence. No heuristic logic is duplicated here — we only annotate.
    """
    evidence: dict[CandidatePair, Evidence] = defaultdict(Evidence)
    entity_by_id = {e.entity_id: e for e in glossary.entities}

    def _tag(pairs: set[CandidatePair], name: str) -> None:
        for pair in pairs:
            if pair[0] == pair[1]:
                continue
            evidence[pair].heuristics.add(name)

    _tag(_source_substring_candidates(glossary), HEURISTIC_SOURCE_SUBSTRING)
    _tag(_translation_containment_candidates(glossary), HEURISTIC_TRANSLATION_CONTAINMENT)
    _tag(_shared_reading_candidates(glossary), HEURISTIC_SHARED_READING)
    _tag(_alias_overlap_candidates(glossary), HEURISTIC_ALIAS_OVERLAP)
    _tag(_jw_similarity_candidates(glossary, threshold=config.jw_threshold), HEURISTIC_JW)

    # Annotate same_category, JW magnitude, and containment direction for the
    # pairs we already have. This is annotation of an authoritative pair set,
    # not re-qualification.
    for (id_a, id_b), ev in evidence.items():
        ea = entity_by_id.get(id_a)
        eb = entity_by_id.get(id_b)
        if ea is None or eb is None:
            continue
        ev.same_category = ea.category == eb.category
        if HEURISTIC_JW in ev.heuristics:
            ev.jw_score = _best_jw_score(ea, eb)
        if HEURISTIC_SOURCE_SUBSTRING in ev.heuristics:
            ev.source_contains = _containment_direction_source(ea, eb)
        if HEURISTIC_TRANSLATION_CONTAINMENT in ev.heuristics:
            ev.translation_contains = _containment_direction_translation(ea, eb)

    logger.debug("Cluster candidate generation produced %d pairs", len(evidence))
    return dict(evidence)
```

New private helpers (all read-only annotators, reusing existing predicates' rules):

- `_best_jw_score(ea, eb) -> float | None` — returns the max bi-directional
  `string_similarity` over source-form pairs and translation-form pairs (mirrors
  `_jw_any_match` at `glossary_clustering.py:197`, but returns the magnitude instead
  of a bool; reuses `_all_translation_forms`).
- `_containment_direction_source(ea, eb) -> str | None` — returns the entity_id of
  the side whose source form contains the other's; `None` if neither strictly
  contains or both directions hold (ambiguous). Reuses the same `len > 1` skip rule
  as `_has_source_substring_overlap` (`:60`).
- `_containment_direction_translation(ea, eb) -> str | None` — same idea over
  translation forms via `_all_translation_forms`, mirroring
  `_has_translation_containment` (`:105`).

Direction is recorded because it is cheap and the auto-merge canonical picker can
use "the container is usually the more specific name." It is advisory; the picker
falls back to length/first-seen when direction is `None`.

---

## Confidence Scoring (`glossary_clustering.py`)

```python
def score_candidate_confidence(evidence: Evidence) -> ClusterConfidence:
    """Tier a candidate pair from its recorded evidence.

    HIGH (auto-merge) requires multiple strong, agreeing signals AND same
    category. Everything else is MEDIUM (LLM confirmation). This phase never
    returns LOW.

    Reads evidence only — performs no entity comparison, no recomputation.
    """
    strong = {
        HEURISTIC_SOURCE_SUBSTRING,
        HEURISTIC_TRANSLATION_CONTAINMENT,
        HEURISTIC_SHARED_READING,
        HEURISTIC_ALIAS_OVERLAP,
    }
    strong_count = len(evidence.heuristics & strong)

    # A high-magnitude JW (>= 0.90) counts as a strong signal; the candidate-gen
    # threshold (0.75) is intentionally looser, so JW alone at the candidate
    # level is NOT strong.
    high_jw = evidence.jw_score is not None and evidence.jw_score >= 0.90
    if high_jw:
        strong_count += 1

    if strong_count >= 2 and evidence.same_category:
        return ClusterConfidence.HIGH

    return ClusterConfidence.MEDIUM
```

### Design rationale

**Why "2+ strong signals AND same category"?** A single strong signal can mislead:

- Source substring alone: `大罪` (sin) ⊂ `大罪司教` (Sin Archbishop) — related but
  different entities/types.
- Translation containment alone: "Ram" ⊂ "Rampage" — unrelated.
- JW alone: high scores on short strings can be coincidental.

Two agreeing strong signals plus same-category is much safer. A false auto-merge is
worse than an unnecessary LLM call, so the bar is deliberately conservative.

**Why JW >= 0.90 for auto-merge vs. 0.75 for candidacy?** Candidate generation casts
a wide net (the LLM filters the noise). Auto-merge skips the LLM, so it requires a
higher bar. This mirrors the build-stage philosophy where
`find_entity_for_mention` (`glossary.py:469`) only auto-attaches at JW >= 0.95.

---

## Auto-Merge Canonical Picker (`glossary_clustering.py`)

When merging without LLM input, pick winner/loser and canonical name
deterministically.

```python
def pick_canonical_for_auto_merge(
    ea: GlossaryEntity,
    eb: GlossaryEntity,
    evidence: Evidence,
) -> tuple[GlossaryEntity, GlossaryEntity, str]:
    """Choose (winner, loser, preferred_canonical_name) for a HIGH-confidence merge.

    Priority:
      1. Containment direction, if known — the container (more specific) name wins.
      2. Longer canonical_name (tends to be more specific).
      3. Earlier first_seen_chunk.
      4. Stable fallback (ea).
    """
    # 1. Prefer the container side when direction is known.
    container_id = evidence.translation_contains or evidence.source_contains
    if container_id == ea.entity_id:
        return ea, eb, ea.canonical_name
    if container_id == eb.entity_id:
        return eb, ea, eb.canonical_name

    # 2. Longer canonical name.
    if len(ea.canonical_name) > len(eb.canonical_name):
        return ea, eb, ea.canonical_name
    if len(eb.canonical_name) > len(ea.canonical_name):
        return eb, ea, eb.canonical_name

    # 3. Earlier first_seen_chunk.
    if ea.first_seen_chunk and eb.first_seen_chunk:
        if ea.first_seen_chunk <= eb.first_seen_chunk:
            return ea, eb, ea.canonical_name
        return eb, ea, eb.canonical_name

    # 4. Stable fallback.
    return ea, eb, ea.canonical_name
```

Field names are `canonical_name` / `surface_forms[].translation` (current schema),
not the older `canonical_english` / `english` naming from earlier drafts.

---

## Routing in the Clustering Loop (`glossary.py`)

Insert between `candidate_list = sorted(candidates)` (`glossary.py:1742`) and the
batch construction (`:1745`). The existing per-iteration `id_map`, remap helpers
(`_resolve_id`, `remap_entity_id`), `merge_entities`, and per-batch save logic are
reused unchanged.

```python
candidate_list = sorted(candidates)  # sorted keys of the evidence dict

# Partition by confidence (skip when disabled).
high_pairs: list[CandidatePair] = []
llm_pairs: list[CandidatePair] = []
if cluster_config.auto_merge_enabled:
    for pair in candidate_list:
        if score_candidate_confidence(candidates[pair]) == ClusterConfidence.HIGH:
            high_pairs.append(pair)
        else:
            llm_pairs.append(pair)
else:
    llm_pairs = list(candidate_list)

logger.info(
    "Cluster %s: %d high-confidence auto-merges, %d pairs for LLM review",
    item_id, len(high_pairs), len(llm_pairs),
)

# --- Auto-merge phase (no LLM) ---
for eid_a, eid_b in high_pairs:
    resolved_a = _resolve_id(eid_a, id_map)
    resolved_b = _resolve_id(eid_b, id_map)
    if resolved_a == resolved_b:
        continue  # already merged via an earlier auto-merge this iteration
    ea = _entity_by_id(resolved_a)
    eb = _entity_by_id(resolved_b)
    if ea is None or eb is None:
        continue

    winner, loser, pref_name = pick_canonical_for_auto_merge(ea, eb, candidates[(eid_a, eid_b)])

    existing_sources = {sf.source for sf in winner.surface_forms}
    new_sf_labels = [
        f"`{sf.source}` -> {sf.translation}"
        for sf in loser.surface_forms
        if sf.source not in existing_sources
    ]

    merge_entities(winner, loser, pref_name)
    glossary.entities.remove(loser)
    id_map[loser.entity_id] = winner.entity_id
    _remap_build_meta_conflicts(build_meta, loser.entity_id, winner.entity_id)

    iteration_merge_entries.append({
        "winner_id": winner.entity_id,
        "loser_id": loser.entity_id,
        "winner_name": winner.canonical_name,
        "loser_name": loser.canonical_name if loser.canonical_name != winner.canonical_name
                      else pref_name or loser.canonical_name,
        "result_name": winner.canonical_name,
        "reasoning": "HIGH CONFIDENCE AUTO-MERGE (multiple heuristics agreed)",
        "auto_merged": True,
        "surface_forms_added": new_sf_labels,
    })
    merges_this_iteration += 1

if high_pairs:
    _save_glossary(work_dir, glossary, glossary_cluster_path(work_dir))
    _save_build_meta(work_dir, build_meta)

# --- LLM phase ---
# Build batches from llm_pairs instead of candidate_list. The existing batch loop
# already remaps each pair through id_map before building prompts
# (glossary.py:1759-1762), so pairs whose entity was just auto-merged are resolved
# or dropped as self-merges automatically.
```

### Critical invariant (must have a dedicated test)

`high_pairs` and `llm_pairs` are partitioned **before** auto-merges run. An
auto-merge can absorb an entity that also appears in an `llm_pairs` entry. This is
handled because:

1. Auto-merge populates the **same per-iteration `id_map`** the LLM loop already
   consults.
2. The LLM batch loop remaps every pair through `id_map` and drops self-merges
   before building prompts (existing logic, `glossary.py:1759-1762`, `:1810-1815`).

The dict key used for `candidates[(eid_a, eid_b)]` is the **original** pre-remap
pair (evidence was recorded against original IDs), while merge resolution uses the
**remapped** IDs. Keep these straight: look up evidence by original key, act on
resolved entities.

Notable detail: the merge-entry keys are `winner_name` / `loser_name` /
`result_name` (matching `write_cluster_report` at `glossary_clustering.py:527-530`)
— **not** the `*_english` keys from older drafts, which would raise `KeyError` in the
report renderer.

---

## Report Changes (`glossary_clustering.py :: write_cluster_report`)

`merge_log` entries gain an `auto_merged: bool` field. LLM-confirmed merges set it
`False` (or omit it; the renderer treats missing as `False`).

Add summary counts to the report header. This requires extending
`write_cluster_report`'s signature (currently
`(report_path, merge_log, total_iterations, total_candidates_evaluated)`):

```markdown
# Glossary Clustering Report

- Iterations: 2
- Total candidate pairs evaluated: 31
- Auto-merges (high confidence): 12
- LLM-confirmed merges: 3
- Merges performed: 15
```

Per-merge sections add a tag line, e.g. `- **Type:** auto-merge` vs.
`- **Type:** LLM-confirmed`, so every decision remains auditable.

Counts derive from `merge_log` (`sum(1 for e in merge_log if e.get("auto_merged"))`),
so no new accumulator threading is strictly required beyond passing the existing
`merge_log` — but if the header line is desired, the function reads it from the log.

---

## Config (`config.py :: GlossaryClusterConfig`)

Add one field:

```python
class GlossaryClusterConfig(BaseModel):
    max_iterations: int = 3
    jw_threshold: float = 0.75
    batch_size: int = 10
    auto_merge_enabled: bool = False
    """When False, every candidate goes to the LLM regardless of confidence
    (useful for auditing all merge decisions or debugging the scorer)."""
```

Default `False`. Consider running one real book with `auto_merge_enabled: false`
first to eyeball which pairs the scorer would have auto-merged before trusting it in
bulk (consistent with the project's "measure first" stance on this optimization).

---

## Embeddings Readiness (out of scope here, prepared for)

This phase deliberately does the structural work embeddings will also require, so
embeddings become an additive change rather than another refactor:

- **Evidence carries scores, not just booleans.** `Evidence.cosine` is reserved.
  An embedding heuristic will populate it; nothing else changes shape.
- **Candidate generation already returns per-pair evidence.** Adding embeddings is
  one more tagging call plus annotating `ev.cosine` — same pattern as the existing
  five heuristics.
- **The `LOW` tier exists.** Embeddings produce loose semantic-only candidates; the
  natural use of `LOW` is to auto-reject weak embedding-only pairs (e.g. cosine
  below a floor, no corroborating string heuristic) before they reach the LLM.
  `score_candidate_confidence` will gain that branch when embeddings land.
- **Scoring reads evidence only.** Extending it to weigh `cosine` (and to treat
  "embedding + a string heuristic agree" as strong) is a local edit.

No embedding model, dependency, config, or heuristic is added in this phase.

---

## Implementation Order

1. Add `Evidence` dataclass, `HEURISTIC_*` constants, `Candidates` type alias, and
   `ClusterConfidence` enum to `glossary_clustering.py`.
2. Add read-only annotators: `_best_jw_score`, `_containment_direction_source`,
   `_containment_direction_translation`.
3. Rewrite `generate_cluster_candidates` to return `Candidates` (tag + annotate).
4. Add `score_candidate_confidence` and `pick_canonical_for_auto_merge`.
5. Add `auto_merge_enabled` to `GlossaryClusterConfig`.
6. Wire partition + auto-merge phase into `glossary_cluster` (`glossary.py`),
   feeding `llm_pairs` to the existing batch loop.
7. Add `auto_merged` to merge-log entries; update `write_cluster_report` (tag lines
   + summary counts).
8. Tests (below).

---

## Tests (`tests/test_glossary_clustering.py`)

### Candidate evidence
- `generate_cluster_candidates` returns a dict keyed by ordered pairs (existing
  membership/`len`/`sorted` assertions still hold over dict keys).
- A pair flagged by two heuristics records both names in `evidence.heuristics`.
- `jw_score` is populated (and `>= jw_threshold`) only when the JW heuristic fired;
  `None` otherwise.
- `same_category` reflects the two entities' categories.
- `source_contains` / `translation_contains` record the container entity_id for
  containment matches; `None` when not a containment match or bidirectional.

### Confidence scoring
- Source substring + translation containment + same category -> HIGH.
- Shared reading + alias overlap + same category -> HIGH.
- JW >= 0.90 + translation containment + same category -> HIGH.
- Single strong signal + same category -> MEDIUM.
- Two strong signals + different category -> MEDIUM (missing same_category).
- JW only at 0.78 (below 0.90) + same category -> MEDIUM.
- Scorer never returns LOW in this phase.

### Canonical picker
- Containment direction wins: container/longer name is preferred.
- No direction: longer `canonical_name` wins ("Vincent Volakia" over "Vincent").
- Equal length: earlier `first_seen_chunk` wins.
- Null `first_seen_chunk`: stable fallback to `ea`.

### Routing / integration (mocked LLM in `glossary_cluster`)
- HIGH pairs merge with no LLM call; LLM is invoked only for MEDIUM pairs.
- Auto-merged entities appear in the report tagged as auto-merge with the fixed
  reasoning string; counts line is correct.
- `auto_merge_enabled=False` routes every candidate to the LLM (zero auto-merges).
- **Invariant test:** an entity in a HIGH pair is also present in a MEDIUM pair;
  after auto-merge, the MEDIUM pair is correctly remapped through `id_map` (resolved
  or dropped as self-merge) and no merge references a removed entity.
- Evidence is looked up by the original pre-remap pair key while merges act on
  resolved IDs (no `KeyError`, no stale-entity merge).

### Report
- `merge_log` entries carry `auto_merged`; renderer shows per-merge type and a
  summary count split (auto vs. LLM-confirmed).
- Missing `auto_merged` key is treated as LLM-confirmed (backward-safe).

---

## Explicit Non-Goals

- No embedding heuristic, model, dependency, or config (only the reserved
  `Evidence.cosine` field and the `LOW` enum value).
- No change to the five heuristic functions' signatures or behavior (option (a)).
- No change to build-time linking (`find_entity_for_mention`).
- No change to reconcile, export, or translation injection.
- No persisted/serialized representation of `Evidence` — it is in-memory, per
  iteration.
- No rewrite of `merge_entities` or the remap/`id_map` machinery (reused as-is).
- **Auto-merge ON by default is now explicitly deferred.** This phase ships the
  machinery with `auto_merge_enabled` defaulting to **False** (see the addendum
  below). Trusting HIGH-confidence auto-merge in production is a goal of the
  embeddings phase, not this one.

---

## Addendum: Implementation Findings & Decision to Default OFF

**Status:** Implemented. `auto_merge_enabled` defaults to **False**.

The evidence/scoring/auto-merge machinery described above was implemented as
specified (option (a) wrapper, `Evidence`, `score_candidate_confidence`,
`pick_canonical_for_auto_merge`, report tagging, config flag). It is correct and
fully tested. What changed from the original spec is only the **default and the
trust level of the HIGH tier**.

### Live-run findings (perfect-world-cn)

A real clustering run (118 build entities, LM Studio / `qwen3.6-35b-a3b-mtp`,
`auto_merge_enabled=True`) produced **4 auto-merges, 2 of which were wrong**:

| Auto-merge | Verdict | Why |
|---|---|---|
| `Dark Huo Ling'er` <- `Huo Ling'er` | correct | aspect/counterpart of same character |
| `Nine Heavens and Ten Earths` <- `Nine Heavens` | correct | containment of same place |
| **`Quasi-Immortal Emperor` (准仙帝) <- `Immortal Emperor` (仙帝)** | **WRONG** | their own summaries say these are *adjacent but distinct* cultivation realms ("a realm just below Immortal Emperor") |
| **`Da Zhuang` (大壮) <- `Huang` (荒)** | **WRONG** | distinct characters; merged on a coincidental JW≈0.91 + spurious containment. End identity was only salvaged because the LLM later folded `Da Zhuang` into `Shi Hao` |

### Root cause

The string-only scorer cannot distinguish two cases that emit **identical
evidence** — source/translation containment + high Jaro-Winkler + same category:

- *qualifier denotes the SAME entity* (`Dark` X is an aspect of X), vs.
- *qualifier denotes a DISTINCT rank/thing* (`Quasi-` X is a different realm
  than X).

This is precisely the false-merge outcome the original design's "a false
auto-merge is worse than an unnecessary LLM call" rationale meant to avoid; the
threshold as specified does not achieve it on real data.

### Decision

- **Default `auto_merge_enabled=False`.** Everything routes to the LLM, i.e.
  identical observable behavior to before this phase. The machinery is dormant
  until explicitly opted in.
- **Do NOT harden the string-only scorer now.** Two rejected options:
  - *Qualifier blocklist* (treat 准/Quasi-/Dark- prefixes as MEDIUM): rejected —
    language-specific and fragile, contradicts the language-agnostic design.
  - *Require a non-containment strong signal* (shared_reading / alias_overlap):
    rejected — readings are usually null on CN/JA glossary entries, so this
    rarely fires and would gut the feature.
- **Defer real hardening to the embeddings phase.** Embedding agreement becomes
  the corroborating "strong signal" that makes HIGH safe: `Quasi-Immortal
  Emperor` vs `Immortal Emperor` are *near* in embedding space but their
  summaries distinguish the adjacent realms; `Da Zhuang` / `Huang` are *far*
  apart, overriding the spurious JW. `score_candidate_confidence` will weigh
  `Evidence.cosine` then, and `auto_merge_enabled=True` becomes recommendable.

### Test baseline for the embeddings phase

A regression test encodes **today's known-unsafe behavior**: a "containment + JW
+ same category" pair scores **HIGH** right now. That test is annotated so the
embeddings phase can flip it to MEDIUM as a deliberate, visible change rather
than a silent regression.
