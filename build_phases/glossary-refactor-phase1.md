# dao-bridge-translator: Phase 1 Entity-Centric Glossary — Consolidated Spec

## Mental Model

```text
Extract:    What notable source terms did this chunk mention?
Link:       Does this mention obviously belong to an existing entity?
Cluster:    Looking across the accumulated glossary, did we accidentally create duplicate entities?
Reconcile:  Clean up translation/name/speech-style conflicts inside entities.
Translate:  Which entities are relevant to this chunk based on any of their surface forms?
```

---

## Purpose

Refactor the glossary from a `source_term`-keyed flat list into an entity-centric glossary where each entity owns a pool of surface forms.

The current system treats each Japanese string as the identity key. That breaks when one entity is referred to by multiple forms:

- `アベル` (Abel)
- `アベルちゃん` (Abel-chan)
- `ヴィンセント・ヴォラキア` (Vincent Volakia)
- `ヴィンセント・ヴォラキア皇帝` (Emperor Vincent Volakia)

These are all the same character, but the current system stores them as unrelated glossary entries. This causes:

1. **Broken reconciliation**: conflicts are only detected when two batches produce different English proposals for the same `source_term`. Entries with different source_terms that refer to the same entity are never connected.
2. **Broken translation-time injection**: glossary entries are injected by scanning chunk text for `source_term` matches. If the chunk uses a variant form, the entity's glossary entry is missed.
3. **Junk entries**: pronouns (`オレ`, `貴様`), generic honorifics, and common nouns are extracted as glossary entries when they shouldn't be.

---

## Core Concepts

### ExtractedMention

A temporary raw observation returned by the extraction LLM. It means:

> "The model saw this term in this chunk and thinks it may be notable."

It is not stored directly in the final glossary. Build code decides whether each mention attaches to an existing entity or creates a new one.

### SurfaceForm

A source-language form attached to a known entity. It means:

> "This Japanese string is one way the text refers to this entity, and when this specific form appears, it usually renders as this specific English form."

Important: each surface form carries its own English rendering. `アベル → Abel` and `ヴィンセント・ヴォラキア皇帝 → Emperor Vincent Volakia` are both surface forms on the same entity, but they should not always translate the same way.

### GlossaryEntity

The canonical glossary object. It means:

> "This is the person/place/item/concept. It may have many source-language surface forms, one canonical English name, and accumulated context."

---

## Data Shapes

### New schema types (`schemas.py`)

```python
class SurfaceForm(BaseModel):
    """A source-language text form that refers to an entity."""

    source: str                          # The Japanese string, e.g. "アベル"
    reading: str | None = None           # From furigana, if available
    english: str                         # English rendering for THIS specific form
    context_hints: list[str] = Field(default_factory=list)  # Low-confidence contextual hints from extraction
    notes: str | None = None             # Usage notes
    first_seen_chunk: str | None = None
    occurrence_count: int = 1


class GlossaryEntity(BaseModel):
    """A single entity in the glossary."""

    entity_id: str                       # Stable identifier (generated slug or UUID)
    category: str                        # Validated against config.glossary.categories
    canonical_english: str               # The primary English name, e.g. "Abel"
    summary: str | None = None           # Accumulated understanding of this entity
    surface_forms: list[SurfaceForm] = Field(default_factory=list)

    # Carried from existing schema
    aliases: list[str] = Field(default_factory=list)
    nicknames: dict[str, str] = Field(default_factory=dict)
    speech_style: str | None = None
    notes: str | None = None
    source: GlossarySource = "extracted"
    source_books: list[str] = Field(default_factory=list)

    # Temporal tracking
    first_seen_chunk: str | None = None
    latest_evidence_chunk: str | None = None


class Glossary(BaseModel):
    entities: list[GlossaryEntity] = Field(default_factory=list)
    version: int = 2
    book_id: str | None = None
    book_metadata: dict = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### Extraction response models

```python
class ExtractedMention(BaseModel):
    """A raw mention observed in a chunk/batch."""

    source: str                          # Exact source-language term as written
    reading: str | None = None           # Pronunciation from furigana, else null
    english: str                         # Proposed English rendering for this form
    category: str                        # Allowed category
    summary_update: str | None = None    # One concise sentence about what this entity appears to be
    context_hint: str | None = None      # Low-confidence hint, e.g. "same person as アベル"
    notes: str | None = None
    aliases: list[str] = Field(default_factory=list)
    nicknames: dict[str, str] = Field(default_factory=dict)
    speech_style: str | None = None


class GlossaryExtractionResponse(BaseModel):
    mentions: list[ExtractedMention] = Field(default_factory=list)
    corrections: list[GlossaryCorrectionEntry] = Field(default_factory=list)
```

### Field naming conventions

New code should prefer:

```text
entries -> entities
source_term -> surface_form.source
english -> canonical_english (entity level) or surface_form.english (form level)
```

---

## How ExtractedMention Becomes Entity/SurfaceForm

The extraction LLM returns `ExtractedMention` objects. Build code handles all linking.

Field mapping:

```text
ExtractedMention.source         -> SurfaceForm.source
ExtractedMention.reading        -> SurfaceForm.reading
ExtractedMention.english        -> SurfaceForm.english
ExtractedMention.context_hint   -> SurfaceForm.context_hints (appended)
ExtractedMention.notes          -> SurfaceForm.notes
ExtractedMention.category       -> GlossaryEntity.category
ExtractedMention.summary_update -> GlossaryEntity.summary (merge input)
ExtractedMention.english        -> GlossaryEntity.canonical_english (new entity only)
```

Pseudo-code:

```python
def merge_mention(glossary: Glossary, mention: ExtractedMention, chunk_id: str) -> None:
    entity = find_entity_for_mention(glossary, mention)

    if entity is None:
        sf = SurfaceForm(
            source=mention.source,
            reading=mention.reading,
            english=mention.english,
            context_hints=[mention.context_hint] if mention.context_hint else [],
            notes=mention.notes,
            first_seen_chunk=chunk_id,
            occurrence_count=1,
        )
        entity = GlossaryEntity(
            entity_id=next_entity_id(mention.category, glossary),
            category=mention.category,
            canonical_english=mention.english,
            summary=mention.summary_update,
            surface_forms=[sf],
            aliases=list(mention.aliases),
            nicknames=dict(mention.nicknames),
            speech_style=mention.speech_style,
            notes=mention.notes,
            first_seen_chunk=chunk_id,
            latest_evidence_chunk=chunk_id,
        )
        glossary.entities.append(entity)
        return

    add_or_update_surface_form(entity, mention, chunk_id)
    merge_entity_summary(entity, mention.summary_update, chunk_id)
    merge_aliases_nicknames_speech_notes(entity, mention)
```

---

## Build-Time Incremental Linking

Incremental linking happens during glossary build, after extraction. Its purpose is to avoid creating obvious duplicates while processing chunk batches.

It is separate from translation-time injection and separate from the later clustering stage.

### Candidate retrieval order

For each extracted mention, try to find an existing entity using cheap checks first:

1. Exact surface-form source match (highest confidence)
2. Same non-null reading AND same proposed English
3. Source substring containment (mention.source is in an existing surface form, or vice versa)
4. English containment (mention.english contains or is contained by an existing entity's canonical_english or surface form english)
5. Jaro-Winkler similarity on source and/or English (see String Similarity Helper below)
6. Same category bonus (not sufficient alone, but boosts confidence of other matches)

### Safe auto-attach rules

Auto-attach ONLY when the match is unambiguous:

- `アベル` appears again and an entity already has surface form `アベル` → attach
- Same non-null reading AND same proposed English → attach
- Exact source form exists in an entity's surface_forms list → attach

For weaker signals (substring containment, Jaro-Winkler > 0.75 but < 0.95), **create a new entity** and let clustering merge it later.

### Do not over-link during build

Do not automatically merge:

```text
アベル + ヴィンセント・ヴォラキア皇帝
```

unless the source text or accumulated summary makes the connection explicit. Hidden identities, aliases without clear textual connection, and title-vs-name associations should be handled by the clustering stage.

### Important: do not require the extraction LLM to return entity_id

The extraction LLM just reports mentions. Build code is the authority on attachment. This keeps the extraction prompt simple and the LLM's job focused on "what terms did this chunk contain?"

---

## String Similarity Helper

Add a utility based on the bi-directional Jaro-Winkler approach from the reference repository (`context-aware-translation`).

```python
import jellyfish

def string_similarity(a: str, b: str) -> float:
    """
    Compute string similarity using bi-directional Jaro-Winkler.

    Forward comparison handles prefix matches (standard Jaro-Winkler behavior).
    Reversed comparison handles suffix matches by converting them to prefix matches.
    Returns the maximum of both to catch both patterns.
    """
    if not a or not b:
        return 0.0

    a = a.lower().strip()
    b = b.lower().strip()

    if a == b:
        return 1.0

    forward = jellyfish.jaro_winkler_similarity(a, b)
    reverse = jellyfish.jaro_winkler_similarity(a[::-1], b[::-1])
    return max(forward, reverse)
```

Suggested thresholds:

```text
>= 0.95   Very strong candidate; auto-attach if category and other evidence agree
0.75-0.95 Candidate for clustering/LLM review
< 0.75    Ignore unless another signal matches
```

Use this for candidate generation, never as a blind merge authority.

---

## Summary Field

Each entity carries a `summary: str | None` field that accumulates understanding of the entity across chunks.

### Summary from extraction

Each `ExtractedMention` may include a `summary_update`: one concise sentence about what this entity appears to be in this passage.

### Summary merge during build

Simple initial approach:

```python
def merge_entity_summary(entity: GlossaryEntity, summary_update: str | None, chunk_id: str) -> None:
    if not summary_update:
        return
    entity.latest_evidence_chunk = chunk_id
    if not entity.summary:
        entity.summary = summary_update
    elif summary_update not in entity.summary:
        entity.summary = entity.summary + " " + summary_update
```

Do not let summaries grow unbounded. Add a max length and truncate if needed.

### Future enhancement: LLM summary compressor

Can later be replaced by an LLM call that takes `old_summary + new_observation → concise_updated_summary`, following the pattern from the reference repository's `LLMTermMemoryUpdater`. That system uses a two-phase approach:

1. **Bootstrap**: takes initial observations and produces a first summary
2. **Incremental update**: asks the LLM "does this new evidence change the summary?" Returns unchanged or updated summary

The reference repository also versions summaries with `effective_start_chunk` and `latest_evidence_chunk` for temporal awareness (no future knowledge leakage during translation). This is a Phase 2 enhancement.

### Summary's role in clustering

The summary is key input for embedding-based candidate generation (see Clustering section). An enriched embedding string like:

```text
"character. Abel. アベル. Abel. Masked man claiming to be the emperor, traveling with Subaru"
```

carries enough semantic signal for the embedding model to connect it with:

```text
"character. Vincent Volakia. ヴィンセント・ヴォラキア. Vincent Volakia. The legitimate Emperor of Volakia"
```

even though the Japanese strings share zero characters.

---

## Context Hints

`SurfaceForm.context_hints: list[str]` provides a lightweight channel for the extraction LLM to pass soft signals downstream without making hard entity-linking decisions.

### How hints flow through the pipeline

1. **Extraction**: LLM sees `ヴィンセント` in a chunk where Abel removes his mask. Returns `ExtractedMention(source="ヴィンセント", english="Vincent", context_hint="same person as アベル based on unmasking scene")`
2. **Build**: no exact surface form match, creates new entity. The hint is stored as `SurfaceForm.context_hints = ["same person as アベル based on unmasking scene"]`
3. **Clustering**: candidate pair generated via English containment or embedding similarity. The LLM confirmation step sees the context hint — strong evidence to confirm the merge

### Rules

- Build code NEVER auto-merges based on a context hint alone
- Hints are advisory evidence consumed by clustering and human review
- Most surface forms will have an empty hints list — hints only accumulate on ambiguous cases
- When surface forms merge during clustering, their hints union naturally

---

## Extraction Prompt Changes

Update `glossary_extract.txt` so extraction returns raw mentions.

### The prompt should ask for:

- `source`: exact source-language term as written in the text
- `reading`: pronunciation from furigana if available, else null
- `english`: proposed English rendering for THIS specific surface form
- `category`: one of the allowed categories
- `summary_update`: one concise sentence about what this mention/entity appears to be in this passage
- `context_hint`: optional low-confidence hint about entity identity (e.g. "appears to be the same person as アベル")
- `notes`: brief usage notes, or null
- `aliases`, `nicknames`, `speech_style`: only if clearly available in this passage

### Negative extraction rules (add to prompt):

```text
DO NOT extract:
- Standalone pronouns (オレ, 俺, 私, 貴様, お前) unless used as a unique title/name
- Generic honorific suffixes (さん, 様, ちゃん, 君) by themselves
- Common nouns/descriptions unless used as a stable named identifier
- Speech tics or verbal habits as standalone glossary terms
- Generic titles alone (皇帝, 王, 兵士) unless the text uses them as a unique persistent identifier for a specific character
- First-person/second-person pronoun usage patterns (these are speech-style observations, not glossary entities)
```

### Important separation of concerns

Extraction answers: "What notable source terms did this chunk contain?"

Extraction does NOT answer: "What is the final identity graph of this book?"

### Optional extraction-time context

Do not inject the full glossary into extraction prompts. An optional compact entity list may be injected for reference:

```text
Known nearby entities:
- character_000001: Abel; surface forms: アベル; summary: composed traveler
- character_000002: Rem; surface forms: レム
```

This is advisory only. The extraction LLM is not required to use it and should not return entity_ids.

---

## New Clustering Stage

Add between build and reconcile:

```text
glossary_build -> glossary_cluster -> glossary_reconcile -> glossary_export
```

Purpose: find duplicate entities that build-time linking did not safely merge.

### Candidate generation

Generate unordered entity pairs using heuristics. Each heuristic catches a different class of duplicate:

**String-level heuristics (cheap, deterministic):**

1. **Japanese substring containment**: any surface form of entity A is a substring of any surface form of entity B, or vice versa. Catches: `アベル` ↔ `アベルちゃん`, `ヴィンセント・ヴォラキア` ↔ `ヴィンセント・ヴォラキア皇帝`
2. **English containment**: canonical_english or any surface_form.english of A contains or is contained by B's. Catches: `Abel` ↔ `Abel-chan`, `Vincent Volakia` ↔ `Emperor Vincent Volakia`
3. **Shared non-null reading**: two entities share the same reading field on any surface form
4. **Alias overlap**: an alias of entity A matches a surface form source or alias of entity B
5. **Jaro-Winkler similarity**: bi-directional Jaro-Winkler on source forms and/or English names above threshold (0.75). Catches: honorific variants, minor romanization differences. Use on English names to catch LLM extraction inconsistencies like `Petelgeuse` vs `Petelgeous`
6. **Same or compatible category**: not sufficient alone, but boosts confidence of other heuristic matches

**Semantic heuristic (optional, future enhancement):**

7. **Embedding similarity**: embed enriched entity descriptions using a lightweight multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`, ~420MB, CPU-friendly). This catches semantic connections that no string-level heuristic can reach — hidden identities, title-vs-name, unrelated surface forms referring to the same entity via accumulated summary context.

Entity text for embedding:

```python
def entity_embedding_text(entity: GlossaryEntity) -> str:
    forms = ", ".join(sf.source for sf in entity.surface_forms)
    english = ", ".join(sf.english for sf in entity.surface_forms)
    hints = " ".join(h for sf in entity.surface_forms for h in sf.context_hints)
    parts = [entity.category, entity.canonical_english, forms, english, entity.summary or "", hints]
    return ". ".join(p for p in parts if p)
```

Cosine similarity between all entity pairs is trivially fast for glossary-sized collections (hundreds to low thousands of entities). No FAISS or vector DB needed — just `sklearn.metrics.pairwise.cosine_similarity` on the embedding matrix.

Embeddings should be configurable (on/off in glossary config) so the pipeline works without the model dependency.

### LLM confirmation

Send candidate pairs to the LLM in batches. For each pair, provide:

- entity_id, category, canonical_english
- summary
- all surface_forms with their English renderings
- context_hints from surface forms (if any)
- notes, speech_style if present

Ask: "Are these the same entity? If yes, which canonical English name should be kept?"

Response model:

```python
class GlossaryClusterDecision(BaseModel):
    entity_id_a: str
    entity_id_b: str
    same_entity: bool
    preferred_entity_id: str | None = None
    preferred_canonical_english: str | None = None
    reasoning: str


class GlossaryClusterResponse(BaseModel):
    decisions: list[GlossaryClusterDecision] = Field(default_factory=list)
```

### Merge behavior

If confirmed same entity:

- Keep preferred entity_id (or earlier first_seen entity if no preference)
- Union surface_forms, deduplicating by normalized `source` — if two forms share the same source string but differ in English rendering, keep one form and resolve the English conflict (prefer first-seen or flag for reconcile). Do not keep duplicate surface forms just because the English differs
- Choose preferred canonical_english from LLM decision
- Move non-canonical English renderings to their respective surface forms (they stay as surface_form.english)
- Union aliases
- Merge nicknames (existing value wins on key conflict)
- Accumulate speech_style observations using current delimiter pattern
- Merge notes conservatively
- Merge summaries conservatively (concatenate, or use LLM compressor if available)
- Union context_hints across merged surface forms
- Keep earliest first_seen_chunk
- Keep latest latest_evidence_chunk
- Remove the losing entity from the glossary

### Iterative clustering

After merges, run candidate generation again. Reason: merging A+B may create new heuristic matches to entity C (e.g., B had a surface form that is a substring of C's, but A didn't).

Cap iterations:

```text
max_cluster_iterations = 3
```

Stop early if no new candidates are generated.

Write a clustering report similar to the existing reconcile report.

### Pipeline state integration

New stage `glossary_cluster` with item tracking per candidate pair batch, following the existing `PipelineState` pattern.

---

## Reconcile Stage Changes

Reconcile should operate on entities instead of source terms.

For Phase 1, keep this simple:

- Consolidate accumulated speech_style observations within each entity
- Optionally clean up canonical English conflicts within one entity
- Produce a reconcile report

Do not rebuild the whole reconciliation system yet.

---

## Translation-Time Injection

This is the immediate quality payoff.

### Current behavior

Scans chunk text for one `source_term` per glossary entry. Misses variant forms.

### New behavior

Scan all surface forms for each entity:

```python
def find_relevant_entities(chunk_text: str, glossary: Glossary) -> list[GlossaryEntity]:
    relevant = []
    for entity in glossary.entities:
        for sf in entity.surface_forms:
            if sf.source and sf.source in chunk_text:
                relevant.append(entity)
                break  # One match is enough; don't duplicate
    return relevant
```

### Prompt rendering

When rendering glossary context for translation, include:

- canonical English name
- category
- summary (if present)
- all surface forms with their per-form English renderings
- speech style (if character and present)
- notes (if present)

Example:

```text
[character] Vincent Volakia
Summary: Emperor of Volakia; sometimes travels under the name Abel.
Surface forms:
- アベル -> Abel
- ヴィンセント・ヴォラキア -> Vincent Volakia
- ヴィンセント・ヴォラキア皇帝 -> Emperor Vincent Volakia
Speech style: calm, authoritative, controlled.
```

**Important**: do not normalize every surface form to canonical English during translation. When the source text says `アベル`, the translation should say "Abel," not "Vincent Volakia." Each surface form has its own English rendering for a reason.

---

## Export Changes

Update markdown export to group by category and display entities:

```markdown
## Character

### Vincent Volakia (`character_000012`)

Summary: Emperor of Volakia; sometimes travels under the name Abel.

Surface forms:
- `アベル` -> Abel
- `ヴィンセント・ヴォラキア` -> Vincent Volakia
- `ヴィンセント・ヴォラキア皇帝` -> Emperor Vincent Volakia

Speech style: calm, authoritative, controlled.
Notes: ...
```

---

## Migration

Migration from old glossary format is optional but straightforward if needed:

```text
old GlossaryEntry.source_term -> SurfaceForm.source (sole surface form)
old GlossaryEntry.reading     -> SurfaceForm.reading
old GlossaryEntry.english     -> SurfaceForm.english
old GlossaryEntry.english     -> GlossaryEntity.canonical_english
old GlossaryEntry.category    -> GlossaryEntity.category
old GlossaryEntry.notes       -> GlossaryEntity.notes / summary
```

Generate `entity_id` from a slug of the English name or a UUID. Do not spend significant effort on migration — old glossaries can be regenerated.

---

## Implementation Order

Recommended order, designed for two shippable PRs:

### PR 1: Schema + Surface Form Scanning (immediate quality improvement)

1. Add new schema types: `SurfaceForm`, `GlossaryEntity`, `ExtractedMention`
2. Update glossary load/save/export paths to use `glossary.entities`
3. Update extraction prompt and response model to return `mentions` (with negative extraction rules and `summary_update`/`context_hint` fields)
4. Update build merge logic:
   - exact surface-form match → attach to existing entity
   - otherwise → create new entity
   - summary_update gets stored/merged
   - context_hint gets appended to surface form
5. Update translation-time injection to scan all surface forms
6. Add string similarity helper (`string_similarity` function)
7. Add tests for schema, build merge, translation injection

### PR 2: Clustering + Reconcile Adaptation

8. Add candidate generation heuristics (substring, English containment, reading, alias, Jaro-Winkler)
9. Add `glossary_cluster` stage with LLM confirmation and iterative merge
10. Adapt reconcile to entity schema
11. Update export to new format
12. Add tests for similarity, clustering, merge behavior

### Future: Enhancements

- Embedding-based candidate generation (configurable, requires `sentence-transformers` dependency)
- LLM summary compressor (bootstrap + incremental update pattern from reference repository)
- Versioned summaries with temporal awareness (no future knowledge leakage)
- Master glossary adaptation to entity-centric format

---

## Tests

### Schema

- `GlossaryEntity` serializes/deserializes correctly
- `SurfaceForm` stores source/reading/english/context_hints
- `ExtractedMention` response validates
- Migration from old format produces valid new format (if migration implemented)

### Build merge

- Repeated exact source form attaches to existing entity
- New source form creates new entity
- Same surface form occurrence_count increments
- summary_update stored on new entity
- summary_update merged on existing entity
- context_hint appended to surface form
- Aliases, nicknames, speech_style merged correctly

### Translation injection

- Entity is selected if ANY surface form appears in chunk text
- Entity is NOT duplicated if multiple surface forms appear in same chunk
- Entity is NOT selected if no surface form appears in chunk text
- Rendering includes canonical English and per-form surface form English renderings

### String similarity

- Forward Jaro-Winkler catches prefix matches
- Reversed Jaro-Winkler catches suffix matches (e.g. honorific additions)
- Exact match returns 1.0
- Empty strings return 0.0
- Threshold filtering works correctly

### Clustering

- Substring candidate: `アベル` ↔ `アベルちゃん`
- English containment: `Vincent` ↔ `Emperor Vincent Volakia`
- Shared reading candidate
- Alias overlap candidate
- Jaro-Winkler candidate above threshold
- Merge unions surface forms
- Merge preserves per-surface English renderings
- Merge keeps preferred canonical English from LLM decision
- Merge unions context_hints
- Merge keeps earliest first_seen_chunk and latest latest_evidence_chunk
- Iterative clustering catches transitive merges (A+B reveals match to C)
- Iteration cap prevents infinite loops

---

## Explicit Non-Goals for Phase 1

Do not build these yet:

- Full lore database with relationship graphs
- Spoiler-aware timeline / versioned summaries
- Master glossary redesign
- External NER model integration (spaCy, MeCab)
- Mandatory embeddings dependency (optional/configurable only)
- Automatic hidden-identity merging without LLM confirmation
- Complex rendering rules for translation prompts

---

## Reference Material

### Academic Papers

**LlmLink** (COLING 2025) — "Dual LLMs for Dynamic Entity Linking on Long Narratives with Collaborative Memorisation and Prompt Optimisation"
- PDF: https://aclanthology.org/2025.coling-main.751.pdf
- Relevant: their "memorisation scheme" for maintaining an entity registry across chunks. Uses one LLM for local NER per chunk, a second for cross-chunk entity linking.
- Key takeaway: **separate extraction from linking**. Different cognitive tasks.

**LINK-KG** (2025) — "LLM-Driven Coreference-Resolved Knowledge Graphs"
- PDF: https://arxiv.org/pdf/2510.26486
- Relevant: Section III-A, "Coreference Resolution" — the **"type-specific prompt cache"** concept. Stores alias-to-canonical mappings per entity type. Three-phase pipeline: NER-LLM extracts, Mapping-LLM updates prompt cache, Resolve-LLM uses cache.
- Key takeaway: the **prompt cache pattern** — a growing alias→canonical mapping injected into each subsequent extraction call.

**Fischer & Volk** (MT Summit 2025) — "Name Consistency in LLM-based Machine Translation of Historical Texts"
- URL: https://aclanthology.org/2025.mtsummit-1.16/
- Key finding: LLMs achieve ~60% accuracy on person name translation without glossary, ~90% on place names. Glossary injection substantially boosts accuracy. Validates the importance of this refactor.

### Reference Codebase

The `context-aware-translation` repository (forked at `D:\AITools\context-aware-translation\`) implements related patterns:

- `utils/string_similarity.py` — bi-directional Jaro-Winkler (source of the `string_similarity` helper)
- `application/contracts/terms.py` — `TermTableRow` with `term_key` as canonical identity, `term` as surface string
- `core/term_memory.py` — `TermMemoryVersion` with `effective_start_chunk`, `latest_evidence_chunk`, `summary_text` for versioned lore
- `core/term_memory_builder.py` — bootstrap + incremental update pattern for evolving summaries
- `llm/summarizor.py` — prompts for summary compression and incremental update (update prompt returns `{"u":0}` unchanged or `{"u":1,"s":"new summary"}`)
- `core/context_manager.py` — `_group_by_similarity` uses union-find with Jaro-Winkler to cluster similar terms before batched translation; `TranslationContextManager` handles occurrence mapping and glossary injection

Key patterns to potentially adopt in future phases:
- Versioned summaries with temporal awareness for translation-time injection
- LLM-driven summary compression (bootstrap + incremental update)
- Union-find clustering with similarity cache for efficient batching

### Existing Codebase Files to Read

Before implementing, read these files in `dao-bridge-translator`:

- `src/dao_bridge/schemas.py` — current data models
- `src/dao_bridge/glossary.py` — full build/reconcile/export pipeline (~1200 lines)
- `src/dao_bridge/prompts/glossary_extract.txt` — current extraction prompt
- `src/dao_bridge/prompts/glossary_reconcile_term.txt` — current reconcile prompt
- `src/dao_bridge/translate.py` — glossary injection during translation (search for glossary-related scanning/rendering)
- `src/dao_bridge/config.py` — `GlossaryConfig` and `GlossaryPhaseConfig`
- `tests/` — existing test structure
