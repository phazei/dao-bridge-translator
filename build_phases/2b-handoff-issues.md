# Phase 2B — Handoff: Issues Found During Live Verification

## Status

Phase 2B (LLM summary compression) is committed (`fe5f4b5`) and works
mechanically. During live verification on real books we found several issues
worth fixing. **All exploratory code changes from the verification session were
discarded** — the working tree is back to the committed `fe5f4b5` state. This
doc records what we learned so the work can be redone cleanly, one change at a
time, with review.

This is a problem inventory, **not** an approved implementation plan. Each item
below should be discussed and scoped before any code is written.

**Progress:** Issue 4 (state-tracking bug, Option A), Issue 3 (compression
progress bar + console labels), and Issue 2 (schema echo) are now **RESOLVED** —
see their sections. Issue 1 (nicknames) remains open and is the next pickup.

## Test setup used

- Real book project: `translation_projects/rezero32-2b/` (ReZero vol, ja→en,
  ~348 entities, 147 chunks). Config: LM Studio at `http://127.0.0.1:1234/v1`,
  model `qwen3.6-35b-a3b-mtp`, `summary_compress_enabled: true`.
- Smaller comparison projects already present: `translation_projects/perfect-world-cn/`
  (no 2B) and `perfect-world-cn-2b/` (2B build only).
- Always run via the project venv: `.\.venv\Scripts\dao-bridge.exe ...`.

---

## Issue 1 — `nicknames` extraction is unreliable (schema + prompt)

**Where:** `ExtractedMention.nicknames: dict[str, str]` (`schemas.py`),
`prompts/glossary_extract.txt`, consumed at `translate.py` (rendered as
`{speaker} calls them "{nick}"`).

**Symptoms observed:**
- Validation failures during glossary build, e.g. batch `0011.b3`:
  `mentions.18.nicknames.Abel: Input should be a valid string` and
  `...nicknames.Otto: ...` — the model emitted `null` values for nickname keys.
- The retry loop recovered each time, but it cost a full extra round-trip per
  failure.

**Ground-truth investigation (batch 0011.b3, chunks 0011.009–012):**
- The failing mention was the character **ミディアム (Medium)**, who has a verbal
  tic of suffixing names with 「ちん」(-chin). The text genuinely contains
  「アベルちん」(Abel-chin), 「スバルちん」(Subaru-chin), 「ズィクルちん」(Zick-chin),
  and addresses Otto as 「オットーくん」(Otto-kun).
- Model returned `{"Zick":"Medium-jo", "Abel":null, "Otto":null}`. So:
  - It correctly sensed Medium relates to Zick/Abel/Otto.
  - `Abel:null` / `Otto:null` were the validation failures — it had no clean
    nickname so it padded with `null`. (For Abel it actually had "Abel-chin"
    available in the text and dropped it.)
  - `Zick:Medium-jo` was **directionally inverted** — "Medium-jo" (嬢) is what
    Zick calls Medium, not what Medium calls Zick.

**Root causes:**
1. **Direction confusion.** The `{speaker: nickname}` mapping is a hard 4-way
   binding (find a nickname → who says it → who it's about → key it correctly).
   Local models invert it frequently.
2. **Verbal tics don't belong here.** A suffix a character applies to *everyone*
   (like "-chin") is a speech pattern for `speech_style`, not a per-person
   nickname. The model kept cramming the tic into `nicknames`.
3. **`null`/empty padding.** When the model has no nickname for a listed speaker
   it emits `null`/`""` instead of omitting the entry, which fails the
   `dict[str, str]` schema.

**Overlap question raised:** how do `nicknames` and `aliases` differ? In code:
- `aliases: list[str]` — alternate *names for the entity*; used as a **clustering
  merge signal** (`glossary_clustering.py` matches an alias against other
  entities' surface forms). Load-bearing for dedup.
- `nicknames: dict[speaker, str]` — *how others address this entity*; used
  **only** at translation time for consistent forms of address. Not a clustering
  or embedding signal.
- They overlap (the string "Betty" is both an alias for Beatrice and the value
  of a nickname), which is part of why the field is hard. `nicknames` is the
  lowest-value, highest-error field.

**Decision reached:** keep the field (it has a real, distinct purpose) but make
the prompt instruction *short and language-agnostic* — the verification session's
first attempt was too long and ReZero/JP-specific. The trimmed instruction should
cover exactly three points:
- direction (key = speaker, value = name that speaker uses for THIS entity);
- only record nicknames actually shown in the text (no fabrication);
- a uniform suffix/pet-form applied to everyone is a verbal tic → `speech_style`,
  not a nickname; every value must be a non-empty string, `{}` if none.

**Caveat / open question:** this is prompt clarity only — it reduces but does not
eliminate the error rate, because the `{speaker: nickname}` binding is inherently
hard for a local model. If a clean full rebuild still shows nicknames garbled
across *multiple* characters (not just Medium), the structural fallback is to
drop the speaker-keyed dict and fold "names this entity is known by" into
`aliases` + put address styles in `speech_style`. Evaluate before committing to
that larger change.

**Verification still needed:** the trimmed prompt was applied and a single fresh
extraction of Medium looked correct (`{"Emitaria":"Medium-chan"}`, no nulls, tic
moved to speech_style, speaker name slightly mangled but cosmetic). That result
came from a *partial* build and was never validated across the whole book.

---

## Issue 2 — Summary-compress schema echo (regression from the 2B tightening) — RESOLVED

**Where:** `GlossarySummaryCompressResponse` (`schemas.py`), the auto-schema
injection in `LLMClient.complete_json` (`llm_client.py`), the compress call in
`compress_entity_summary` (`glossary.py`).

**Status: FIXED.** `complete_json` now injects a concrete **example instance** of
the response model (e.g. `{"summary": "..."}`), generated from the Pydantic
model, instead of the raw `model_json_schema()` envelope. The echo happened
because the model parroted the *schema envelope*
(`{"properties": ..., "required": ...}`) — itself valid JSON missing every field.
An example instance has no envelope to echo. See "Resolution".

**Live reproduction (qwen3.6-35b-a3b-mtp, 6 trials each, faithful prompt + the
exact injected schema):** *worse* than the doc's earlier ~7% estimate — it was
**deterministic**: WITH injection, 0/6 OK (6/6 echoed the schema, copying its
`description` verbatim); WITHOUT injection, 6/6 OK.

**Multi-model + multi-shape investigation (later finding that broadened the
fix):** the bug is **model-specific** and **not limited to single-field models**.
Tested 4 models × 3 response shapes (1-field compress, 2-field reconcile,
array-valued toc), 6 trials each, WITH schema injection (OK count):

| model | 1-field | 2-field | array |
|---|---|---|---|
| gemma-4-26b-a4b-it | 6/6 | 6/6 | 6/6 |
| gemma-4-31b-it | 6/6 | 6/6 | 6/6 |
| qwen3.6-35b-a3b-mtp | 0/6 | 1/6 | **0/6** |
| qwen3.6-27b | 6/6 | **0/6** | 6/6 |

Takeaways: both Gemma models are immune (they ignore the schema); the Qwen3.6
family echoes — and `35b-mtp` echoes **arrays too**, disproving the earlier
"arrays are safe" / "single-field only" claims. Injecting an **example
instance** instead was 6/6 across **every** model and shape. End-to-end through
the real (fixed) `complete_json` with `max_retries=1` (so any first-attempt echo
raises): gemma 5/5 all shapes; `35b-mtp` 5/5 compress, 4/5 reconcile (one
non-echo blip), 5/5 toc; `27b` 5/5 all shapes.

**Symptoms observed:** during the deferred compression pass, **every** multi-
observation entity failed validation on the first attempt with
`summary: Field required`, then recovered on retry. This is far worse than the
~7% first-attempt miss rate seen before the tightening.

**Raw response captured (entity `place_000001`):** the model returned the JSON
**schema definition** instead of an instance:
```json
{ "description": "LLM response for entity summary compression (Phase 2B).",
  "properties": { "summary": { "title": "Summary", "type": "string" } },
  "required": ["summary"], ... }
```
It even copied the schema's `description` verbatim. That object is valid JSON,
parses fine, but has no top-level `summary` key → validation fails → retry.

**Root cause (confirmed):** NOT a prompt/schema mismatch — both correctly
describe a single `summary` field. The earlier 2B "tightening" removed the
`changed` field, making the response model a **single required scalar field**.
A single-field schema is structurally almost identical to its own
`model_json_schema()` output, and `complete_json` injects that schema into every
prompt. With the dueling "here's the schema" + "return one field" instructions,
the model parrots the schema back. The old two-field `{changed, summary}` shape
was structurally distinct from its schema and incidentally avoided this.

**Affected call sites (all `complete_json` callers, not just single-field):**
the reconcile surface-form and entity-conflict calls (`GlossaryReconcileResponse`,
2 fields) echo heavily on `qwen3.6-27b` (0/6) and `35b-mtp` (1/6) — this is the
"reconcile" symptom seen live. `GlossarySpeechMergeResponse` (single field) was
the latent peer. The array models (`GlossaryClusterResponse.decisions`,
`TocTranslationResponse.titles`) are **not** safe either on `35b-mtp`. Because
the example-instance fix lives in `complete_json`, **every** caller is fixed at
once with no per-call-site changes.

**Resolution (implemented):** inject an example *instance*, not the schema
envelope (chosen over making `summary` optional, which would silently degrade to
naive concatenation when echoed). The model can't echo a schema envelope that
was never shown to it, and weaker prompts/models still get concrete shape
guidance.

- **`complete_json` (`llm_client.py`).** Injects
  `_example_instruction(response_model)` — a JSON example built by
  `_example_instance()`, which walks `model.model_fields`: scalars →
  `"..."`/`0`/`0.0`/`True`, `list[X]` → `[example(X)]`, `Optional[X]`/unions →
  first non-None member, and **recurses into nested `BaseModel` subclasses** (so
  `GlossaryClusterResponse` → `{"decisions": [{...}]}`). The example stays in
  sync with the schema automatically (still Pydantic-driven).

**Tests added (all passing):**
- `tests/test_llm_client.py`: `TestExampleInstruction` (the builder handles
  scalars, arrays, and nested models; the injected hint contains no
  `properties`/`required`/`$defs` envelope keys) and `TestShapeInjection` (an
  example instance — not the schema — is appended; the hint is injected once and
  retries carry only the error; array models get an array example).

`ruff check` on touched files clean (only the pre-existing `resolved_sf` /
`mock_llm_cls` / `I001` warnings); full suite **804 passed**.

---

## Issue 3 — No progress/visibility during the compression pass — RESOLVED

**Status: FIXED.** The single build progress task now follows both sub-phases
(extraction batches + the deferred compression pass), `--force-summaries` runs
through the same bar, and per-call log labels (`[summary:<entity_id>]`,
`[0011.b3]`) now render correctly **on the console**, not just in `run.log`. See
"Resolution" below. Verified live on `rezero32-2b`:
```
INFO Compressing entity summaries (deferred pass)...
INFO [summary:character_000069] LLM request start (1/3): ...
INFO [summary:character_000069] LLM request success (1/3): ...
⠧ Compressing summaries ━━━╸━━━━━━  98/355 title_000008  0:00:56
```

**Where:** CLI progress wiring in `cli.py` (`_run_glossary_build_with_progress`),
`compress_entity_summaries` / `GlossaryBuildProgress` in `glossary.py`, the
per-call label prefix in `LLMClient.complete` / `complete_json`
(`llm_client.py`), and the file-handler formatter in `logging.py`.

**Symptoms observed:** once extraction batches finished and the deferred
compression pass started ("Compressing entity summaries (deferred pass)..."),
the Rich progress bar froze/disappeared (it's `transient`) with no indication of
progress. On a 348-entity book this was a long, silent stretch. Additionally,
the per-call labels showed during build (`[0020.b2]`) but **not** during
compression (`[summary:<id>]`) on the console.

**Root cause (progress):** the progress bar was driven only by the per-batch
build callback (`GlossaryBuildProgress`). The compression pass had an
`on_progress` parameter but no caller wired it up, so it reported nothing.

**Root cause (labels) — CORRECTION to the earlier "NOT a bug" note:** the
console labels really *were* missing for the colon-form `[summary:<id>]`, and it
was a genuine bug, not faint rendering. The console `RichHandler` has
`markup=True`, so `[summary:place_000001]` parses as a Rich **style tag** and is
silently dropped (rendered as empty). Build IDs like `[0020.b2]` survived only
because they are not parseable as a style name. The labels were present in
`run.log` (plain formatter, no markup), which is why they looked file-only. Any
tag-like label (the colon-form `summary:<id>`, `cluster.<id>`) was vulnerable.

**Gotcha found (progress):** Rich's `progress.reset(task_id, ...)` **clears all
custom task fields**. If you reset and only re-supply `total`/`description`, the
next render throws `KeyError: 'item'`. Every custom field (`item`, `of_batch`)
must be re-supplied on `reset()`.

**Resolution (implemented):**

- **Single bar across both phases (`glossary.py` + `cli.py`).**
  `GlossaryBuildProgress` gained `phase` (`"extract"` | `"compress"`) and
  `phase_label` (defaulted, so existing extraction emits stay valid). A new
  `_compress_progress_adapter` wraps the compression pass's bare
  `on_progress(entity_id)` into a `GlossaryBuildProgress(phase="compress", ...)`;
  it is wired into all three compression call sites (tail-of-build, the Issue 4
  resume path, and `--force-summaries` via `_recompress_summaries_only`).
- **CLI wrapper phase switch (`cli.py`).** `_run_glossary_build_with_progress`
  switched its static `"Building glossary"` text column to `{task.description}`
  and adopted a reconcile-style phase switch: on a phase change it calls
  `progress.reset(task_id, ...)` **re-supplying every custom field**
  (`item`, `of_batch`, `description`) to avoid the `KeyError: 'item'` gotcha, and
  relabels to "Compressing summaries" with a fresh entity total.
- **`--force-summaries` routed through the wrapper (`cli.py`).** It no longer
  calls `glossary_build(...)` directly; it goes through
  `_run_glossary_build_with_progress(force_summaries=True)`, so it shows the
  compress bar too.
- **Console labels (`llm_client.py`).** A `_context_prefix()` helper builds the
  `[label] ` prefix via `rich.markup.escape(f"[{label}]")`, used at both ctx
  sources — `complete()` (start/success/error lines) and `complete_json()`
  (parse/validation failure lines). Escaping makes any tag-like label render
  literally on the console; non-tag-like labels (`0020.b2`) are returned
  unchanged, so build output is unaffected.
- **Clean `run.log` (`logging.py`).** Escaping adds a backslash (`\[summary:…]`)
  that the plain file formatter would otherwise show. Added
  `_PlainMarkupFormatter` (file handler only) that does a **targeted**
  `replace("\\[", "[")` — the exact inverse of `rich.markup.escape`, which only
  ever inserts `\` before a tag-opening `[`. So `run.log` shows clean
  `[summary:<id>]`. **A blanket `rich.markup.render().plain` was rejected**: it
  also eats arbitrary markup-like text in messages (e.g.
  `something [not a tag] here` → `something  here`, and bracketed content in raw
  LLM responses) — data loss in the log. Intentional styled tags elsewhere
  (`[bold]` in `state.py`) are left raw in `run.log`, unchanged from before.

**Tests added (all passing):**
- `tests/test_llm_client.py` (`TestContextLabel`): labels escaped at the source
  and verified to render literally through a real markup-enabled Rich console
  (batch and colon-form shapes); `complete_json` forwards the label to the
  start line.
- `tests/test_logging.py` (`TestPlainMarkupFormatter`): formatter unescapes the
  label, leaves non-escaped brackets and JSON arrays untouched, keeps `[bold]`
  raw.
- `tests/test_glossary.py`: `compress_entity_summaries` `on_progress` fires once
  per entity (LLM/bootstrap/resume-skip); the CLI wrapper drives a real Rich
  `Progress` through an extract→compress phase switch with no `KeyError`.

`ruff check` clean on touched files; full suite **792 passed**.

**Note on processing order (by design, not a bug):** the compression pass
iterates `glossary.entities` in **list/insertion order** (first-seen during
extraction, categories mixed) — so `character_000069` can be immediately
followed by `place_000020`. Entity ID numbers are a per-category creation
counter (`{category}_{max+1}`), independent of compression order. Within one
entity, observations are sorted by `chunk_id` so the summary reflects
chronological order.

---

## Issue 4 — `run` skips unfinished summary compression (state-tracking bug) — RESOLVED

**Status: FIXED (Option A).** Compression is kept as a tail pass of
`glossary_build`, but the coarse stage flag is now driven by *all* build
sub-steps (extraction + compression): the flag may only be `completed` once
`meta.summary_compress_done` is True (when compression is enabled). The skip
path reads `summary_compress_done` purely as a *stale-flag detector* — it is not
an alternate gate that overrides the coarse flag. See "Resolution" below.

**Where:** the "already completed — skipping" early return in `glossary_build`
(`glossary.py`); interaction with `meta.summary_compress_done` and the coarse
stage-completion flag.

**Symptoms observed:** after an **interrupted `--force-summaries`** run:
- `--force-summaries` had nulled all 348 summaries and set
  `meta.summary_compress_done = False`, then recompressed only ~8 entities before
  being aborted. Verified on disk: **340 of 348 entities had `summary: null`**.
- Running `dao-bridge run` then printed "Glossary build already completed —
  skipping" and proceeded **straight into `glossary-cluster`** — i.e. it
  clustered over a glossary whose summaries were almost all null.

**Root cause:** the Phase 2B compression pass is **not its own pipeline stage** —
it runs at the tail of the `glossary_build` stage and its completion is tracked
only by `meta.summary_compress_done`. But the early-return skip logic only checks
the coarse `glossary_build` stage flag (still `completed`), never
`summary_compress_done`. So an interrupted `--force-summaries` leaves the project
in an inconsistent state that `run` silently steps over. Clustering/embeddings
(2A) depend on summaries, so this produces bad clusters.

**Fix direction discussed:** before the "already completed — skip" return, if
`summary_compress_enabled` and `not meta.summary_compress_done`, **resume the
compression pass** (it already resume-skips entities that already have a summary)
and invalidate downstream cluster/reconcile, instead of skipping. Consider
whether `--force-summaries` should also reopen/track the stage so partial runs
are not reported as fully complete.

**Resolution (implemented):** Option A, in `glossary.py`.

- **Invariant enforced:** when `summary_compress_enabled` is True, the
  `glossary_build` stage may be `completed` only when `meta.summary_compress_done`
  is True. The coarse flag remains the single gate `run`/skip consult; it is now
  driven by every build sub-step, compression included.
- **Change 1 — early-return skip (`glossary_build`).** When the stage is
  `completed` but `summary_compress_enabled and not meta.summary_compress_done`,
  it no longer skips. It reopens the stage, resumes `compress_entity_summaries`
  (resume-skips entities that already have a summary), invalidates downstream
  cluster/reconcile, then re-marks the stage completed. A plain `run`/
  `glossary-build` now self-heals an interrupted compression instead of stepping
  over it into clustering.
- **Change 2 — `_recompress_summaries_only` (`--force-summaries`).** Added
  `reopen_stage(...)` *before* nulling summaries (so the flag is honestly
  `running` while it works — an interrupt mid-pass leaves `running` +
  `summary_compress_done=False`, which Change 1 resumes) and
  `mark_stage_completed(...)` after the pass.
- **`--force` was already correct** (resets flag → rebuild → compress →
  mark complete); a test now pins that invariant.
- **`meta.summary_compress_done` is NOT an alternate gate.** It records whether
  the compression sub-step finished; the skip path reads it only to detect that
  the coarse flag is stale (says completed, but compression was interrupted) so
  it can fix it. The coarse flag is still the authority.

**Tests added (`tests/test_glossary.py`, all passing):** resume-not-skip on stale
flag; resume invalidates downstream; fully-complete build still skips (no LLM
client constructed); compression-disabled completed build skips immediately;
`--force-summaries` leaves stage honestly completed; `--force-summaries`
interrupted leaves resumable state (`running`, not `completed`). Shared
`_build_then_gut_compression` helper reproduces the aborted-`--force-summaries`
state. `pytest tests/test_glossary.py tests/test_state.py` → 155 passed; no new
ruff errors on touched files (pre-existing `F841 resolved_sf`, `mock_llm_cls`
F841, and import-organize warnings are unchanged from HEAD).

**Not done (deferred Option B):** summary compression was **not** promoted to a
separate pipeline stage. The design question below is intentionally left open.

**Design question (still open):** should summary compression be promoted to a
real, separately-tracked pipeline stage (its own entry in `STAGE_NAMES`/state)
rather than a tail-of-build pass? Option A leaves two pieces of state (the coarse
flag + `summary_compress_done`) that must be kept consistent by hand; Option B
would fold this into the stage machinery and also resolve Issue 3's visibility
gap. Weigh the added complexity against keeping the two-flag invariant correct
over time.

---

## Current project state (rezero32-2b) — needs cleanup before reuse

As left by the discarded session:
- `glossary_build.json`: 348 entities, **~340 with `summary: null`** (gutted by
  the aborted `--force-summaries`).
- `_glossary_build_meta.json`: `summary_compress_done: false`.
- A `glossary-cluster` run was started over the gutted glossary; its output (if
  written) is contaminated and should not be trusted.

**Recommended recovery (once fixes land):** rebuild the glossary cleanly with
`glossary-build --force` (extraction + a complete compression pass), which also
gives an uncontaminated baseline to evaluate the Issue 1 prompt fix across the
whole book. Or, if extraction output is trusted, finish compression first
(resume per Issue 4) before any clustering. Either way, do NOT cluster until
every entity that has observations has a non-null summary.

---

## Suggested order of work (for discussion, not committed)

1. ~~**Issue 2** (schema echo)~~ — **DONE** (example-instance injection in
   `complete_json`; see Issue 2 "Resolution"). Fixes the echo across all callers
   and all response shapes on the affected models (compress, speech-merge, and
   the reconcile calls); tests added.
2. ~~**Issue 4** (`run` skipping unfinished compression)~~ — **DONE** (Option A,
   see Issue 4 "Resolution"). Stage flag now honors compression; resumes on a
   stale flag; tests added.
3. ~~**Issue 3** (compression progress bar + `--force-summaries` through it +
   console labels)~~ — **DONE** (see Issue 3 "Resolution"). Single bar across
   both phases; tag-like labels now render on the console; clean `run.log`;
   tests added.
4. **Issue 1** (nickname prompt) — separate concern from 2B; trim prompt, then
   evaluate on a full clean rebuild before deciding whether the schema needs a
   structural change.

Each item: one focused change, shown and reviewed before moving on.

**Remaining open:** Issue 1 (nicknames) only — it is the next pickup. (The
`complete_json validation failure (1/3): summary Field required` lines visible in
the Issue 3 live sample above are now gone; see Issue 2 "Resolution".)

## Verification conventions

- Run everything via `.\.venv\Scripts\dao-bridge.exe` (or
  `.\.venv\Scripts\python.exe -m pytest ...`).
- After any change: `ruff check src/ tests/` and `pytest` for the touched
  modules. Note: `glossary.py` has a **pre-existing** `F841 resolved_sf` ruff
  warning (around the reconcile surface-form resolution, line drifts as the file
  grows) unrelated to 2B — don't be alarmed by it, and don't let it mask new
  warnings. `tests/test_glossary.py` also has pre-existing `F841 mock_llm_cls`
  and import-organize (`I001`) warnings, unchanged from HEAD.
- Per-call LLM labels (`[summary:<id>]`, `[0020.b2]`) now render on the
  **console** as well as in `run.log` (Issue 3 fix). For raw failed-validation
  responses (logged at DEBUG), still inspect
  `translation_projects/<proj>/logs/run.log` directly.
