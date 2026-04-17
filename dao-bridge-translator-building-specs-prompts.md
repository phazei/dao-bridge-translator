
---

## **Prompt 1: Scaffold + I/O Layer for dao-bridge-translator**

> Build the foundation of `dao-bridge-translator`, a Python package that translates Japanese light novel EPUBs to English using LLM APIs. This prompt covers the package scaffold, CLI structure, configuration, state tracking, LLM client, logging, and the extraction + cleaning pipeline. Subsequent prompts will build chunking + assembly, classification + glossary, translation, and EPUB rebuild.
> 
> **Package name:** `dao-bridge-translator` (PyPI) / `dao_bridge` (module) / `dao-bridge` (CLI).
> **GitHub:** https://github.com/phazei/dao-bridge-translator (currently empty).
> **License:** Apache.
> **Python:** 3.12+.
> 
> **Dependencies (keep minimal):**
> - `ebooklib` — EPUB reading/writing
> - `beautifulsoup4` + `lxml` — HTML parsing
> - `markdownify` — HTML → markdown
> - `markdown` — markdown → HTML
> - `openai` — OpenAI-compatible API client (works with llama-server, vLLM, LM Studio, Claude, OpenAI, OpenRouter, etc.)
> - `pydantic` — schema validation
> - `click` — CLI
> - `pyyaml` — config
> - `tiktoken` — token counting (cl100k_base as the default approximation)
> - `rich` — progress bars, logging
> 
> **Package structure:**
> ```
> dao-bridge-translator/
> ├── src/dao_bridge/
> │   ├── __init__.py
> │   ├── cli.py
> │   ├── config.py
> │   ├── schemas.py
> │   ├── state.py
> │   ├── llm_client.py
> │   ├── extract.py
> │   ├── clean.py
> │   ├── workdir.py
> │   ├── logging.py
> │   └── prompts/            # external prompt templates (empty for now)
> ├── tests/
> ├── pyproject.toml
> ├── README.md
> └── LICENSE
> ```
> 
> **Work directory layout:**
> 
> Each book is a self-contained work directory. Multiple books can coexist as siblings sharing a master glossary one level up, but the code enforces nothing about directory structure beyond the book-level work directory. The master glossary path in config is just a path (absolute or relative to the work directory).
> 
> ```
> work/
>   config.yaml
>   manifest.json
>   state.json
>   raw/
>     001.xhtml
>     002.xhtml
>     ...
>   clean/
>     001.md
>     002.md
>     ...
>   chunks/                  # populated by chunker (later prompt)
>   translations/            # populated by translator (later prompt)
>   assembled/               # populated by assembler (later prompt)
>   summaries/               # populated by translator (later prompt)
>   glossary.json            # populated by glossary stage (later prompt)
>   logs/
>     run.log
> ```
> 
> Spine items are zero-padded 3-digit for lexical sort ordering. Later, chunks will use dot notation `NNN.MMM` (spine.chunk_index).
> 
> **Config schema (YAML, loaded into pydantic models):**
> 
> This is the complete config schema for the entire pipeline. Stages implemented in later prompts will use their respective sections; sections for unimplemented stages are loaded and validated but not acted on yet.
> 
> ```yaml
> source_epub: "/path/to/book.jp.epub"
> work_dir: "./work"
> 
> models:
>   classify:
>     base_url: "http://localhost:8080/v1"
>     api_key: "not-needed"
>     model: "qwen3-30b-a3b"
>     temperature: 0.0
>   glossary:
>     base_url: "http://localhost:8080/v1"
>     api_key: "not-needed"
>     model: "gemma-4-26b-a4b"
>     temperature: 0.2
>   translate:
>     base_url: "http://localhost:8080/v1"
>     api_key: "not-needed"
>     model: "gemma-4-26b-a4b"
>     temperature: 0.3
>   summarize:                              # optional — falls back to models.translate if absent
>     base_url: "http://localhost:8080/v1"
>     api_key: "not-needed"
>     model: "qwen3-30b-a3b"
>     temperature: 0.2
> 
> chunking:
>   target_tokens: 2000
>   max_tokens: 2400              # hard ceiling except for remainder-absorption
>   min_chunk_tokens: 400         # if remainder < this, absorb into previous chunk
>   flex_window_ratio: 0.2        # prefer scene break in last 20% of target
>   scene_break_patterns:
>     - '^\s*[\*]{3,}\s*$'
>     - '^\s*[◇]{3,}\s*$'
>     - '^\s*[＊]{3,}\s*$'
>     - '^\s*[・]{3,}\s*$'
>     - '^\s*[×](\s*[×])+\s*$'
>     - '^\s*[─]{4,}\s*$'
>     - '^\s*\*\s+\*\s+\*\s*$'
>   normalize_scene_breaks: "* * *"   # null to preserve originals
>   chunkable_classifications:
>     - chapter
>     - frontmatter
>     - backmatter
>     - toc_authored
>   # Items with classifications NOT in this list are skipped (chunk_count = 0).
> 
> glossary:
>   categories:
>     - character
>     - place
>     - ability
>     - title
>     - term
>     - item
>     - species
>     - clan
>     - organization
>     - other
>   category_hints:
>     character: "Named individuals including full names, given names, family names"
>     clan: "Family names, noble houses, tribes, or group identities multiple characters belong to"
>     organization: "Formal groups, guilds, military units, institutions"
>   master_glossary_path: null            # optional cross-book master (absolute or relative to work_dir)
>   crosscheck:
>     enabled: true
>     llm_assist: false
>     on_conflict: "prefer_master"         # prefer_master | prefer_book | flag_only
>   promote_on_complete: false
> 
> glossary_phase:
>   target_tokens_per_call: 8000  # greedy-pack chunks until adding the next exceeds
>   overlap_chunks: 0             # no overlap needed for entity extraction
> 
> translation_phase:
>   chunks_per_call: 1            # translate one chunk per LLM call
>   overlap_chunks: 1             # send the full previous chunk (source + translation) as context
>   cross_spine_overlap: true     # when starting a new spine, include last chunk of previous spine
>   double_pass: true
>   rolling_summary: true
>   summary_max_tokens: 2000      # sliding window budget for prior summaries in context
>   glossary_injection: "relevant" # "relevant" (only matching entries) or "all" (full glossary)
>   qa_check: true
>   qa_max_retries: 1             # retry full chunk translation this many times after QA failure before halting
>   min_length_ratio: 0.3         # programmatic floor: translation_tokens / source_tokens
>   max_length_ratio: 2.0         # programmatic ceiling
> 
> output:
>   epub_path: "./book.en.epub"
>   title_suffix: " (English Translation)"
>   new_identifier: false          # true to generate new UUID for output EPUB
>   css: "original"                # "original" (keep source CSS) or "default" (add minimal CSS)
>   add_translation_note: true     # add machine translation note to metadata description
>   validate: false                # run epubcheck on output if available on PATH
> 
> languages:
>   source: "ja"
>   target: "en"
> 
> llm:
>   max_retries: 3
>   retry_backoff_seconds: 2
>   request_timeout_seconds: 300
> ```
> 
> **Pydantic schemas (`schemas.py`):**
> 
> These are the complete schemas for the entire pipeline. Later prompts will use them as-is. Fields not populated by early stages are nullable and default to None.
> 
> - `ManifestItem`: `spine_index` (int), `padded_id` (str, computed property, zero-padded-3), `original_href` (str), `raw_path` (str), `clean_path` (str | None), `classification` (literal: `"chapter" | "frontmatter" | "backmatter" | "toc_auto" | "toc_authored" | "illustration" | "unknown"` | None, default None), `title` (str | None), `token_count` (int | None), `paragraph_count` (int | None), `chunk_count` (int | None — set by chunker later).
> - `Manifest`: `source_epub_path` (str), `book_id` (str — derived from EPUB metadata at init: prefer ISBN, fall back to normalized title+volume, fall back to filename stem), `spine` (list[ManifestItem]), `images` (list[str] — image item paths), `metadata` (dict — original EPUB metadata: title, author, language, identifier, etc.).
> - `Chunk`: `chunk_id` (str, format "NNN.MMM"), `spine_index` (int), `chunk_index` (int — per-spine, starting at 1), `source_file` (str — path to clean markdown file), `block_range` (tuple[int, int] — inclusive start, inclusive end of block indices in source file), `token_count` (int), `extended_for_remainder` (bool), `text` (str), `ends_at_scene_break` (bool).
>   - Note: no `mode` field, no `overlap_range`, no `new_text_starts_at_block`. Chunks are phase-agnostic. Each phase assembles calls from them.
> - `GlossaryEntry`: `japanese` (str | None — None for English-reference-only entries from import-reference), `reading` (str | None — from furigana), `english` (str), `category` (str — validated against `config.glossary.categories` at load time), `first_seen_chunk` (str | None), `aliases` (list[str] = []), `nicknames` (dict[str, str] = {} — `{speaker_english_name: nickname_english}`, only used by specific speakers), `speech_style` (str | None = None — prose description, characters only), `notes` (str | None = None), `source` (literal: `"seed" | "extracted" | "user" | "master"`), `source_books` (list[str] = [] — populated for master-glossary entries).
> - `Glossary`: `entries` (list[GlossaryEntry]), `version` (int — bumped on merges), `book_id` (str | None — None for master, set for per-book), `book_metadata` (dict = {} — title, author, volume; per-book only), `created_at` (datetime), `updated_at` (datetime).
> - `TranslatedChunk`: `chunk_id` (str), `source_text` (str — copy of Japanese source), `pass1_translation` (str — Pass 1 output, kept for debugging), `translated_text` (str — final translation: Pass 2 if double_pass, else Pass 1), `pass_count` (int — 1 or 2), `qa_result` (str | None — `"pass" | "fail"` | None if QA disabled), `qa_issues` (list[str] = [] — issues from QA, empty if passed), `total_attempts` (int — count of full chunk translation attempts, each = Pass 1 + optional Pass 2 + QA), `overlap_chunk_id` (str | None — which chunk provided overlap context), `summary_generated` (str | None — rolling summary snippet for this chunk), `token_usage` (dict — `{prompt_tokens, completion_tokens, total_tokens}` summed across all calls for the final successful attempt), `model_used` (str), `created_at` (datetime), `duration_seconds` (float — wall clock for all passes + assessment + summary of final attempt).
> 
> **LLM client (`llm_client.py`):**
> 
> Class `LLMClient(config: ModelConfig)`:
> - `complete(messages: list[dict], max_tokens: int | None = None) -> CompletionResult` — multi-turn completion. `messages` follows OpenAI chat format: `[{"role": "system"|"user"|"assistant", "content": "..."}]`. Returns a result object with `text`, `token_usage`, `model`, `finish_reason`.
> - `complete_json(messages: list[dict], response_model: type[BaseModel], max_retries: int = 3) -> BaseModel` — JSON output variant. Injects "respond with JSON matching this schema" instruction into the last user message (schema derived from the Pydantic model), parses the response, validates by constructing the Pydantic model. On parse/validation failure, retries with the error appended to the conversation. The retry counter resets on any successful parse. After `max_retries` consecutive failures, raises `LLMStructuredOutputError`.
> 
> Both methods use the OpenAI Python SDK pointed at `config.base_url`. Retry transient errors (rate limits, timeouts, connection errors) with exponential backoff per `llm.max_retries` / `llm.retry_backoff_seconds`.
> 
> Critical: the client takes **messages, not a single prompt string**. Later phases (translation) need multi-turn structure to cleanly separate previous-context from target-text.
> 
> **State tracking (`state.py`):**
> 
> JSON-backed state file at `state.json` in the work directory. All writes use the atomic write helper from `workdir.py` (write to `.tmp`, then `os.replace`).
> 
> Schema:
> ```json
> {
>   "run": {
>     "source_epub": "/path/to/book.epub",
>     "started_at": "2025-01-15T10:30:00Z",
>     "status": "running"
>   },
>   "stages": {
>     "extract": {"status": "completed", "started_at": "...", "completed_at": "..."},
>     "clean": {"status": "running", "started_at": "...", "completed_at": null, "error_message": null}
>   },
>   "items": {
>     "extract:001": {"status": "completed", "completed_at": "..."},
>     "translate:003.015": {"status": "failed_qa", "error_message": "...", "completed_at": "..."}
>   }
> }
> ```
> 
> - `stages` keys: `extract`, `clean`, `classify`, `chunk`, `glossary_build`, `glossary_reconcile`, `glossary_crosscheck`, `translate`, `assemble`, `rebuild`
> - `items` keys: `stage:item_id` where `item_id` is a padded spine id like `003` for per-spine work, a chunk id like `003.027` for per-chunk work, or a batch id like `glossary_build.batch.001` for batch work.
> - Item status values: `pending`, `started`, `completed`, `failed`, `failed_qa` (translation-specific: produced output but failed quality check).
> 
> Define Pydantic models for the state structure. Validate on load.
> 
> Functions for: marking stage started/completed/failed, marking item started/completed/failed, checking if a stage is done, iterating items needing work (status != completed), atomic updates. All operations idempotent — marking something completed twice is a no-op.
> 
> **Logging (`logging.py`):**
> 
> A `setup_logging(work_dir: Path, verbose: bool = False)` function that configures:
> - A `rich.logging.RichHandler` on the console: INFO by default, DEBUG when `--verbose` is passed.
> - A `logging.FileHandler` writing to `{work_dir}/logs/run.log` with timestamps, level, and module context.
> - A module-level logger accessible via `logging.getLogger("dao_bridge")` that all modules import.
> 
> All modules should add logging at appropriate levels while being implemented. No need to pre-specify every log line — add them as the code is written. Use DEBUG for detailed operational info, INFO for stage progress and key decisions, WARNING for recoverable issues, ERROR for failures.
> 
> **Workdir helpers (`workdir.py`):**
> 
> Path helpers for all file locations used throughout the pipeline. Every module uses these instead of constructing paths manually.
> 
> - `raw_path(work_dir, spine_index) -> Path` — `raw/NNN.xhtml`
> - `clean_path(work_dir, spine_index) -> Path` — `clean/NNN.md`
> - `chunk_dir(work_dir, spine_index) -> Path` — `chunks/NNN/`
> - `chunk_path(work_dir, chunk_id) -> Path` — `chunks/NNN/NNN.MMM.json`
> - `translation_dir(work_dir, spine_index) -> Path` — `translations/NNN/`
> - `translation_path(work_dir, chunk_id) -> Path` — `translations/NNN/NNN.MMM.json`
> - `assembled_path(work_dir, spine_index) -> Path` — `assembled/NNN.md`
> - `summary_path(work_dir) -> Path` — `summaries/rolling_summary.json`
> - `glossary_path(work_dir) -> Path` — `glossary.json`
> - `manifest_path(work_dir) -> Path` — `manifest.json`
> - `state_path(work_dir) -> Path` — `state.json`
> - `log_dir(work_dir) -> Path` — `logs/`
> - `ensure_dirs(work_dir)` — creates all subdirectories if they don't exist.
> - `atomic_write(path: Path, data: str | bytes)` — writes to `path.tmp`, then `os.replace()` to `path`. Used for all JSON writes (manifest, state, glossary, chunks, translations).
> - `pad_spine(spine_index: int) -> str` — zero-padded 3-digit string.
> - `format_chunk_id(spine_index: int, chunk_index: int) -> str` — "NNN.MMM" string.
> - `parse_chunk_id(chunk_id: str) -> tuple[int, int]` — returns (spine_index, chunk_index).
> 
> **CLI (`cli.py` with click):**
> ```
> dao-bridge init <epub> [--config config.yaml] [--work-dir ./work]
> dao-bridge extract [--work-dir ./work] [--force]
> dao-bridge clean [--work-dir ./work] [--force]
> dao-bridge status [--work-dir ./work]
> 
> # Placeholders (exit with "not yet implemented"):
> dao-bridge classify, glossary-build, glossary-reconcile, glossary-crosscheck,
>   glossary-promote, glossary-import-reference, glossary-export,
>   chunk, translate, assemble, rebuild, run
> ```
> 
> All commands accept `--verbose` to enable DEBUG-level console logging.
> 
> Every command checks state before starting and skips completed work unless `--force`. `run` (placeholder for now) will eventually chain all stages.
> 
> **`extract.py`:**
> 1. Open EPUB with ebooklib.
> 2. Iterate `book.spine` in order. For each item of type `ITEM_DOCUMENT`:
>    - Write raw XHTML to `raw/NNN.xhtml` (zero-padded).
>    - Record original href in manifest.
> 3. Enumerate all `ITEM_IMAGE` items and record paths (for rebuild).
> 4. Preserve original EPUB metadata (title, author, language, identifier, etc.) in manifest.
> 5. Derive `book_id` from EPUB metadata: prefer ISBN, fall back to normalized title+volume, fall back to filename stem. Store in manifest.
> 6. Warn if there are `ITEM_DOCUMENT` items not in spine.
> 
> **`clean.py`:**
> 
> For each `raw/NNN.xhtml`:
> 1. Parse with BeautifulSoup (lxml).
> 2. Pre-process ruby: replace each `<ruby>X<rt>Y</rt></ruby>` with text `{X|Y}`. Handle nested `<rb>`/`<rt>` variants and `<rp>` fallback parens.
> 3. Strip `<style>` and `<script>` entirely.
> 4. Strip purely presentational elements: empty `<div>`/`<span>`, elements with only class/id attributes and no semantic tag, etc.
> 5. Convert to markdown with `markdownify`, configured to preserve `<br>`, `<b>`/`<strong>`, `<i>`/`<em>`, headings, `<hr>` (converted to scene break marker). Strip classes/ids.
> 6. Normalize whitespace: collapse runs of blank lines to exactly two, strip trailing whitespace.
> 7. Write to `clean/NNN.md`.
> 8. Count paragraphs (blocks separated by blank lines) and tokens (tiktoken cl100k).
> 9. Update manifest with counts (using atomic write).
> 
> **Tests (`pytest`):**
> - Schema round-trip and validation tests for all schemas (ManifestItem, Manifest, Chunk, GlossaryEntry, Glossary, TranslatedChunk).
> - `clean.py` fixture tests: plain prose, ruby text, nested ruby with `<rp>` fallback, `<br>` tags, bold/italic, headings, `<hr>`, images, scripts/styles, messy calibre-converted markup with deeply nested divs.
> - `state.py` idempotency tests: re-running stages, crash recovery mid-stage, JSON round-trip.
> - `llm_client.py` tests with mocked OpenAI SDK: retry logic, `complete_json` Pydantic validation + retry on invalid output, retry counter reset on success.
> - `workdir.py` tests: atomic write (verify temp file is cleaned up, verify content is correct after replace), path helpers produce correct paths, `parse_chunk_id` round-trips with `format_chunk_id`.
> - Integration test: sample mini-EPUB → `init` → `extract` → `clean` → verify output structure.
> 
> **Deliverables:**
> - All modules with docstrings.
> - `pyproject.toml` with metadata, dependencies, CLI entry point.
> - `README.md`: project overview, installation, quickstart showing currently-working commands. Include a recommended multi-book layout with books as sibling directories sharing a `../master_glossary.json`.
> - `LICENSE` (Apache).
> - Tests passing.
> 
> Do not implement chunking, classification, glossary, translation, assembly, or rebuild. Placeholders should exit cleanly with "not yet implemented."

---

---


## **Prompt 2: Chunker + Assembler for dao-bridge-translator**

> Build the chunking and assembly modules for `dao-bridge-translator`. This builds on the foundation from Prompt 1 (scaffold, I/O layer, extract + clean stages). Assumes the package structure, schemas, config, state tracking, and LLM client are already in place.
> 
> **Scope of this prompt:**
> - `src/dao_bridge/chunk.py` — deterministic paragraph-aware chunking, per-spine, classification-aware
> - `src/dao_bridge/assemble.py` — reassembly of translated chunks back into per-spine markdown
> - CLI commands `chunk` and `assemble`
> - Comprehensive tests
> 
> **Prerequisites (should already exist from Prompt 1):**
> - `ManifestItem`, `Manifest`, `Chunk` pydantic schemas in `schemas.py` (with correct classification literals including `toc_auto`/`toc_authored`, and `Chunk.block_range` documented as inclusive start/end)
> - Config loading for the `chunking` section (including `chunkable_classifications`)
> - State tracking with `chunk` and `assemble` stages defined
> - `workdir.py` path helpers
> 
> **Note:** The schemas, classification literals, and config fields needed by this prompt were already included in Prompt 1's forward-compatible definitions. No schema or config updates should be needed.
> 
> ---
> 
> ## chunk.py
> 
> **Purpose:** For each spine item that should be chunked, produce a deterministic sequence of chunk JSON files based on the cleaned markdown content.
> 
> ### Block parsing
> 
> Parse `clean/NNN.md` into a list of `Block` objects. A block is the atomic unit the chunker works with — blocks are never split across chunks.
> 
> Block types:
> - `paragraph` — a run of non-empty lines separated from neighbors by blank lines. Lines joined by `<br>` or markdown hard line breaks (two trailing spaces) stay in one paragraph block.
> - `scene_break` — a paragraph whose content matches one of the configured `scene_break_patterns` regex patterns. Tagged so chunker can prefer to break at these.
> - `heading` — a line starting with `#` (ATX-style markdown heading).
> - `hr` — a line that is `---`, `***`, or `___` alone (markdown horizontal rule). Treated as a scene break for chunking purposes.
> 
> Implementation: a state machine over the lines of the file. Accumulate non-blank lines into a paragraph-in-progress, emit when hitting a blank line. Headings and HRs flush the current paragraph and become their own blocks.
> 
> Each block carries:
> - `index: int` — position in the block list for this file, starting at 0
> - `kind: Literal["paragraph", "scene_break", "heading", "hr"]`
> - `text: str` — the markdown content of the block, preserving internal `<br>` and inline formatting
> - `token_count: int` — computed once via `tiktoken.cl100k_base`
> 
> Define `Block` as a dataclass or pydantic model in `chunk.py` (internal; not exposed in `schemas.py` since consumers work with `Chunk` objects).
> 
> ### Scene break detection and normalization
> 
> When parsing blocks, test each paragraph's text against the configured `scene_break_patterns` regex list (strip surrounding whitespace first). If any pattern matches, the block's `kind` becomes `scene_break`. `hr` blocks are always treated as scene breaks by the chunker.
> 
> If `normalize_scene_breaks` is set in config (non-null), the block's `text` field is replaced with the normalized form during block parsing. This applies to both `scene_break` and `hr` blocks. The original `clean/NNN.md` file is not modified — normalization only affects the in-memory block and the chunk's saved `text`.
> 
> ### The greedy packing algorithm
> 
> ```
> blocks = parse_blocks(clean_markdown)
> chunks = []
> current_blocks = []
> current_tokens = 0
> chunk_index = 1
> 
> for block in blocks:
>     if current_tokens + block.token_count > target_tokens:
>         # Try to find a scene break or heading in the flex window
>         flex_min = target_tokens * (1 - flex_window_ratio)
>         break_idx = find_last_break_point_in_range(
>             current_blocks,
>             min_cumulative_tokens=flex_min,
>             max_cumulative_tokens=target_tokens,
>         )
>         if break_idx is not None:
>             emit_chunk(current_blocks[:break_idx + 1], chunk_index)
>             leftover = current_blocks[break_idx + 1:]
>             current_blocks = leftover + [block]
>             current_tokens = sum(b.token_count for b in current_blocks)
>         else:
>             emit_chunk(current_blocks, chunk_index)
>             current_blocks = [block]
>             current_tokens = block.token_count
>         chunk_index += 1
>     else:
>         current_blocks.append(block)
>         current_tokens += block.token_count
> 
> # Handle final accumulated chunk
> if current_blocks:
>     if chunks and current_tokens < min_chunk_tokens:
>         # Absorb into previous chunk
>         previous = chunks.pop()
>         combined = previous.blocks + current_blocks
>         emit_chunk(combined, previous.chunk_index, extended=True)
>     else:
>         emit_chunk(current_blocks, chunk_index)
> ```
> 
> `find_last_break_point_in_range` scans `current_blocks` for the latest block whose `kind` is `scene_break`, `heading`, or `hr` AND whose cumulative token position (sum of token_counts from block 0 through this block, inclusive) falls between `min_cumulative_tokens` and `max_cumulative_tokens`. Returns the index within `current_blocks`, or `None` if no such break point exists.
> 
> When multiple break points fall within the flex window, picks the latest one (closest to target). This maximizes chunk size while respecting natural breaks.
> 
> ### Edge cases
> 
> - **Empty file (zero blocks):** produce zero chunks. Manifest `chunk_count` for this item becomes 0. Log a warning.
> - **File smaller than `target_tokens`:** produces exactly one chunk containing all blocks.
> - **File smaller than `min_chunk_tokens`:** produces exactly one chunk (nothing to absorb into).
> - **Single block exceeds `max_tokens`:** that block becomes its own chunk, exceeding max. Log a warning. The pipeline still works, it just sends an oversized chunk to the LLM. This is pathological but should not crash.
> - **Final remainder < `min_chunk_tokens`:** absorbed into previous chunk. Mark that chunk `extended_for_remainder = true`. If there is no previous chunk (i.e., the whole file is < `min_chunk_tokens`), just emit the single small chunk.
> 
> ### Chunk output
> 
> Each chunk is written (via `atomic_write`) to `chunks/NNN/NNN.MMM.json` (zero-padded 3-digit for both spine and chunk index). Uses the `Chunk` pydantic model from `schemas.py`.
> 
> The `text` field is the LLM-ready content: blocks joined by `\n\n` (paragraph separator), scene breaks already normalized if configured.
> 
> ### Classification filtering
> 
> Before chunking any spine item, check its `classification` against `chunking.chunkable_classifications`. If not in the list (or classification is `unknown` — treat `unknown` as chunkable with a warning), skip chunking for that item. Set `manifest.spine[i].chunk_count = 0`, leave the `chunks/NNN/` directory empty (or don't create it), and record the stage as complete for that item.
> 
> ### Validation before emitting chunks
> 
> After building all chunks for a spine item but before writing them to disk, validate:
> - Every block index in `[0, n_blocks)` appears in exactly one chunk's `block_range`. No gaps, no overlaps.
> - Every chunk has at least one block.
> - Chunk indices are sequential starting from 1.
> - Sum of chunks' token counts approximately equals sum of blocks' token counts (within ±1 to account for join whitespace rounding).
> 
> If any check fails, raise a clear error before writing anything. No partial output.
> 
> ### State tracking integration
> 
> The `chunk` stage is per-spine-item. For each item:
> - Mark stage started
> - Chunk (or skip based on classification)
> - Mark stage completed with `chunk_count` recorded
> - On error, mark failed with error message, raise
> 
> Items whose `chunk` stage is already marked complete are skipped unless `--force` is passed.
> 
> The `chunk` command must error clearly if any spine item has `classification: null` (i.e., classify hasn't been run). Error message: "Classification required before chunking. Run `dao-bridge classify` first."
> 
> ### CLI
> 
> ```
> dao-bridge chunk [--work-dir ./work] [--spine N] [--force] [--verbose]
> ```
> 
> - No `--spine`: chunk all eligible items that haven't been chunked yet.
> - `--spine N`: chunk just spine N (by integer index). Useful for iterating during development.
> - `--force`: rechunk even if already complete. Deletes existing `chunks/NNN/` directory first.
> 
> Progress via `rich.progress.Progress`.
> 
> After chunking, update `manifest.json` (via `atomic_write`) with the final `chunk_count` for each processed item.
> 
> ---
> 
> ## assemble.py
> 
> **Purpose:** For each spine item that has translated chunks, concatenate them into a single translated markdown file.
> 
> ### Behavior
> 
> For each spine item in the manifest where `chunk_count > 0`:
> 1. Load all chunks from `chunks/NNN/` (sorted by `chunk_index`).
> 2. Load corresponding translations from `translations/NNN/NNN.MMM.json` (assumed to exist — translation stage is a future prompt, but assembler should work with whatever produces files matching the `TranslatedChunk` schema).
> 3. Verify all chunk IDs have corresponding translations. If any are missing, raise an error listing the missing ones.
> 4. Concatenate `translated_text` of each chunk in order, joined by `\n\n`.
> 5. Write to `assembled/NNN.md` (via `atomic_write`).
> 
> For spine items with `chunk_count == 0` (skipped by chunker — illustrations, auto-tocs): no assembly work. The rebuild stage will handle these by passing through raw XHTML.
> 
> ### CLI
> 
> ```
> dao-bridge assemble [--work-dir ./work] [--spine N] [--force] [--verbose]
> ```
> 
> Same semantics as `chunk`: default all, `--spine N` for one, `--force` to overwrite.
> 
> Only runs for items whose translation stage is complete. If translations are missing for any chunks of an item, that item is skipped with a warning (assembly can be re-run after translations finish).
> 
> ### State tracking
> 
> The `assemble` stage is per-spine-item. Mark started/completed/failed as appropriate.
> 
> ### Validation
> 
> Before writing `assembled/NNN.md`:
> - All expected chunk translations are present.
> - Concatenated output is non-empty.
> - Rough token count of output is within reasonable bounds of sum of input translation token counts (sanity check).
> 
> ---
> 
> ## Tests
> 
> Unit tests for `chunk.py`:
> - Block parsing: plain prose, multi-line paragraphs with `<br>`, headings, HRs, scene breaks with various patterns, mixed content.
> - Scene break normalization: normalized form applied when configured, original preserved when null.
> - Greedy packing: empty file (zero chunks), single small block (one chunk), exact-target file (one chunk), target+1 file (two chunks or one if below min_chunk), scene break in flex window (break at scene break), scene break outside flex window (break at target), multiple scene breaks in flex window (latest chosen), oversized single block (emit with warning).
> - Remainder absorption: tiny remainder absorbed into previous chunk with `extended_for_remainder = true`.
> - Classification filtering: items with `illustration` or `toc_auto` classification produce zero chunks.
> - Determinism: running the chunker twice on the same input produces byte-identical JSON output (modulo timestamps if any).
> - Block coverage: every block appears in exactly one chunk, no gaps, no duplicates.
> 
> Unit tests for `assemble.py`:
> - Multiple chunks assemble in correct order.
> - Missing translation raises clear error.
> - Single-chunk spine assembles to single block of output.
> - Skipped items (chunk_count == 0) are not assembled (no file produced).
> - Out-of-order chunk files on disk still assemble in correct order (sort by chunk_index, not filename-sort-dependent).
> 
> Integration tests:
> - Full pipeline from Prompt 1 plus chunk and assemble: mini EPUB → init → extract → clean → (manual classification setting via manifest edit in test) → chunk → (manual translation injection in test) → assemble.
> - Verify assembled output matches expected.
> 
> Test fixtures:
> - `tests/fixtures/clean/short_chapter.md` — under min_chunk_tokens, one block.
> - `tests/fixtures/clean/single_chunk_chapter.md` — under target, multiple blocks.
> - `tests/fixtures/clean/two_chunk_chapter.md` — needs splitting, no scene breaks.
> - `tests/fixtures/clean/scene_break_chapter.md` — multiple scene breaks in various positions.
> - `tests/fixtures/clean/oversized_paragraph.md` — one paragraph larger than max_tokens.
> - `tests/fixtures/clean/tiny_remainder.md` — final content absorbs into previous.
> 
> ---
> 
> ## Deliverables
> 
> - `src/dao_bridge/chunk.py` with full implementation and docstrings.
> - `src/dao_bridge/assemble.py` with full implementation and docstrings.
> - Updated `src/dao_bridge/cli.py` with `chunk` and `assemble` commands replacing their placeholder stubs.
> - All tests passing.
> - Updated `README.md` quickstart showing chunk and assemble commands.

---

---

## **Prompt 3a: Classification Module for dao-bridge-translator**

> Build the classification stage for `dao-bridge-translator`. This builds on Prompts 1 and 2 (scaffold, I/O, chunker, assembler). Classification must run before chunking — the chunker consults each spine item's classification to decide whether to produce chunks for it.
> 
> **Scope:**
> - `src/dao_bridge/classify.py` — per-spine-item classification with structural hints + LLM fallback
> - CLI command `classify`
> - Prompt template `src/dao_bridge/prompts/classify.txt`
> - Tests
> 
> **Prerequisites (from earlier prompts):**
> - `ManifestItem`, `Manifest` pydantic schemas in `schemas.py` (classification literals already correct from Prompt 1)
> - `LLMClient` in `llm_client.py` with `complete_json()` method (accepts Pydantic model class)
> - Config loading with `models.classify` section
> - State tracking with per-item granularity
> - `workdir.py` path helpers
> 
> **Classification values** (already in `ManifestItem.classification` literal from Prompt 1):
> - `chapter` — regular narrative prose
> - `frontmatter` — preface, prologue, author's note at start, copyright page, title page
> - `backmatter` — afterword, author's note at end, acknowledgments
> - `toc_auto` — auto-generated table of contents (nav structure, simple link list)
> - `toc_authored` — ToC that's actual authored content (commentary, illustrations, prose)
> - `illustration` — spine item that's essentially just an image
> - `unknown` — fallback when neither structural hints nor LLM produced a confident answer
> 
> **Classification strategy — three layers:**
> 
> **Layer 1: Structural hints (no LLM).** Inspect the raw XHTML (`raw/NNN.xhtml`) for deterministic signals:
> 
> - If the root element or any ancestor has `epub:type="toc"` or contains `<nav epub:type="toc">`: classify as `toc_auto`.
> - If the file's visible text content (after stripping tags) is under 30 tokens AND it contains at least one `<img>` tag: classify as `illustration`.
> - If the file contains only whitespace and/or a single title heading with no body content: classify as `frontmatter` with title extracted.
> 
> These hints short-circuit the LLM call. Log when a hint is used.
> 
> **Layer 2: LLM classification** for items not resolved by layer 1. Send:
> - First ~500 characters of the raw XHTML (preserves structural context like `<head>`, class names, epub:type attributes).
> - First ~1500 tokens of the cleaned markdown (the actual prose content).
> - Positional context: "this is spine item N of M".
> 
> The LLM returns structured JSON via `complete_json()` with a Pydantic response model. Classification and title populate the manifest item.
> 
> **Layer 3: Manual override.** After the stage runs, users can edit `manifest.json` directly to correct classifications. The classify command respects existing non-null classifications and skips them unless `--force` is passed. Document this in the CLI help text and README.
> 
> **Structured output Pydantic model:**
> 
> Define a `ClassificationResponse` Pydantic model (internal to classify.py or in schemas.py) with fields:
> - `classification`: one of the seven valid values.
> - `title`: str | None — extracted title if discernible. For chapters typically "Chapter N: Title". For frontmatter "Preface", "Prologue", etc. For illustrations null.
> - `confidence`: literal `"high" | "medium" | "low"`. Items marked `low` get logged for user review.
> - `reasoning`: str — free-form explanation for debugging. Logged but not acted on.
> 
> **Prompt template (`prompts/classify.txt`):**
> 
> The template uses simple `{variable}` substitution. It instructs the classifier on:
> - The seven classification values and what distinguishes them.
> - The distinction between `toc_auto` (machine-generated, simple link list) and `toc_authored` (has prose commentary, integrated illustrations, or section introductions alongside navigation).
> - To prefer content over position (spine position is a weak signal).
> - To extract title when present.
> - To return exactly the JSON schema specified, no extra keys, no markdown fences.
> 
> Keep the prompt under 400 tokens to leave room for the content excerpts.
> 
> **Classify module (`classify.py`):**
> 
> Functions:
> - `classify_item(item: ManifestItem, raw_xhtml: str, clean_markdown: str, position: tuple[int, int], config, llm_client) -> ClassificationResult` — classifies one item. `position` is `(spine_index, total_spine_items)`. Returns a result with classification, title, confidence, reasoning, and which layer produced the result.
> - `apply_structural_hints(raw_xhtml: str, clean_markdown: str) -> ClassificationResult | None` — layer 1 only. Returns None if no hint matches.
> - `llm_classify(raw_excerpt: str, clean_excerpt: str, position: tuple[int, int], config, llm_client) -> ClassificationResult` — layer 2.
> - `run_classify_stage(work_dir: Path, config, force: bool = False) -> None` — iterates manifest, classifies each item needing it, updates manifest atomically.
> 
> Define an internal `ClassificationResult` dataclass with fields: `classification`, `title`, `confidence`, `reasoning`, `source` (`"structural" | "llm"`).
> 
> **CLI:**
> 
> ```
> dao-bridge classify [--work-dir ./work] [--spine N] [--force] [--verbose]
> ```
> 
> - Default: classify all items without a classification set.
> - `--spine N`: classify only spine N.
> - `--force`: reclassify all items (or just spine N if also passed).
> 
> Progress via `rich.progress.Progress`. After classification, print a summary: counts per classification value, and list items with `confidence: low` for user attention.
> 
> **State tracking:**
> 
> `classify` is a per-item stage. `item_id` format is the padded spine id (e.g., `003`). Mark started/completed/failed. Idempotent — re-running skips already-classified items.
> 
> **Tests:**
> 
> - Structural hint: `<nav epub:type="toc">` XHTML → `toc_auto` without LLM call. Verify no LLM call is made.
> - Structural hint: XHTML with only an `<img>` and no prose → `illustration`.
> - Structural hint: XHTML with only a title heading → `frontmatter` with title extracted.
> - LLM classification with mocked `complete_json`: verify prompt includes raw excerpt, clean excerpt, and position; verify response parsed correctly via Pydantic model.
> - Manual override preservation: item with existing classification is skipped without `--force`.
> - `--force` reclassifies everything.
> - Confidence `low` items are logged but still saved with their classification.
> - Invalid LLM response: `complete_json` retries are exercised; ultimate failure marks item as `unknown` with a warning rather than crashing the whole stage.
> - Integration: mini EPUB → init → extract → clean → classify → verify manifest classifications.
> 
> Fixtures: small XHTML files exercising each structural hint case, plus one "ambiguous" XHTML that requires LLM classification.
> 
> **Deliverables:**
> 
> - `src/dao_bridge/classify.py` with full implementation and docstrings.
> - `src/dao_bridge/prompts/classify.txt` — external prompt template.
> - Updated `src/dao_bridge/cli.py` with `classify` command replacing its placeholder stub.
> - All tests passing.
> - Updated `README.md` quickstart including `classify` step and notes on manual override.

---

## **Prompt 3b: Glossary Module for dao-bridge-translator**

> Build the glossary stage for `dao-bridge-translator`. This builds on Prompts 1, 2, and 3a. The glossary stage extracts proper nouns and character information from the chunked Japanese text, reconciles within-book conflicts, and optionally cross-checks against a cross-book master glossary.
> 
> **Scope:**
> - `src/dao_bridge/glossary.py` — build, reconcile, crosscheck, promote, export, import-reference operations
> - CLI commands: `glossary-build`, `glossary-reconcile`, `glossary-crosscheck`, `glossary-promote`, `glossary-export`, `glossary-import-reference`
> - Prompt templates in `src/dao_bridge/prompts/`
> - Tests
> 
> **Prerequisites:**
> - All from Prompts 1, 2, 3a.
> - Chunks written to `chunks/NNN/NNN.MMM.json`.
> - `LLMClient.complete_json()` available (accepts Pydantic model class).
> - `GlossaryEntry` and `Glossary` schemas already defined in `schemas.py` from Prompt 1 (with all fields: nicknames, speech_style, source, source_books, book_id, book_metadata, timestamps).
> - Config `glossary` and `glossary_phase` sections already loaded from Prompt 1.
> 
> **Note:** No schema or config updates should be needed — Prompt 1 already includes the complete definitions.
> 
> Validate `entry.category` against `config.glossary.categories` at load time. Raise a clear error listing invalid categories if mismatched.
> 
> ---
> 
> ## Operations
> 
> ### glossary-build
> 
> Extracts the per-book glossary from Japanese chunks.
> 
> **Packing:** greedy-pack chunks across all spines (in spine + chunk order) until adding the next chunk would exceed `glossary_phase.target_tokens_per_call`. Emit a batch, start a new one. Last batch emitted as-is regardless of size.
> 
> **Per-batch LLM call:**
> 
> Prompt includes:
> - Instructions on what to extract: proper nouns from the configured categories, with Japanese form, reading (from `{kanji|reading}` furigana markup), proposed English translation, category, aliases, speech style and nicknames for characters.
> - The configured category list with hints.
> - The existing per-book glossary so far (accumulated from previous batches). Rendered compactly, grouped by category. If this exceeds half the model's context, truncate to the most recently added entries with a note.
> - The batch's chunk text (the `text` field from each chunk, joined with chunk separators).
> 
> Structured output via `complete_json()` with a Pydantic response model:
> 
> ```json
> {
>   "entries": [
>     {
>       "japanese": "ナツキ・スバル",
>       "reading": "ナツキ・スバル",
>       "english_proposed": "Natsuki Subaru",
>       "category": "character",
>       "aliases": ["スバル"],
>       "nicknames": {},
>       "speech_style": "Speaks casually in modern Japanese, frequent slang and 21st-century references.",
>       "notes": "Protagonist."
>     }
>   ],
>   "corrections": [
>     {
>       "existing_english": "Priscilla",
>       "japanese": "プリシラ・バーリエル",
>       "corrected_english": "Priscilla Barielle",
>       "reason": "Full name appears in this section."
>     }
>   ]
> }
> ```
> 
> **Merge logic** per returned entry:
> 
> - If Japanese form not in per-book glossary: add with `source: extracted`, `first_seen_chunk` set to the first chunk in this batch where the term appears (or the batch's first chunk if not trackable).
> - If Japanese form exists (from a prior batch): union aliases, merge nicknames dict, concatenate speech_style observations (dedup identical sentences), preserve existing English. Log differing English proposals in an internal conflict list.
> - If `source: user`: never modify.
> 
> **Corrections** are logged to an internal conflict list; not applied here. Reconcile handles them.
> 
> **Save after each batch:** write the current state of the glossary to `glossary.json` (via `atomic_write`) after every batch completes. This makes the stage resumable — if it crashes mid-stage, resume picks up at the next unprocessed batch.
> 
> **State tracking:** per-batch items with `item_id` like `glossary_build.batch.001`. Track batches, not individual entries.
> 
> ### glossary-reconcile
> 
> Resolves within-book conflicts from the build stage.
> 
> **Conflict sources:**
> - Entries where multiple batches proposed different English forms.
> - `corrections` entries from build batches that suggested changes to previously-extracted terms.
> - Entries flagged with inconsistent categories across batches.
> 
> **Resolution:**
> 
> For each conflict, call the LLM with:
> - The Japanese term and its reading.
> - The current English form in the glossary.
> - Alternative English forms proposed, each with a short context snippet from the chunk where it was proposed.
> - A request for the best single English form, with reasoning.
> 
> Apply the winner. Log the decision and reasoning to a reconciliation report at `<work_dir>/glossary_reconcile_report.md`.
> 
> **Speech style observations** across batches are handled differently: not conflicts, but cumulative. The reconcile stage consolidates multiple speech_style strings for the same character into a single coherent description via an LLM call ("merge these observations about how this character speaks into one coherent description"). This gives the translator a clean speech_style field rather than three redundant sentences.
> 
> **State tracking:** per-conflict items. Also per-character for speech-style consolidation.
> 
> ### glossary-crosscheck
> 
> Compares per-book glossary against the master glossary.
> 
> **No-op if `config.glossary.master_glossary_path` is null or file doesn't exist.** Log and skip.
> 
> **Matching:**
> 
> For each per-book entry, find matches in the master:
> 1. Exact Japanese form match.
> 2. If no Japanese (per-book entry lacks it): reading match, then English exact match.
> 3. Alias overlap: any per-book alias equals any master alias.
> 
> If multiple master entries match, collect all; flag as a multi-match conflict for user review.
> 
> **Per-field comparison** for single-match cases:
> 
> - `english`: if different, this is an English-form conflict. Apply `on_conflict` rule.
> - `category`: if different, prefer master (categories are canonical).
> - `aliases`, `nicknames`: union (additive, no conflict possible).
> - `speech_style`: if per-book has new info not in master's speech_style, flag for promotion (will update master when promoted).
> - `notes`: append per-book's notes to master's if different, separated by newline.
> 
> **`on_conflict: prefer_master`**: adopt master's value in the per-book glossary for conflicting fields. Log the change.
> **`on_conflict: prefer_book`**: keep per-book's value; will override master at promote time.
> **`on_conflict: flag_only`**: keep per-book's value, flag in report, don't modify either glossary.
> 
> **LLM-assisted conflicts** (when `llm_assist: true`):
> 
> For English-form conflicts and multi-match conflicts, call the LLM: "Master glossary has term A with translation X. This book's glossary has translation Y. Are these the same term being translated differently, or distinct terms that happen to share a form? If same, which translation is better? Provide reasoning." The recommendation goes into the report but is not auto-applied.
> 
> **Output:**
> 
> `<work_dir>/glossary_crosscheck_report.md` with sections:
> - Summary counts (matches / new / conflicts / adjusted)
> - New entries (no master match) — candidates for promote
> - Matched entries, no differences
> - Matched entries, adjustments applied (per `on_conflict`)
> - Conflicts requiring review
> - Multi-match warnings (same term matches multiple master entries)
> 
> ### glossary-promote
> 
> Merges the per-book glossary into the master.
> 
> **Input:** `<work_dir>/glossary.json` (per-book, post-crosscheck, possibly user-edited). `config.glossary.master_glossary_path` (master, may not exist yet).
> 
> **Behavior:**
> 
> - Back up master to `<master_path>.backup.YYYYMMDDHHMMSS` if it exists.
> - Load master (or create empty if doesn't exist).
> - For each per-book entry:
>   - If not in master (by matching rules above): add with `source: master`, `source_books: [book_id]`.
>   - If in master: union aliases and nicknames, add `book_id` to `source_books` if absent, merge speech_style via the same consolidation logic as reconcile.
>   - English form: `prefer_master` is the default; but if the per-book glossary was created with `on_conflict: prefer_book`, adopt the per-book English (the crosscheck stage already signaled the intent to override).
> - Write master (via `atomic_write`) with updated `version`.
> - `--dry-run`: print the diff without writing.
> 
> ### glossary-import-reference
> 
> Extracts English proper nouns from an English EPUB (or directory of EPUBs) and merges them into the master glossary. Useful for seeding the master from authoritative translations before starting work on a new book.
> 
> **Input:** path to an English EPUB or directory.
> 
> **Behavior:**
> 
> - For each EPUB:
>   - Extract spine items using existing `extract` logic (treat as a standalone operation, don't require a full work dir).
>   - Clean to markdown using existing `clean` logic.
>   - Pack the English text into batches by token budget.
>   - For each batch, LLM call: "Extract all proper nouns from this English text. For each, give the English form, category, and any aliases. Do not attempt to guess Japanese forms." Structured JSON output via `complete_json()`.
>   - Merge into master with `source: seed`, `japanese: null`, `reading: null`, `source_books: [english_book_id]`.
> - Entries from multiple English books: merge on English exact match, union aliases and source_books.
> 
> This is a utility command run out-of-pipeline. Does not require a `work_dir` for the target translation — it just needs a path to the English EPUB(s) and a master glossary path.
> 
> ```
> dao-bridge glossary-import-reference <english-epub-or-dir> [--master PATH] [--dry-run] [--verbose]
> ```
> 
> ### glossary-export
> 
> Produces a human-readable markdown view of the per-book glossary.
> 
> **Output:** `<work_dir>/glossary.md` (or stdout with `--stdout`).
> 
> Format: grouped by category, entries sorted alphabetically within each category by English form. Each entry shows Japanese, reading (if any), English, aliases, nicknames (if any), speech style (if any), notes (if any).
> 
> Also supports `--master` to export the master glossary instead of per-book.
> 
> ---
> 
> ## Prompt templates
> 
> Create these in `src/dao_bridge/prompts/`:
> 
> - `glossary_extract.txt` — build-stage extraction prompt. Variables: `{categories}`, `{category_hints}`, `{existing_glossary}`, `{chunk_batch}`.
> - `glossary_reconcile_term.txt` — reconcile prompt for term conflicts. Variables: `{japanese}`, `{reading}`, `{current_english}`, `{alternatives}`.
> - `glossary_reconcile_speech.txt` — reconcile prompt for speech-style consolidation. Variables: `{character_name}`, `{observations}`.
> - `glossary_crosscheck_conflict.txt` — optional LLM-assisted conflict reasoning. Variables: `{japanese}`, `{reading}`, `{master_entry}`, `{book_entry}`.
> - `reference_extract.txt` — English proper noun extraction for import-reference. Variables: `{categories}`, `{text_batch}`.
> 
> All prompts under 800 tokens. Use simple `{variable}` substitution, no Jinja.
> 
> ---
> 
> ## CLI
> 
> ```
> dao-bridge glossary-build [--work-dir ./work] [--force] [--verbose]
> dao-bridge glossary-reconcile [--work-dir ./work] [--force] [--verbose]
> dao-bridge glossary-crosscheck [--work-dir ./work] [--force] [--verbose]
> dao-bridge glossary-promote [--work-dir ./work] [--master PATH] [--dry-run] [--verbose]
> dao-bridge glossary-import-reference <epub-or-dir> [--master PATH] [--dry-run] [--verbose]
> dao-bridge glossary-export [--work-dir ./work] [--master PATH] [--stdout] [--output PATH] [--verbose]
> ```
> 
> All stage commands integrate with state tracking, skip completed work without `--force`. The `run` command (chains stages) should now include build + reconcile + crosscheck (if master configured).
> 
> ---
> 
> ## Tests
> 
> **glossary-build:**
> - Packing algorithm: chunks packed greedily, last batch may be smaller.
> - Mocked LLM: entries merged correctly, aliases unioned, speech styles accumulated.
> - Resume after crash: re-running from a partial glossary.json picks up at next batch.
> - Corrections logged to internal conflict list.
> - `user`-sourced entries never modified.
> 
> **glossary-reconcile:**
> - Mocked LLM picks winning English form; glossary updated accordingly.
> - Speech style consolidation merges observations for each character.
> - Report file generated with all decisions.
> 
> **glossary-crosscheck:**
> - No-op when master path is null.
> - Exact Japanese matches detected.
> - Alias overlap matches detected.
> - Multi-match flagged for review.
> - `on_conflict` modes behave correctly.
> - Report generated with all sections populated.
> 
> **glossary-promote:**
> - New entries added with correct source_books.
> - Existing entries get book_id appended to source_books.
> - Master backed up before write.
> - `--dry-run` produces diff output without writing.
> 
> **glossary-import-reference:**
> - Single EPUB processed end-to-end with mocked LLM.
> - Directory of EPUBs processed, results merged.
> - Entries have `source: seed`, `japanese: null`.
> 
> **glossary-export:**
> - Markdown output grouped by category, sorted alphabetically.
> - All optional fields rendered when present, omitted when null.
> 
> **Integration:**
> - Full pipeline through glossary stages with mini-book fixture and mocked LLM.
> - Crosscheck against a prepared master glossary fixture.
> 
> ---
> 
> ## Deliverables
> 
> - `src/dao_bridge/glossary.py` with all operations, full docstrings.
> - All prompt templates.
> - Updated `src/dao_bridge/cli.py` with glossary subcommands replacing placeholders where they exist.
> - All tests passing.
> - Updated `README.md` explaining the glossary flow, master glossary concept, and when to use which command.

---

---

## **Prompt 4: Translation Module for dao-bridge-translator**

> Build the translation stage for `dao-bridge-translator`. This is the core of the pipeline — translating chunked Japanese text to English using LLM calls with glossary context, overlap for style continuity, rolling summaries for narrative continuity, a revision pass, and quality assessment.
> 
> **Scope:**
> - `src/dao_bridge/translate.py` — per-chunk translation with double-pass, QA assessment, overlap, rolling summary
> - CLI command `translate`
> - Prompt templates in `src/dao_bridge/prompts/`
> - Tests
> 
> **Prerequisites:**
> - All from Prompts 1, 2, 3a, 3b.
> - Chunks at `chunks/NNN/NNN.MMM.json`.
> - Per-book glossary at `<work_dir>/glossary.json`.
> - `LLMClient` with `complete()` and `complete_json()` methods.
> - State tracking with per-chunk granularity.
> - `TranslatedChunk` schema already defined in `schemas.py` from Prompt 1 (with all fields: pass1_translation, pass_count, qa_result, qa_issues, total_attempts, overlap_chunk_id, summary_generated, duration_seconds).
> - Config `translation_phase` section already loaded from Prompt 1 (with all fields: summary_max_tokens, glossary_injection, qa_check, qa_max_retries, min_length_ratio, max_length_ratio).
> - `models.summarize` config with fallback to `models.translate` if absent.
> 
> ---
> 
> ## Translation flow per chunk
> 
> For each chunk (processed sequentially in spine order, then chunk order within each spine):
> 
> ### 1. Gather context
> 
> **Glossary rendering:**
> If `glossary_injection: "relevant"`: scan the chunk's `text` for each glossary entry's `japanese` field (substring match). Include only matching entries. If `glossary_injection: "all"`: include the entire glossary.
> 
> Render format, grouped by category:
> ```
> GLOSSARY (terms in this section)
> 
> Characters:
> - ナツキ・スバル [Natsuki Subaru] — protagonist
>   Speech: Speaks casually, modern slang, 21st-century references.
>   Nicknames: Ram calls him "Barusu".
> - ラム [Ram] — Rem's twin
>   Speech: Blunt, dismissive, cold detachment. Short sentences.
> 
> Places:
> - グァラル [Guaral] — fortress city in Vollachia
> 
> Abilities:
> - 死に戻り [Return by Death]
> ```
> 
> Include `speech_style` and `nicknames` only for character entries. Omit fields that are null/empty.
> 
> **Overlap:**
> - If this chunk is `NNN.MMM` where `MMM > 1`: load the translation for `NNN.(MMM-1)`. Use its `source_text` and `translated_text`.
> - If this chunk is `NNN.001` and `cross_spine_overlap` is enabled: find the previous spine in the manifest that has `chunk_count > 0`. Load its last chunk's translation.
> - If this is the very first chunk of the book (`001.001`) or if `overlap_chunks: 0`: no overlap.
> - If the required overlap chunk hasn't been translated: raise an error. Translation must proceed sequentially when overlap is enabled.
> 
> **Rolling summary:**
> If `rolling_summary` is enabled: load `<work_dir>/summaries/rolling_summary.json`. This is a list of `{chunk_id, summary}` objects. Render the last N entries such that total tokens ≤ `summary_max_tokens`. Oldest entries are excluded first (sliding window). The entries remain in the file; the window just determines how many are injected into context.
> 
> ### 2. Build message sequence and execute
> 
> The translation uses a progressively-extended conversation. Each pass appends messages to the same conversation history, allowing LLM servers with prefix caching (llama-server, vLLM) to reuse KV cache from prior passes. Three separate API calls are made, each sending the full growing message list.
> 
> **API Call 1 — Pass 1 (initial translation):**
> 
> ```
> Messages:
>   system: [Translation instructions from prompts/translate_pass1.txt]
>           [Rendered glossary]
>           [Rolling summary, if available]
>   
>   user (if overlap exists):
>     "Here is the preceding section and its English translation, 
>      for style and voice continuity. Do not retranslate this."
>     
>     JAPANESE:
>     [Previous chunk source_text]
>     
>     ENGLISH:
>     [Previous chunk translated_text]
>   
>   user:
>     "Translate the following Japanese text to English:"
>     
>     [Current chunk text]
> ```
> 
> Response: Pass 1 translation text.
> 
> **API Call 2 — Pass 2 (revision, if `double_pass` enabled):**
> 
> Append to the message list from Call 1:
> 
> ```
>   assistant: [Pass 1 translation — the actual response from Call 1]
>   
>   system: [Revision instructions from prompts/translate_pass2.txt]
>   
>   user:
>     "Please revise your translation above. Compare against the 
>      original Japanese for accuracy, naturalness, and glossary 
>      consistency. Output only the revised translation."
> ```
> 
> Response: Pass 2 (revised) translation text.
> 
> **API Call 3 — QA assessment (if `qa_check` enabled):**
> 
> Before calling the LLM judge, run programmatic sanity checks:
> - Compute token ratio: `translation_tokens / source_tokens`.
> - If ratio < `min_length_ratio` (default 0.3): fail immediately with "output suspiciously short — possible refusal or truncation."
> - If ratio > `max_length_ratio` (default 2.0): fail immediately with "output suspiciously long — possible repetition loop."
> - If programmatic checks pass, proceed to LLM assessment.
> 
> Append to the message list from Call 2 (or Call 1 if double_pass disabled):
> 
> ```
>   assistant: [Pass 2 translation — the actual response from Call 2]
>   
>   system: "Assess translation quality. Respond in JSON only."
>   
>   user:
>     "Assess the translation above against the original Japanese source. 
>      Respond with JSON: {\"result\": \"pass\" or \"fail\", \"issues\": [...]}
>      
>      Fail if any of:
>      - Significant mistranslation of meaning
>      - Missing content (paragraphs or sentences skipped)
>      - Repetition loops
>      - Refusal to translate
>      - Glossary violations (proper nouns not matching the glossary)"
> ```
> 
> Response: parsed via `complete_json()` into a Pydantic QA response model.
> 
> If `complete_json()` exhausts retries on the QA assessment response (malformed JSON), treat it the same as a QA failure: save what we have, halt translation. The error message should distinguish "QA assessment returned unparseable JSON after retries" from "QA assessment returned valid JSON with result=fail."
> 
> ### 3. Handle QA result
> 
> - **Pass:** proceed to save.
> - **Fail:** retry the entire chunk (fresh Pass 1 + Pass 2 + assessment, new conversation). Up to `qa_max_retries` attempts (default 1). If still failing after all retries, mark as `failed_qa` in state, log issues, save the translation we have, and **halt the translation pipeline**. QA failure likely indicates a model issue that needs to be addressed before continuing.
> 
> On re-run of the translate command, `failed_qa` chunks are retried automatically (same as `failed` — the user presumably fixed something). No `--force` needed.
> 
> ### 4. Generate rolling summary
> 
> If `rolling_summary` enabled:
> 
> Make a small LLM call (using the `summarize` model config, which falls back to `models.translate` if not configured):
> 
> ```
> System: [Instructions from prompts/translate_summary.txt]
>         [Prior summaries — same sliding window as translation context,
>          last N entries within summary_max_tokens budget]
> 
> User: [The final translated text for this chunk]
> ```
> 
> Append `{chunk_id, summary}` to `summaries/rolling_summary.json` (via `atomic_write` of the full file). If an entry for this chunk_id already exists (retry / re-translation), overwrite it.
> 
> ### 5. Save results
> 
> Write `TranslatedChunk` to `translations/NNN/NNN.MMM.json` (via `atomic_write`).
> 
> Create the `translations/NNN/` directory if it doesn't exist.
> 
> ---
> 
> ## Prompt templates
> 
> Create in `src/dao_bridge/prompts/`:
> 
> **`translate_pass1.txt`** — system prompt for initial translation. Variables: `{glossary}`, `{rolling_summary}`.
> 
> Content guidance (the actual prompt text — write this fully in the deliverable):
> - Role: translating a Japanese light novel for fluent English readers.
> - Natural, readable English prose. Not literal — prioritize readability and the author's narrative voice.
> - Preserve each character's speech style as described in the glossary.
> - Translate most honorifics naturally (先生 → teacher/professor, 様 → Lord/Lady). Keep -sama, -dono only when dramatically significant. Be consistent.
> - `{kanji|reading}` notation indicates author-intended readings for names/terms. Use the reading to inform translation. Don't reproduce the notation in output.
> - Preserve paragraph structure: each source paragraph → one output paragraph. Do not merge or split.
> - Preserve scene break markers exactly as they appear.
> - Use glossary English forms for all proper nouns without deviation.
> - Do NOT add translator's notes, footnotes, or commentary.
> - Do NOT censor or sanitize content.
> - Do NOT summarize — translate the complete text.
> - Output only the English translation, nothing else.
> 
> **`translate_pass2.txt`** — system prompt for revision pass. No variables needed (glossary already in context from Pass 1's system message).
> 
> Content guidance:
> - Role: revising an English translation of a Japanese light novel.
> - Compare draft against original Japanese.
> - Fix mistranslations, awkward phrasing, unnatural English.
> - Ensure proper nouns match glossary.
> - Ensure character voices match speech_style descriptions.
> - Preserve paragraph structure.
> - Output only the revised translation, nothing else.
> 
> **`translate_summary.txt`** — system prompt for rolling summary generation. Variables: `{prior_summaries}`.
> 
> Content guidance:
> - Summarize key events, character actions, new information revealed.
> - Under 200 words.
> - Focus on plot-relevant details useful for a translator working on the next section.
> - Don't include opinions or analysis.
> - Prior summaries provided for narrative continuity context.
> 
> All prompts should be well-crafted and ready to use. Under 500 tokens each.
> 
> ---
> 
> ## Config
> 
> All config fields for this prompt are already defined in Prompt 1's complete config schema. No additions needed.
> 
> ---
> 
> ## translate.py module structure
> 
> Functions:
> 
> - `translate_chunk(chunk: Chunk, config, glossary: Glossary, overlap: TranslatedChunk | None, rolling_summaries: list[dict], llm_client: LLMClient) -> TranslatedChunk`
>   - Orchestrates Pass 1 + Pass 2 + QA + summary for one chunk.
>   - Handles retries on QA failure (up to `qa_max_retries`).
>   - Returns the completed TranslatedChunk.
> 
> - `build_pass1_messages(chunk, glossary, overlap, rolling_summaries, config) -> list[dict]`
>   - Constructs the message list for Pass 1.
> 
> - `extend_pass2_messages(messages: list[dict], pass1_response: str, config) -> list[dict]`
>   - Appends Pass 2 system + user messages.
> 
> - `extend_qa_messages(messages: list[dict], pass2_response: str, config) -> list[dict]`
>   - Appends QA assessment system + user messages.
> 
> - `programmatic_qa_check(source_text: str, translated_text: str, config) -> QAResult | None`
>   - Returns QAResult on failure, None on pass (proceed to LLM judge).
> 
> - `render_glossary(glossary: Glossary, chunk_text: str, mode: str) -> str`
>   - Renders glossary for prompt injection. `mode` is "relevant" or "all".
> 
> - `render_rolling_summary(summaries: list[dict], max_tokens: int) -> str`
>   - Renders the sliding window of summaries for prompt injection.
> 
> - `generate_summary(translated_text: str, chunk_id: str, rolling_summaries: list[dict], llm_client: LLMClient, config) -> str`
>   - Generates the rolling summary for one chunk. Includes prior summaries in context (same sliding window as translation).
> 
> - `load_overlap(chunk: Chunk, manifest: Manifest, config) -> TranslatedChunk | None`
>   - Finds and loads the correct overlap chunk (same spine or cross-spine).
> 
> - `run_translate_stage(work_dir: Path, config, force: bool = False, from_chunk: str | None = None, to_chunk: str | None = None) -> None`
>   - Main stage runner. Iterates chunks in order, skips completed, handles state.
>   - On QA failure after retries exhausted: saves translation, marks `failed_qa`, halts pipeline.
>   - Prints end-of-run summary on exit (see below).
> 
> Internal dataclass:
> - `QAResult`: `result` (pass/fail), `issues` (list[str]), `source` ("programmatic" | "llm").
> 
> ---
> 
> ## CLI
> 
> ```
> dao-bridge translate [--work-dir ./work] [--spine N] [--chunk NNN.MMM] [--from NNN.MMM] [--to NNN.MMM] [--force] [--verbose]
> ```
> 
> - Default: translate all untranslated chunks in sequential order.
> - `--spine N`: only chunks in spine N.
> - `--chunk NNN.MMM`: translate a single specific chunk (shorthand for `--from NNN.MMM --to NNN.MMM`).
> - `--from NNN.MMM`: start translating from this chunk (inclusive). Without `--to`, continues to end of book.
> - `--to NNN.MMM`: stop translating after this chunk (inclusive). Must be used with `--from`.
> - `--force`: retranslate even if already completed.
> 
> Range semantics: `--from 003.005 --to 005.002` translates all chunks whose chunk_id falls within that range (zero-padded string comparison). This naturally spans across spines.
> 
> **End-of-run summary:** On exit (success or failure), print a summary:
> - On success: "Translated X chunks. Total tokens: Y. Average time per chunk: Z seconds."
> - On QA halt: "Translated X chunks successfully. Halted at chunk NNN.MMM: [QA failure reason]. Fix the issue and re-run to continue."
> - On infrastructure failure: "Translated X chunks successfully. Failed at chunk NNN.MMM: [error]. Re-run to retry."
> 
> Progress display via `rich`:
> - Current chunk ID.
> - Pass indicator (Pass 1 / Pass 2 / QA / Summary).
> - Tokens processed so far.
> - Average tokens/second.
> - Estimated time remaining (based on remaining chunks × average time per chunk).
> - Count of completed chunks.
> 
> ---
> 
> ## State tracking
> 
> `translate` stage is per-chunk. `item_id` is the chunk_id (`003.015`). States: `pending`, `started`, `completed`, `failed`, `failed_qa`.
> 
> `failed_qa` is distinct from `failed` (infrastructure error). `failed_qa` means the translation was produced but didn't pass quality checks. The translation is still saved (it might be usable with manual review).
> 
> The translate command:
> - Skips `completed` items (unless `--force`).
> - Retries `failed` items (infrastructure errors are often transient).
> - Retries `failed_qa` items (user presumably fixed the issue — swapped model, edited glossary, adjusted prompts).
> - Halts on new QA failure after retries exhausted.
> 
> Sequential enforcement: when `overlap_chunks > 0`, before translating chunk N, verify chunk N-1 is `completed`. If not, error: "Chunk NNN.MMM depends on NNN.{MMM-1} which has not been translated."
> 
> ---
> 
> ## Summaries file
> 
> `<work_dir>/summaries/rolling_summary.json`:
> 
> ```json
> [
>   {"chunk_id": "001.001", "summary": "Subaru arrives at..."},
>   {"chunk_id": "001.002", "summary": "Priscilla proposes..."},
>   ...
> ]
> ```
> 
> Append-only during normal operation (via `atomic_write` of the full file). Overwrite existing entry on chunk re-translation. The file is the source of truth; the sliding window is computed at render time, not by deleting old entries.
> 
> ---
> 
> ## Tests
> 
> All tests with mocked LLM responses.
> 
> **Core translation flow:**
> - Pass 1 only (double_pass disabled): single API call, output saved as `translated_text`, `pass1_translation` equals `translated_text`.
> - Double pass: two API calls, `pass1_translation` and `translated_text` are different, `pass_count` is 2.
> - Message construction: verify Pass 1 messages include system + glossary + summary + overlap (when applicable) + source text.
> - Message extension: verify Pass 2 messages append to Pass 1 messages correctly.
> - Message extension: verify QA messages append to Pass 2 messages correctly.
> 
> **Overlap:**
> - Same-spine overlap: chunk 003.015 includes 003.014's source + translation in user message.
> - Cross-spine overlap: chunk 003.001 includes last chunk of spine 002.
> - Cross-spine disabled: chunk 003.001 has no overlap.
> - First chunk of book: no overlap, overlap messages omitted.
> - Missing overlap chunk: raises error when overlap is enabled.
> 
> **Rolling summary:**
> - Summary generated after each chunk, appended to rolling_summary.json.
> - Summary generation includes prior summaries in context (same sliding window).
> - Sliding window: when total exceeds max_tokens, oldest entries excluded from rendering (but remain in file).
> - Re-translation overwrites existing summary entry for that chunk_id.
> - Summary disabled: no summary call made, no file written.
> 
> **Glossary rendering:**
> - "relevant" mode: only entries whose Japanese appears in chunk text.
> - "all" mode: entire glossary.
> - Speech style included for character entries.
> - Nicknames included for character entries.
> - Non-character entries: speech_style and nicknames omitted.
> 
> **QA:**
> - Programmatic check: too-short output triggers fail without LLM call.
> - Programmatic check: too-long output triggers fail without LLM call.
> - Programmatic check: normal-length output proceeds to LLM judge.
> - LLM judge pass: chunk saved as completed.
> - LLM judge fail: chunk retried up to qa_max_retries, then halts pipeline.
> - Persistent QA failure: chunk saved as failed_qa, pipeline halts with clear message.
> - QA disabled: no assessment, chunk saved as completed after Pass 2.
> - Malformed QA JSON after complete_json retries exhausted: treated as QA failure, pipeline halts.
> 
> **Resume and state:**
> - Completed chunks skipped on re-run.
> - Failed chunks retried on re-run.
> - failed_qa chunks retried on re-run (no --force needed).
> - `--chunk NNN.MMM` translates only that chunk.
> - `--from`/`--to` range translates the specified range.
> - Sequential enforcement: error when previous chunk not completed (overlap enabled).
> 
> **End-of-run summary:**
> - Successful run prints chunk count, total tokens, average time.
> - QA halt prints success count and failure details.
> 
> **Edge cases:**
> - Chunk with no glossary matches: glossary section says "No matching glossary entries for this section."
> - Empty rolling summary (first chunk): summary section omitted from system message.
> - Very large glossary with "all" injection: verify it fits within reasonable token budget (log warning if glossary exceeds 5000 tokens).
> 
> ---
> 
> ## Deliverables
> 
> - `src/dao_bridge/translate.py` with full implementation and docstrings.
> - Prompt templates: `translate_pass1.txt`, `translate_pass2.txt`, `translate_summary.txt`.
> - Updated `src/dao_bridge/cli.py` with `translate` command.
> - `summaries/` directory creation in `workdir.py` (should already exist from Prompt 1's `ensure_dirs`).
> - All tests passing.
> - Updated `README.md` with translation stage documentation, config options, and guidance on handling failed_qa chunks (re-run after fixing the issue).

---

---

## **Prompt 5: Rebuild Module for dao-bridge-translator**

> Build the rebuild stage for `dao-bridge-translator`. This is the final pipeline stage — it takes the translated and assembled markdown files and produces an output EPUB by modifying a copy of the original source EPUB. The approach is to preserve the original EPUB structure exactly and only replace the body content of translated XHTML files, translate ToC entries, and update metadata.
> 
> **Scope:**
> - `src/dao_bridge/rebuild.py` — EPUB reconstruction via modified copy
> - `src/dao_bridge/toc.py` — ToC translation (ncx + nav.xhtml)
> - CLI command `rebuild`
> - Fully-implemented `run` command chaining all stages
> - Prompt template `src/dao_bridge/prompts/translate_toc.txt`
> - Static asset `src/dao_bridge/templates/default.css` (optional — see CSS section)
> - Tests
> 
> **Prerequisites:**
> - All from Prompts 1–4.
> - `assembled/NNN.md` files for all translated spine items.
> - Per-book glossary at `<work_dir>/glossary.json`.
> - Original source EPUB accessible (path from config or manifest).
> - Config `output` section already loaded from Prompt 1.
> 
> ---
> 
> ## Core approach: modified copy, not rebuild from scratch
> 
> The output EPUB is constructed by copying the original source EPUB at the ZIP level and replacing only the files that changed. This preserves all original structure: images, fonts, CSS, DRM metadata, Apple/Kindle-specific files, custom OPF entries, archive compression settings, and everything else we don't need to touch.
> 
> **Do NOT use ebooklib's `write_epub()` for output.** It reconstructs the EPUB from its internal model and may lose non-standard entries, custom namespaces, or unusual archive structure. Use ebooklib only for reading and understanding the source EPUB structure.
> 
> **ZIP-level file replacement with proper mimetype handling:**
> 
> EPUB spec requires the `mimetype` file to be the first entry in the ZIP, stored uncompressed (no compression), with no extra field data. Naive `writestr(filename, ...)` loops won't guarantee this.
> 
> Implementation: copy `ZipInfo` objects from the source for each entry to preserve per-entry `compress_type` and other metadata. For the `mimetype` entry specifically, ensure it is written first with `compress_type=ZIP_STORED` and no extra field.
> 
> ```python
> import zipfile
> 
> with zipfile.ZipFile(source_epub, 'r') as src:
>     with zipfile.ZipFile(output_epub, 'w') as dst:
>         for item in src.infolist():
>             if item.filename in modified_files:
>                 # Preserve original ZipInfo metadata (compress_type, etc.)
>                 new_info = item
>                 dst.writestr(new_info, modified_files[item.filename])
>             else:
>                 dst.writestr(item, src.read(item.filename))
> ```
> 
> The `modified_files` dict maps original ZIP paths to new content bytes. Only translated XHTML files, ToC files, and the OPF file are modified. Everything else copies through unchanged.
> 
> **OPF-relative vs ZIP-absolute href resolution:** The manifest stores `original_href` which is the href as it appears in the OPF file (relative to the OPF's location in the ZIP). To find the corresponding ZIP entry, resolve this relative to the OPF file's directory path within the ZIP. For example, if the OPF is at `OEBPS/content.opf` and the href is `Text/chapter1.xhtml`, the ZIP path is `OEBPS/Text/chapter1.xhtml`. Store and use ZIP-absolute paths in the `modified_files` dict.
> 
> ---
> 
> ## Body replacement for translated items
> 
> For each spine item in the manifest where `chunk_count > 0` (it was translated):
> 
> 1. Read the original XHTML from the source EPUB ZIP (using the resolved ZIP path from `original_href`).
> 2. Read the translated markdown from `assembled/NNN.md`.
> 3. Convert the markdown to HTML using a ruby-safe pipeline:
>    a. **Before markdown conversion:** Replace all `{kanji|reading}` notation with unique placeholders (e.g., `RUBY_0001`, `RUBY_0002`). This prevents the `markdown` library's `attr_list` extension (included in `extra`) from mangling the `{...}` syntax.
>    b. **Markdown conversion:** Use the `markdown` library with the `extra` extension.
>    c. **Post-process scene breaks:** Convert normalized scene break markers to `<hr/>` tags.
>    d. **Post-process line breaks:** Ensure `<br>` tags are self-closing (`<br/>`).
>    e. **After markdown conversion:** Replace placeholders with `<ruby>kanji<rt>reading</rt></ruby>` tags.
> 4. Parse the original XHTML with BeautifulSoup using the `lxml-xml` parser (not `lxml` — the HTML-mode parser strips XML declarations and breaks self-closing tag syntax, producing non-XHTML output).
> 5. Find the `<body>` element.
> 6. Clear the body's children.
> 7. Parse the translated HTML into a soup and transplant its children into the original body.
> 8. Serialize back to a string, preserving the XML declaration and doctype if present.
> 9. Add to `modified_files` dict keyed by the ZIP-absolute path.
> 
> **Important:** preserve the original `<body>` tag's attributes (class, id, epub:type, xml:lang, etc.). Only replace the children (content), not the tag itself.
> 
> **Important:** preserve the original `<head>` entirely. This keeps CSS links, charset meta tags, viewport settings, and any publisher-specific head content intact.
> 
> For spine items where `chunk_count == 0` or `chunk_count is None` (illustrations, auto-toc, etc.): leave completely untouched. They copy through at the ZIP level unchanged.
> 
> ---
> 
> ## ToC translation (`toc.py`)
> 
> EPUBs have two ToC mechanisms. Most have both for compatibility:
> 
> **EPUB 2 — `toc.ncx`:** XML file with `<navPoint>` elements containing `<navLabel><text>` title strings and `<content src="..."/>` references.
> 
> **EPUB 3 — `nav.xhtml`:** XHTML file with `<nav epub:type="toc">` containing an `<ol>` of `<li><a href="...">title</a></li>` entries.
> 
> **Translation flow:**
> 
> 1. Find the ToC file(s) in the source EPUB. The OPF identifies the NCX via the `<spine toc="ncx_id">` attribute and the nav document via `<item properties="nav">`.
> 2. Extract all title strings from both ToC files using `get_text()` on each title element. This produces simple text strings even if the original had internal markup (nested spans, etc.). Deduplicate (they usually have the same titles).
> 3. Make one LLM call to translate all titles as a batch:
> 
>    ```
>    System: "Translate these Japanese chapter/section titles to English. 
>    Use the glossary for proper nouns. Maintain the same style and tone 
>    as the original titles. Return a JSON array of translated titles 
>    in the exact same order as the input."
>    
>    Glossary: [rendered glossary — names and places relevant to titles]
>    
>    User: ["第一章　灼熱の血の再会", "第二章　自称英雄ナツキ・スバル", ...]
>    ```
> 
>    Response: `["Chapter 1: A Reunion Akin to Scorching Blood", ...]`
> 
> 4. Use `LLMClient.complete_json()` with a Pydantic model validating the response is a list of strings with the correct length.
> 5. Write translated titles back into both ToC files:
>    - NCX: replace each `<navLabel><text>` content with a single text node.
>    - Nav XHTML: replace each `<a>` content with a single text node (any internal markup like spans is replaced by the flat translated string).
>    - Preserve all `src`/`href` attributes and nesting structure.
> 6. Add both modified ToC files to `modified_files`.
> 
> **Edge cases:**
> - EPUB has only NCX, no nav document: translate NCX only.
> - EPUB has only nav, no NCX: translate nav only.
> - Nested ToC entries (parts containing chapters): preserve nesting, translate all levels.
> - ToC entries that are already in the target language (e.g., "Prologue"): the translator should pass them through unchanged. Include this instruction in the prompt.
> - EPUB with no `toc.ncx` and no nav document: rebuild succeeds, ToC translation is a no-op with a warning.
> 
> **Prompt template (`prompts/translate_toc.txt`):** variables `{glossary}`, `{titles_json}`.
> 
> ---
> 
> ## Metadata updates
> 
> Parse the OPF file from the source EPUB. Modify:
> 
> - `<dc:language>`: set to `config.languages.target` (e.g., `en`).
> - `<dc:title>`: append `config.output.title_suffix` (default: `" (English Translation)"`).
> - Optionally add or update `<dc:description>` with a machine translation note: "Machine translated by dao-bridge-translator using {model_name}. Not professionally edited."
> 
> Do NOT modify `<dc:identifier>` by default. Configurable via `config.output.new_identifier` — if true, generate a new UUID and replace.
> 
> Add the modified OPF to `modified_files`.
> 
> ---
> 
> ## CSS handling
> 
> By default, the original EPUB's CSS is preserved (it copies through untouched at the ZIP level). Since we're replacing only `<body>` children and keeping the original `<head>` with its CSS links, the original styles apply to our translated content.
> 
> In practice, some original CSS class names won't match our generated HTML (we stripped them during cleaning). This means some styling may be lost (e.g., first-paragraph drop caps, specific character-dialogue formatting). This is acceptable — the base typography (fonts, margins, line-height) still works because those are typically applied to `body`, `p`, `h1`-`h6`, etc.
> 
> Include a minimal `src/dao_bridge/templates/default.css` with basic styling for:
> - `ruby` / `rt` elements (for any restored ruby text)
> - `hr` (scene breaks — clean centered styling)
> - Basic body/paragraph typography as a fallback
> 
> This CSS is only injected if `config.output.css` is set to `"default"`. When set to `"original"` (the default), no CSS is added or modified. When set to `"default"`, the `default.css` is added to the EPUB and a `<link>` is added to the `<head>` of each modified XHTML file.
> 
> ---
> 
> ## Optional epubcheck validation
> 
> If `config.output.validate` is true: after writing the output EPUB, check if `epubcheck` is available on PATH. If found, run it via subprocess on the output file. Log the result (pass/fail and any warnings). If not found, log a warning ("epubcheck not found on PATH, skipping validation") and continue. This is a non-blocking safety net — validation failure does not prevent the EPUB from being written.
> 
> ---
> 
> ## rebuild.py module structure
> 
> Functions:
> 
> - `run_rebuild_stage(work_dir: Path, config, force: bool = False) -> None`
>   - Main stage runner. Validates all translated items are assembled, then builds output EPUB.
> 
> - `build_modified_files(manifest: Manifest, work_dir: Path, source_epub_path: str, config) -> dict[str, bytes]`
>   - For each translated spine item: load assembled markdown, convert to HTML, replace body in original XHTML. Returns dict of modified ZIP paths → content.
> 
> - `replace_xhtml_body(original_xhtml: str, translated_markdown: str) -> str`
>   - Core body-replacement function. Handles ruby placeholder strategy, markdown→HTML conversion, scene break conversion, body child replacement. Uses `lxml-xml` parser.
> 
> - `markdown_to_html(md_text: str) -> str`
>   - Converts markdown to HTML with the ruby-safe pipeline (placeholder → convert → restore).
> 
> - `restore_ruby_tags(html: str, placeholder_map: dict) -> str`
>   - Replaces `RUBY_NNNN` placeholders with `<ruby>` tags.
> 
> - `write_epub_modified_copy(source_epub: str, output_epub: str, modified_files: dict[str, bytes]) -> None`
>   - ZIP-level copy with file replacement. Preserves mimetype as first entry, uncompressed. Preserves per-entry ZipInfo.
> 
> - `resolve_zip_path(opf_dir: str, href: str) -> str`
>   - Resolves OPF-relative href to ZIP-absolute path.
> 
> ## toc.py module structure
> 
> Functions:
> 
> - `find_toc_files(book) -> tuple[str | None, str | None]`
>   - Returns (ncx_zip_path, nav_zip_path) from the OPF. Either may be None.
> 
> - `extract_toc_titles(toc_content: str, toc_type: str) -> list[str]`
>   - Parses NCX or nav XHTML, extracts title strings (via `get_text()`) in order. `toc_type` is "ncx" or "nav".
> 
> - `translate_titles(titles: list[str], glossary: Glossary, llm_client: LLMClient, config) -> list[str]`
>   - One LLM call via `complete_json()`, returns translated titles in same order.
> 
> - `apply_translated_titles(toc_content: str, toc_type: str, translated_titles: list[str]) -> str`
>   - Writes translated titles back into the ToC XML/XHTML as single text nodes, preserving structure.
> 
> - `translate_toc(source_epub_path: str, glossary: Glossary, llm_client: LLMClient, config) -> dict[str, bytes]`
>   - Orchestrates: find → extract → translate → apply. Returns modified ToC files for the `modified_files` dict.
> 
> - `update_opf_metadata(opf_content: str, config) -> str`
>   - Modifies language, title, description in OPF XML. Returns modified OPF string.
> 
> ---
> 
> ## CLI
> 
> ```
> dao-bridge rebuild [--work-dir ./work] [--force] [--verbose]
> ```
> 
> Validates before running:
> - All translatable spine items have corresponding `assembled/NNN.md` files.
> - Source EPUB is accessible.
> - Output path is writable.
> 
> Errors clearly if any assembled files are missing, listing which spines are incomplete.
> 
> ## Update `run` command
> 
> The `run` command should now chain all implemented stages:
> 
> ```
> dao-bridge run <epub> [--config config.yaml] [--work-dir ./work] [--verbose]
> ```
> 
> Equivalent to executing in order:
> ```
> init → extract → clean → classify → chunk → glossary-build → 
> glossary-reconcile → glossary-crosscheck (if master configured) → 
> translate → assemble → rebuild
> ```
> 
> Each stage checks state and skips if already complete. **If any stage fails, `run` stops immediately and reports which stage failed and the error.** User fixes the issue, re-runs, and it picks up where it left off.
> 
> Stages NOT included in `run` (manual/utility commands):
> - `glossary-import-reference` — pre-pipeline setup
> - `glossary-promote` — post-pipeline
> - `glossary-export` — utility
> - `audit` — post-pipeline (future feature, not implemented yet)
> 
> ---
> 
> ## State tracking
> 
> `rebuild` is a run-level stage (not per-item — it either succeeds or fails as a whole). Mark started/completed/failed.
> 
> Re-running rebuild retries ToC translation without redoing body replacement, since body replacement is deterministic from assembled files (but in practice, re-running from scratch is fast enough that this optimization is optional).
> 
> ---
> 
> ## Tests
> 
> **Body replacement:**
> - Plain prose markdown → XHTML body replacement. Original `<head>` preserved. Original `<body>` attributes preserved.
> - Headings, bold, italic in markdown → correct HTML tags in output.
> - Scene break markers → `<hr/>` in output.
> - `{kanji|reading}` notation: placeholder strategy prevents markdown `attr_list` mangling. Final output contains `<ruby>kanji<rt>reading</rt></ruby>`.
> - `<br>` handling: markdown hard line breaks produce `<br/>` in XHTML.
> - Original XHTML with complex `<head>` (multiple CSS links, meta tags, scripts): head preserved intact after body replacement.
> - Original XHTML with `<body class="chapter" epub:type="bodymatter">`: body attributes preserved, only children replaced.
> 
> **Passthrough items:**
> - Illustration spine items appear unchanged in output EPUB.
> - Multiple consecutive illustrations all present and in correct order.
> - Auto-toc items unchanged.
> 
> **ZIP-level integrity:**
> - All files from source EPUB present in output EPUB.
> - Unmodified files are byte-identical between source and output.
> - Modified files have new content.
> - `mimetype` file is first entry, uncompressed (`compress_type=ZIP_STORED`), no extra field (EPUB spec requirement).
> - Per-entry compress_type preserved from source.
> 
> **ToC translation:**
> - NCX: titles replaced with single text nodes, structure and hrefs preserved.
> - Nav XHTML: titles replaced with single text nodes, structure and hrefs preserved.
> - Nested ToC entries: all levels translated.
> - Mixed-language titles: already-English titles passed through unchanged.
> - EPUB with only NCX: works without nav.
> - EPUB with only nav: works without NCX.
> - LLM returns wrong number of titles: error, not silent corruption.
> - ToC entries with internal markup (nested spans): `get_text()` extracts clean text, write-back uses single text node.
> 
> **Metadata:**
> - Language updated in OPF.
> - Title suffix appended.
> - Translation note added to description when configured.
> - Identifier unchanged by default.
> - Identifier replaced with UUID when `new_identifier: true`.
> 
> **CSS:**
> - `css: "original"`: no CSS modifications, no new files added.
> - `css: "default"`: `default.css` added to EPUB, `<link>` added to each modified XHTML `<head>`.
> 
> **Validation:**
> - `validate: true` with epubcheck available: runs and logs result.
> - `validate: true` without epubcheck on PATH: logs warning, continues.
> - `validate: false`: no validation attempted.
> 
> **Integration:**
> - Full pipeline with mini EPUB fixture and mocked LLM: init through rebuild produces a valid output EPUB.
> - Output EPUB can be read back by ebooklib (round-trip validation).
> - Spine order in output matches source.
> 
> **Edge cases:**
> - Source EPUB with unusual ZIP structure (extra directories, non-standard paths): copies through intact.
> - Very small EPUB (one chapter, no images, no ToC): works.
> - EPUB with no `toc.ncx` and no nav document: rebuild succeeds, ToC translation is no-op with a warning.
> - OPF in subdirectory (e.g., `OEBPS/content.opf`): href resolution produces correct ZIP paths.
> 
> **`run` command:**
> - Full pipeline end-to-end with mini EPUB and mocked LLM.
> - Stage failure halts run, reports error, re-run resumes from failed stage.
> 
> ---
> 
> ## Deliverables
> 
> - `src/dao_bridge/rebuild.py` with full implementation and docstrings.
> - `src/dao_bridge/toc.py` with full implementation and docstrings.
> - `src/dao_bridge/templates/default.css` — minimal fallback CSS.
> - `src/dao_bridge/prompts/translate_toc.txt` — ToC title translation prompt.
> - Updated `src/dao_bridge/cli.py` with `rebuild` command and fully-implemented `run` command chaining all stages. `run` stops on first stage failure.
> - All tests passing.
> - Updated `README.md` with rebuild documentation, output config options, epubcheck integration, full pipeline usage guide, and complete `dao-bridge run` quickstart from source EPUB to translated EPUB.
