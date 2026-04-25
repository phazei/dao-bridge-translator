# dao-bridge-translator

AI translation pipeline for EPUB novels using LLM APIs.

Translates EPUB files through a multi-stage pipeline: extraction, cleaning,
classification, chunking, glossary building, reconciliation, translation,
assembly, and EPUB rebuild. Language-agnostic (source/target configured in
`config.yaml`). Designed to work with any OpenAI-compatible API (local
llama-server, vLLM, LM Studio, OpenAI, Claude, OpenRouter, etc.).

## Installation

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
```

## Quickstart

Translate an EPUB from start to finish in two commands:

```bash
# 1. Initialise a work directory from an EPUB
dao-bridge init /path/to/book.epub --work-dir ./work

# 2. Run the full pipeline (extract through rebuild)
dao-bridge run --work-dir ./work
```

The `run` command chains all stages automatically.  If interrupted, re-run
to pick up where it left off.  Each stage is idempotent and skips completed
work.

### Step-by-step (individual commands)

```bash
# 1. Initialise a work directory from an EPUB
dao-bridge init /path/to/book.epub --work-dir ./work

# 2. Extract spine items from the EPUB
dao-bridge extract --work-dir ./work

# 3. Clean XHTML to markdown
dao-bridge clean --work-dir ./work

# 4. Classify spine items (structural hints + LLM fallback)
dao-bridge classify --work-dir ./work

# 5. Chunk cleaned markdown into translation-ready segments
dao-bridge chunk --work-dir ./work

# 6. Build a per-book glossary from chunked text
dao-bridge glossary-build --work-dir ./work

# 7. Resolve within-book glossary conflicts
dao-bridge glossary-reconcile --work-dir ./work

# 8. Export glossary for human review
dao-bridge glossary-export --work-dir ./work

# 9. Translate all chunks (LLM-powered)
dao-bridge translate --work-dir ./work

# 10. Assemble translated chunks into per-spine markdown
dao-bridge assemble --work-dir ./work

# 11. Build the output EPUB
dao-bridge rebuild --work-dir ./work

# Check pipeline status at any time
dao-bridge status --work-dir ./work
```

Each command is idempotent: re-running skips completed work unless `--force`
is passed. Add `--verbose` to any command for DEBUG-level console output.

## Commands

| Command | Description |
|---------|-------------|
| `init <epub>` | Create work directory, write default `config.yaml` |
| `extract` | Extract EPUB spine items to `raw/NNN.xhtml` |
| `clean` | Convert raw XHTML to markdown in `clean/NNN.md` |
| `classify` | Classify spine items (chapter, frontmatter, illustration, etc.) |
| `chunk` | Chunk cleaned markdown into `chunks/NNN/NNN.MMM.json` |
| `glossary-build` | Extract per-book glossary from chunked source text (spine-aligned batches) |
| `glossary-reconcile` | Resolve within-book glossary conflicts via LLM |
| `glossary-export` | Export glossary as human-readable markdown |
| `translate` | Translate all chunks using LLM (double-pass, QA) |
| `assemble` | Reassemble translated chunks into `assembled/NNN.md` |
| `rebuild` | Build output EPUB from assembled translations |
| `run` | Chain all stages (extract through rebuild) |
| `status` | Display pipeline stage completion status |

The `classify`, `chunk`, `glossary-build`, and `assemble` commands support
`--spine N` to process a single spine item, and `--force` to reprocess even
if already complete.  `glossary-build` also supports `--batch ID` to redo a
specific sub-batch (e.g. `--batch 0003.b2`); `--batch` takes precedence over
`--spine`.

The `translate` command supports `--spine N`, `--chunk ID`, `--from/--to` for
range-based translation, and `--force` to retranslate completed chunks.

### Retrying Failed Items

If some items fail during a stage (e.g., LLM errors during classification or
translation), a plain re-run will automatically retry them -- as long as the
stage has not been marked `completed`.

If the stage *did* complete with some items failed, use `--retry-failed` to
re-enter the stage and retry only the failed items without reprocessing
everything:

```bash
# Retry only failed items in a completed classify stage
dao-bridge classify --work-dir ./work --retry-failed

# Retry only failed chunks in translate
dao-bridge translate --work-dir ./work --retry-failed

# Retry failed items across all stages
dao-bridge run --work-dir ./work --retry-failed
```

`--retry-failed` is mutually exclusive with `--force`. It is supported on
`classify`, `chunk`, `glossary-build`, `glossary-reconcile`, `translate`,
`assemble`, and `run`.

### Glossary Flow

The glossary stages extract and refine a per-book glossary of proper nouns,
character names, and notable terms:

1. **glossary-build** -- Groups chunks by spine item and packs each spine's
   chunks into sub-batches (item IDs like `0003.b2`).  Each sub-batch is sent
   to the LLM for extraction.  Entries accumulate across batches; the glossary
   is saved after each batch for crash-resumability.  Use `--spine N` or
   `--batch ID` to redo specific items.  Conflicting English proposals and
   corrections are logged for the reconcile stage.

2. **glossary-reconcile** -- Resolves within-book conflicts (differing English
   translations, corrections) via LLM calls, and consolidates multiple
   speech-style observations per character. Writes a decision report to
   `glossary_reconcile_report.md`.

3. **glossary-export** -- Renders the glossary as categorized markdown
   (`glossary.md`) for human review and editing before the translation stage.

The intended workflow is: build -> reconcile -> export -> **human review and
editing of glossary.json** -> translate. Human edits to `glossary.json` should
set `"source": "user"` on modified entries to prevent the build stage from
overwriting them.

**Master glossary features** (`glossary-crosscheck`, `glossary-promote`,
`glossary-import-reference`) for multi-book series with consistent terminology
are planned for a future release.

### Translation

The `translate` command runs a multi-pass translation pipeline for each chunk:

1. **Pass 1** -- Initial translation with glossary injection, overlap context
   from the previous chunk, and rolling narrative summary for continuity.
2. **Pass 2** (optional) -- Revision pass comparing the draft against the
   original, with instructions to improve naturalness and accuracy.
3. **QA** (optional) -- Programmatic length-ratio check plus LLM-based quality
   assessment.  On QA failure after retries, the pipeline halts for manual
   intervention.

Rolling summaries are generated after each chunk for story continuity across
the book.

### Rebuild

The `rebuild` command produces the output EPUB:

- **Modified copy approach** -- copies the source EPUB at the ZIP level,
  replacing only translated XHTML body content, ToC entries, and metadata.
  Preserves all original structure: images, fonts, CSS, DRM metadata, and
  everything else.
- **ToC translation** -- Translates chapter/section titles in both
  `toc.ncx` (EPUB 2) and `nav.xhtml` (EPUB 3) via a single LLM call.
- **Metadata updates** -- Sets language to target, appends title suffix,
  optionally adds a machine-translation note and new identifier.
- **CSS options** -- By default preserves original CSS (`css: original`).
  Set `css: default` to inject a minimal fallback stylesheet.
- **Validation** -- Optionally runs `epubcheck` if available on PATH.

### Manual Classification Override

After running `classify`, review `manifest.json` to check the results.
To manually override a classification, edit the `classification` field on
any spine item directly.  Re-running `classify` without `--force` will
preserve your edits and only classify items that still have `null`
classification.  Use `--force` to discard all manual edits and reclassify
from scratch.

## Work Directory Layout

```
work/
  config.yaml          # Pipeline configuration
  manifest.json        # Book metadata, spine items, counts
  state.json           # Pipeline progress tracking
  raw/                 # Extracted XHTML (one per spine item)
    0000.xhtml
    0001.xhtml
    ...
  clean/               # Cleaned markdown (one per spine item)
    0000.md
    0001.md
    ...
  chunks/              # Chunked content for translation
    0000/               #   Per-spine chunk directories
      0000.001.json     #     Chunk JSON (Chunk schema)
      0000.002.json
    0001/
      0001.001.json
  translations/        # Per-chunk translation results
    0000/
      0000.001.json    #     TranslatedChunk JSON
  assembled/           # Reassembled translated markdown
    0000.md
    0001.md
  summaries/           # Rolling translation summaries
    rolling_summary.json
  glossary.json        # Per-book glossary (build -> reconcile -> user edit)
  glossary.md          # Exported glossary for human review
  glossary_reconcile_report.md  # Reconciliation decisions and reasoning
  logs/
    run.log            # Full debug log
```

## Multi-Book Layout

For series with a shared glossary, use sibling work directories:

```
translations/
  master_glossary.json         # Shared across volumes
  rezero-vol1/
    config.yaml                # glossary.master_glossary_path: "../master_glossary.json"
    manifest.json
    state.json
    raw/
    clean/
    ...
  rezero-vol2/
    config.yaml
    ...
  rezero-vol3/
    config.yaml
    ...
```

Set `glossary.master_glossary_path` in each volume's `config.yaml` to point
at the shared master (absolute or relative to the work directory).

## Configuration

The `config.yaml` file controls all pipeline parameters. A default is generated
by `dao-bridge init`. Key sections:

- **models**: Per-task LLM endpoints (classify, glossary, translate, summarize)
- **chunking**: Token targets, scene break patterns
- **glossary**: Categories, master glossary path, crosscheck settings
- **translation_phase**: Double-pass, overlap, QA settings
- **output**: EPUB output path, metadata options
- **languages**: Source and target language codes
- **llm**: Global retry and timeout settings

### Output Configuration

The `output` section controls the rebuilt EPUB:

```yaml
output:
    epub_path: ./book.en.epub           # Output file path (relative to work dir parent)
    title_suffix: ' (English Translation)'  # Appended to the book title
    new_identifier: false               # Generate new UUID for dc:identifier
    css: original                       # 'original' (keep source CSS) or 'default' (inject fallback)
    add_translation_note: true          # Add machine-translation note to dc:description
    validate: false                     # Run epubcheck on output (if available on PATH)
```

### epubcheck Integration

If `validate: true` is set and `epubcheck` is on your PATH, the rebuild
stage will run it on the output EPUB and log the results.  Validation failure
is logged as a warning but does not prevent the EPUB from being written.

Install epubcheck from https://github.com/w3c/epubcheck.

See `src/dao_bridge/config.py` for the full schema with defaults.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with verbose output
pytest -v

# Lint
ruff check src/ tests/
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
