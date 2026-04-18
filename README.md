# dao-bridge-translator

AI translation pipeline for Japanese light novel EPUBs using LLM APIs.

Translates Japanese EPUB files to English through a multi-stage pipeline:
extraction, cleaning, classification, glossary building, chunking, translation,
assembly, and EPUB rebuild. Designed to work with any OpenAI-compatible API
(local llama-server, vLLM, LM Studio, OpenAI, Claude, OpenRouter, etc.).

## Installation

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Initialise a work directory from a Japanese EPUB
dao-bridge init /path/to/book.jp.epub --work-dir ./work

# 2. Extract spine items from the EPUB
dao-bridge extract --work-dir ./work

# 3. Clean XHTML to markdown
dao-bridge clean --work-dir ./work

# 4. Check pipeline status
dao-bridge status --work-dir ./work
```

Each command is idempotent: re-running skips completed work unless `--force`
is passed. Add `--verbose` to any command for DEBUG-level console output.

## Currently Working Commands

| Command | Description |
|---------|-------------|
| `init <epub>` | Create work directory, write default `config.yaml` |
| `extract` | Extract EPUB spine items to `raw/NNN.xhtml` |
| `clean` | Convert raw XHTML to markdown in `clean/NNN.md` |
| `status` | Display pipeline stage completion status |

## Work Directory Layout

```
work/
  config.yaml          # Pipeline configuration
  manifest.json        # Book metadata, spine items, counts
  state.json           # Pipeline progress tracking
  raw/                 # Extracted XHTML (one per spine item)
    000.xhtml
    001.xhtml
    ...
  clean/               # Cleaned markdown (one per spine item)
    000.md
    001.md
    ...
  chunks/              # (future) Chunked content for translation
  translations/        # (future) Per-chunk translations
  assembled/           # (future) Reassembled translated markdown
  summaries/           # (future) Rolling translation summaries
  glossary.json        # (future) Per-book glossary
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
