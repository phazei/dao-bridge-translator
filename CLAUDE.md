# CLAUDE.md — dao-bridge-translator

## Project Overview

AI translation pipeline for EPUB novels using LLM APIs. Translates EPUBs
through a multi-stage pipeline: extraction, cleaning, classification, chunking,
glossary building/clustering/reconciliation, translation, assembly, and EPUB
rebuild.

**Language-agnostic by design** — source and target languages are configured in
`config.yaml` (default: `ja` -> `en`). Works with any OpenAI-compatible API.

## Tech Stack

- **Language:** Python 3.12+
- **Build:** setuptools via `pyproject.toml`
- **CLI framework:** Click (`dao-bridge` command)
- **Data models:** Pydantic v2
- **LLM client:** OpenAI SDK (compatible with any OpenAI-format API)
- **Key libraries:** ebooklib, beautifulsoup4, lxml, markdownify, tiktoken, jellyfish, rich
- **Linter:** Ruff (`ruff check src/ tests/`)
- **Tests:** pytest
- **License:** Apache 2.0

## Repository Layout

```
src/dao_bridge/           # Main package (19 modules)
  cli.py                  # Click CLI entry point
  config.py               # Pydantic config schema (reads config.yaml)
  schemas.py              # Core data models (Glossary, GlossaryEntity, SurfaceForm, Chunk, etc.)
  state.py                # Pipeline progress/state tracking (state.json)
  workdir.py              # Work directory management
  extract.py              # EPUB spine extraction -> raw XHTML
  clean.py                # XHTML -> Markdown conversion
  classify.py             # Spine item classification (chapter, frontmatter, etc.)
  chunk.py                # Markdown chunking for translation-sized segments
  glossary.py             # Glossary building, entity extraction, reconciliation (~2600 lines)
  glossary_clustering.py  # Duplicate entity detection and merging
  similarity.py           # String similarity (Jaro-Winkler) for glossary matching
  translate.py            # Multi-pass LLM translation (pass1, pass2, QA)
  assemble.py             # Reassemble translated chunks into per-spine markdown
  rebuild.py              # Output EPUB construction
  toc.py                  # Table-of-contents translation
  llm_client.py           # OpenAI-compatible LLM API client
  logging.py              # Logging configuration
  prompts/                # 13 LLM prompt templates (.txt files)
  templates/              # Default CSS for rebuilt EPUBs
  lang_names.json         # ISO code -> language name mapping

tests/                    # 17 test files mirroring source modules
  fixtures/               # Test EPUBs and markdown fixtures
  conftest.py             # Shared pytest fixtures

scripts/                  # Utility scripts (benchmarks, fixture generators)
build_phases/             # Design/planning documents (not code)
```

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# CLI entry point
dao-bridge --help
```

## Pipeline Stages (in order)

1. `init` — Create work directory from EPUB, generate `config.yaml`
2. `extract` — Extract EPUB spine items to `raw/*.xhtml`
3. `clean` — Convert XHTML to markdown in `clean/*.md`
4. `classify` — Classify spine items (chapter, frontmatter, illustration, etc.)
5. `chunk` — Split markdown into translation-sized chunks
6. `glossary-build` — Extract entities from chunks via LLM
7. `glossary-cluster` — Merge duplicate entities (heuristic + LLM confirmation)
8. `glossary-reconcile` — Resolve translation conflicts via LLM
9. `glossary-export` — Export glossary as human-readable markdown
10. `translate` — Multi-pass LLM translation (pass1, pass2, QA)
11. `assemble` — Reassemble translated chunks into per-spine markdown
12. `rebuild` — Build output EPUB

All stages are idempotent and crash-resumable via `state.json`.

## Key Architecture Concepts

- **Entity-centric glossary:** `GlossaryEntity` owns multiple `SurfaceForm`
  entries. Each surface form maps a source-language string to its translation.
- **Staged glossary files:** `glossary_build.json` -> `glossary_cluster.json`
  -> `glossary.json`. Each stage reads from the previous and never mutates it.
- **LLM prompts** live in `src/dao_bridge/prompts/*.txt` and use Python
  `.format()` placeholders (`{source_language}`, `{target_language}`,
  `{chunk_batch}`, etc.).
- **Config** is a nested Pydantic model in `config.py`, loaded from `config.yaml`.

## Core Data Models (schemas.py)

- `SurfaceForm` — A source-language text form mapped to its translation
- `GlossaryEntity` — An entity (person, place, item) with surface forms, canonical name, category, metadata
- `Glossary` — Collection of entities with metadata
- `ExtractedMention` — Raw LLM extraction result before entity linking
- `Chunk` / `TranslatedChunk` — Source and translated text segments

## Conventions

- Each pipeline module has a corresponding test file (`glossary.py` -> `test_glossary.py`)
- Ruff enforces: Python 3.12 target, 100-char line length, E/F/I/W rules
- Test fixtures live in `tests/fixtures/`
- `build_phases/` contains design documents, not executable code
