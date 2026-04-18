"""Tests for dao_bridge.config — YAML loading and Pydantic validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dao_bridge.config import ChunkingConfig, LLMConfig, ModelConfig, load_config

# ---------------------------------------------------------------------------
# Minimal config loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_minimal_config(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            'source_epub: "/path/to/book.epub"\nwork_dir: "./work"\n', encoding="utf-8"
        )
        cfg = load_config(cfg_file)
        assert cfg.source_epub == "/path/to/book.epub"
        assert cfg.work_dir == "./work"
        # Defaults should be populated
        assert cfg.languages.source == "ja"
        assert cfg.languages.target == "en"
        assert cfg.llm.max_retries == 3

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_file_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_config(cfg_file)

    def test_invalid_yaml_missing_required(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("work_dir: ./work\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# Full config with all sections
# ---------------------------------------------------------------------------


class TestFullConfig:
    @pytest.fixture
    def full_config_file(self, tmp_path: Path) -> Path:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            """\
source_epub: "/path/to/book.jp.epub"
work_dir: "./work"

models:
  classify:
    base_url: "http://localhost:8080/v1"
    api_key: "not-needed"
    model: "qwen3-30b-a3b"
    temperature: 0.0
  glossary:
    base_url: "http://localhost:8080/v1"
    api_key: "not-needed"
    model: "gemma-4-26b-a4b"
    temperature: 0.2
  translate:
    base_url: "http://localhost:8080/v1"
    api_key: "not-needed"
    model: "gemma-4-26b-a4b"
    temperature: 0.3
  summarize:
    base_url: "http://localhost:8080/v1"
    api_key: "not-needed"
    model: "qwen3-30b-a3b"
    temperature: 0.2

chunking:
  target_tokens: 2000
  max_tokens: 2400
  min_chunk_tokens: 400

glossary:
  categories:
    - character
    - place
  master_glossary_path: "../master.json"

translation_phase:
  double_pass: true
  rolling_summary: true

output:
  epub_path: "./book.en.epub"

languages:
  source: "ja"
  target: "en"

llm:
  max_retries: 5
  retry_backoff_seconds: 3
  request_timeout_seconds: 600
""",
            encoding="utf-8",
        )
        return cfg_file

    def test_full_load(self, full_config_file: Path):
        cfg = load_config(full_config_file)
        assert cfg.models.classify.model == "qwen3-30b-a3b"
        assert cfg.models.translate.temperature == 0.3
        assert cfg.models.summarize is not None
        assert cfg.models.summarize.model == "qwen3-30b-a3b"
        assert cfg.chunking.target_tokens == 2000
        assert cfg.glossary.categories == ["character", "place"]
        assert cfg.glossary.master_glossary_path == "../master.json"
        assert cfg.translation_phase.double_pass is True
        assert cfg.output.epub_path == "./book.en.epub"
        assert cfg.llm.max_retries == 5
        assert cfg.llm.request_timeout_seconds == 600

    def test_summarize_fallback(self, tmp_path: Path):
        """When summarize is absent, summarize_model() returns translate config."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            """\
source_epub: "/path/to/book.epub"
models:
  translate:
    model: "my-translate-model"
""",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.models.summarize is None
        assert cfg.summarize_model().model == "my-translate-model"


# ---------------------------------------------------------------------------
# Individual model validation
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig()
        assert m.base_url == "http://localhost:8080/v1"
        assert m.api_key == "not-needed"

    def test_custom_values(self):
        m = ModelConfig(base_url="https://api.openai.com/v1", api_key="sk-xxx", model="gpt-4")
        assert m.model == "gpt-4"


class TestChunkingConfig:
    def test_defaults(self):
        c = ChunkingConfig()
        assert c.target_tokens == 2000
        assert c.max_tokens == 2400
        assert len(c.scene_break_patterns) > 0
        assert c.normalize_scene_breaks == "* * *"
        assert "chapter" in c.chunkable_classifications


class TestLLMConfig:
    def test_defaults(self):
        llm = LLMConfig()
        assert llm.max_retries == 3
        assert llm.retry_backoff_seconds == 2.0
        assert llm.request_timeout_seconds == 300.0


# ---------------------------------------------------------------------------
# work_dir_path / source_epub_path properties
# ---------------------------------------------------------------------------


class TestConfigPaths:
    def test_work_dir_path_resolved(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('source_epub: "book.epub"\nwork_dir: "./work"\n', encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.work_dir_path.is_absolute()

    def test_source_epub_path_resolved(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('source_epub: "book.epub"\nwork_dir: "./work"\n', encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.source_epub_path.is_absolute()
