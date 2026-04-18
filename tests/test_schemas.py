"""Tests for dao_bridge.schemas — Pydantic model round-trips and validation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dao_bridge.schemas import (
    Chunk,
    Glossary,
    GlossaryEntry,
    Manifest,
    ManifestItem,
    TranslatedChunk,
)

# ---------------------------------------------------------------------------
# ManifestItem
# ---------------------------------------------------------------------------


class TestManifestItem:
    def test_nullable_fields_default_none(self):
        item = ManifestItem(spine_index=0, original_href="x", raw_path="r")
        assert item.clean_path is None
        assert item.classification is None
        assert item.title is None
        assert item.token_count is None
        assert item.paragraph_count is None
        assert item.chunk_count is None

    def test_classification_literal_valid(self):
        item = ManifestItem(
            spine_index=0, original_href="x", raw_path="r", classification="chapter"
        )
        assert item.classification == "chapter"

    def test_classification_literal_invalid(self):
        with pytest.raises(ValidationError):
            ManifestItem(
                spine_index=0, original_href="x", raw_path="r", classification="invalid_type"
            )

    def test_round_trip_json(self):
        item = ManifestItem(
            spine_index=7,
            original_href="chapter007.xhtml",
            raw_path="raw/007.xhtml",
            clean_path="clean/007.md",
            classification="chapter",
            title="Chapter 7",
            token_count=1500,
            paragraph_count=30,
        )
        data = item.model_dump()
        restored = ManifestItem(**data)
        assert restored.spine_index == 7
        assert restored.classification == "chapter"
        assert restored.token_count == 1500


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_minimal_manifest(self):
        m = Manifest(source_epub_path="/books/test.epub", book_id="test-book")
        assert m.spine == []
        assert m.images == []
        assert m.metadata == {}
        assert m.opf_dir == ""
        assert m.spine_padding_width == 4

    def test_spine_padding_width_custom(self):
        m = Manifest(
            source_epub_path="/books/test.epub",
            book_id="test-book",
            spine_padding_width=5,
        )
        assert m.spine_padding_width == 5

    def test_spine_padding_width_round_trip(self):
        m = Manifest(
            source_epub_path="/books/test.epub",
            book_id="test-book",
            spine_padding_width=5,
        )
        data = m.model_dump(mode="json")
        restored = Manifest(**data)
        assert restored.spine_padding_width == 5

    def test_round_trip_json(self):
        m = Manifest(
            source_epub_path="/books/test.epub",
            book_id="test-book",
            opf_dir="OEBPS",
            spine_padding_width=4,
            spine=[
                ManifestItem(spine_index=0, original_href="ch.xhtml", raw_path="raw/0000.xhtml")
            ],
            images=["images/cover.jpg"],
            metadata={"title": "Test Book", "author": "Author"},
        )
        data = m.model_dump()
        restored = Manifest(**data)
        assert len(restored.spine) == 1
        assert restored.spine[0].spine_index == 0
        assert restored.spine_padding_width == 4
        assert restored.images == ["images/cover.jpg"]
        assert restored.metadata["title"] == "Test Book"
        assert restored.opf_dir == "OEBPS"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_basic_construction(self):
        c = Chunk(
            chunk_id="003.015",
            spine_index=3,
            chunk_index=15,
            source_file="clean/003.md",
            block_range=(0, 10),
            token_count=1800,
            text="some text",
        )
        assert c.chunk_id == "003.015"
        assert c.extended_for_remainder is False
        assert c.ends_at_scene_break is False

    def test_round_trip(self):
        c = Chunk(
            chunk_id="001.001",
            spine_index=1,
            chunk_index=1,
            source_file="clean/001.md",
            block_range=(5, 20),
            token_count=2000,
            extended_for_remainder=True,
            text="text here",
            ends_at_scene_break=True,
        )
        data = c.model_dump()
        restored = Chunk(**data)
        assert restored.extended_for_remainder is True
        assert restored.block_range == (5, 20)
        assert restored.ends_at_scene_break is True


# ---------------------------------------------------------------------------
# GlossaryEntry / Glossary
# ---------------------------------------------------------------------------


class TestGlossaryEntry:
    def test_minimal_entry(self):
        e = GlossaryEntry(english="Subaru", category="character", source="extracted")
        assert e.source_term is None
        assert e.aliases == []
        assert e.nicknames == {}

    def test_full_entry(self):
        e = GlossaryEntry(
            source_term="スバル",
            reading="すばる",
            english="Subaru",
            category="character",
            first_seen_chunk="001.001",
            aliases=["Natsuki Subaru"],
            nicknames={"Rem": "Subaru-kun"},
            speech_style="Casual, uses slang",
            notes="Main character",
            source="seed",
            source_books=["isbn-9784041234567"],
        )
        assert e.source_term == "スバル"
        assert e.nicknames["Rem"] == "Subaru-kun"
        assert e.source_books == ["isbn-9784041234567"]

    def test_invalid_source_literal(self):
        with pytest.raises(ValidationError):
            GlossaryEntry(english="X", category="character", source="invalid")

    def test_round_trip(self):
        e = GlossaryEntry(
            source_term="エミリア",
            english="Emilia",
            category="character",
            source="extracted",
            notes="Half-elf",
            source_books=["rezero-vol5", "rezero-vol6"],
        )
        data = e.model_dump()
        restored = GlossaryEntry(**data)
        assert restored.source_term == "エミリア"
        assert restored.notes == "Half-elf"
        assert restored.source_books == ["rezero-vol5", "rezero-vol6"]


class TestGlossary:
    def test_empty_glossary(self):
        g = Glossary()
        assert g.entries == []
        assert g.version == 1
        assert g.book_id is None

    def test_glossary_with_entries(self):
        g = Glossary(
            entries=[
                GlossaryEntry(english="Subaru", category="character", source="seed"),
                GlossaryEntry(
                    source_term="ルグニカ", english="Lugunica", category="place", source="extracted"
                ),
            ],
            book_id="rezero-5",
            book_metadata={"title": "Re:Zero Vol 5"},
        )
        assert len(g.entries) == 2
        assert g.book_id == "rezero-5"

    def test_round_trip_json(self):
        now = datetime.now(timezone.utc)
        g = Glossary(
            entries=[
                GlossaryEntry(english="Test", category="term", source="user"),
            ],
            version=3,
            book_id="test-book",
            created_at=now,
            updated_at=now,
        )
        data = g.model_dump(mode="json")
        restored = Glossary(**data)
        assert restored.version == 3
        assert len(restored.entries) == 1
        assert restored.book_id == "test-book"


# ---------------------------------------------------------------------------
# TranslatedChunk
# ---------------------------------------------------------------------------


class TestTranslatedChunk:
    def test_basic_construction(self):
        tc = TranslatedChunk(
            chunk_id="001.001",
            source_text="日本語テキスト",
            pass1_translation="English text pass 1",
            translated_text="English text final",
            pass_count=2,
            total_attempts=1,
            model_used="test-model",
        )
        assert tc.qa_result is None
        assert tc.qa_issues == []
        assert tc.overlap_chunk_id is None
        assert tc.token_usage == {}

    def test_qa_result_valid_literals(self):
        for val in ("pass", "fail", None):
            tc = TranslatedChunk(
                chunk_id="001.001",
                source_text="src",
                pass1_translation="p1",
                translated_text="final",
                pass_count=1,
                total_attempts=1,
                qa_result=val,
            )
            assert tc.qa_result == val

    def test_qa_result_invalid_literal(self):
        with pytest.raises(ValidationError):
            TranslatedChunk(
                chunk_id="001.001",
                source_text="src",
                pass1_translation="p1",
                translated_text="final",
                pass_count=1,
                total_attempts=1,
                qa_result="maybe",
            )

    def test_round_trip_json(self):
        tc = TranslatedChunk(
            chunk_id="003.015",
            source_text="ソーステキスト",
            pass1_translation="Pass 1",
            translated_text="Final translation",
            pass_count=2,
            qa_result="pass",
            qa_issues=[],
            total_attempts=2,
            overlap_chunk_id="003.014",
            summary_generated="Something happened",
            token_usage={"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800},
            model_used="gemma-4",
            duration_seconds=12.5,
        )
        data = tc.model_dump(mode="json")
        restored = TranslatedChunk(**data)
        assert restored.chunk_id == "003.015"
        assert restored.token_usage["total_tokens"] == 800
        assert restored.duration_seconds == 12.5
