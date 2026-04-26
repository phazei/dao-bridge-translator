"""Tests for dao_bridge.schemas — Pydantic model round-trips and validation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dao_bridge.schemas import (
    Chunk,
    ExtractedMention,
    Glossary,
    GlossaryClusterDecision,
    GlossaryClusterResponse,
    GlossaryEntity,
    GlossaryExtractionResponse,
    Manifest,
    ManifestItem,
    SurfaceForm,
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
# SurfaceForm
# ---------------------------------------------------------------------------


class TestSurfaceForm:
    def test_minimal(self):
        sf = SurfaceForm(source="スバル", english="Subaru")
        assert sf.reading is None
        assert sf.context_hints == []
        assert sf.notes is None
        assert sf.first_seen_chunk is None
        assert sf.occurrence_count == 1

    def test_full(self):
        sf = SurfaceForm(
            source="ヴィンセント",
            reading="ゔぃんせんと",
            english="Vincent",
            context_hints=["same person as アベル"],
            notes="Unmasked name",
            first_seen_chunk="0003.001",
            occurrence_count=5,
        )
        assert sf.reading == "ゔぃんせんと"
        assert sf.context_hints == ["same person as アベル"]
        assert sf.occurrence_count == 5

    def test_round_trip(self):
        sf = SurfaceForm(
            source="アベル",
            english="Abel",
            context_hints=["hint1", "hint2"],
            occurrence_count=3,
        )
        data = sf.model_dump()
        restored = SurfaceForm(**data)
        assert restored.source == "アベル"
        assert restored.context_hints == ["hint1", "hint2"]


# ---------------------------------------------------------------------------
# GlossaryEntity
# ---------------------------------------------------------------------------


class TestGlossaryEntity:
    def test_minimal(self):
        e = GlossaryEntity(
            entity_id="character_000001",
            category="character",
            canonical_english="Subaru",
        )
        assert e.surface_forms == []
        assert e.aliases == []
        assert e.nicknames == {}
        assert e.speech_style is None
        assert e.source == "extracted"
        assert e.summary is None

    def test_full(self):
        e = GlossaryEntity(
            entity_id="character_000001",
            category="character",
            canonical_english="Subaru",
            summary="A young man from another world.",
            surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
            aliases=["Natsuki Subaru"],
            nicknames={"Rem": "Subaru-kun"},
            speech_style="Casual, uses slang",
            notes="Main character",
            source="seed",
            source_books=["isbn-9784041234567"],
            first_seen_chunk="001.001",
            latest_evidence_chunk="005.003",
        )
        assert e.canonical_english == "Subaru"
        assert e.nicknames["Rem"] == "Subaru-kun"
        assert e.source_books == ["isbn-9784041234567"]
        assert e.latest_evidence_chunk == "005.003"

    def test_invalid_source_literal(self):
        with pytest.raises(ValidationError):
            GlossaryEntity(
                entity_id="x",
                category="character",
                canonical_english="X",
                source="invalid",
            )

    def test_round_trip(self):
        e = GlossaryEntity(
            entity_id="character_000001",
            category="character",
            canonical_english="Emilia",
            surface_forms=[
                SurfaceForm(source="エミリア", english="Emilia"),
                SurfaceForm(source="エミリアたん", english="Emilia-tan"),
            ],
            source="extracted",
            notes="Half-elf",
            source_books=["rezero-vol5", "rezero-vol6"],
        )
        data = e.model_dump()
        restored = GlossaryEntity(**data)
        assert restored.canonical_english == "Emilia"
        assert len(restored.surface_forms) == 2
        assert restored.surface_forms[1].source == "エミリアたん"
        assert restored.source_books == ["rezero-vol5", "rezero-vol6"]


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


class TestGlossary:
    def test_empty_glossary(self):
        g = Glossary()
        assert g.entities == []
        assert g.version == 2
        assert g.book_id is None

    def test_glossary_with_entities(self):
        g = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    source="seed",
                ),
                GlossaryEntity(
                    entity_id="place_000001",
                    category="place",
                    canonical_english="Lugunica",
                    surface_forms=[SurfaceForm(source="ルグニカ", english="Lugunica")],
                    source="extracted",
                ),
            ],
            book_id="rezero-5",
            book_metadata={"title": "Re:Zero Vol 5"},
        )
        assert len(g.entities) == 2
        assert g.book_id == "rezero-5"

    def test_round_trip_json(self):
        now = datetime.now(timezone.utc)
        g = Glossary(
            entities=[
                GlossaryEntity(
                    entity_id="term_000001",
                    category="term",
                    canonical_english="Test",
                    source="user",
                ),
            ],
            version=2,
            book_id="test-book",
            created_at=now,
            updated_at=now,
        )
        data = g.model_dump(mode="json")
        restored = Glossary(**data)
        assert restored.version == 2
        assert len(restored.entities) == 1
        assert restored.book_id == "test-book"


# ---------------------------------------------------------------------------
# ExtractedMention
# ---------------------------------------------------------------------------


class TestExtractedMention:
    def test_minimal(self):
        m = ExtractedMention(source="スバル", english="Subaru", category="character")
        assert m.reading is None
        assert m.summary_update is None
        assert m.context_hint is None
        assert m.aliases == []
        assert m.nicknames == {}

    def test_full(self):
        m = ExtractedMention(
            source="ヴィンセント・ヴォラキア",
            reading="ゔぃんせんと・ゔぉらきあ",
            english="Vincent Volakia",
            category="character",
            summary_update="The emperor of Volakia.",
            context_hint="same person as アベル",
            notes="Full name revealed.",
            aliases=["ヴィンセント"],
            nicknames={"Abel": "Your Majesty"},
            speech_style="Calm and authoritative.",
        )
        assert m.context_hint == "same person as アベル"
        assert m.summary_update == "The emperor of Volakia."

    def test_round_trip(self):
        m = ExtractedMention(
            source="テスト",
            english="Test",
            category="term",
            context_hint="a hint",
        )
        data = m.model_dump()
        restored = ExtractedMention(**data)
        assert restored.source == "テスト"
        assert restored.context_hint == "a hint"


# ---------------------------------------------------------------------------
# GlossaryExtractionResponse
# ---------------------------------------------------------------------------


class TestGlossaryExtractionResponse:
    def test_empty(self):
        r = GlossaryExtractionResponse()
        assert r.mentions == []
        assert r.corrections == []

    def test_with_mentions(self):
        r = GlossaryExtractionResponse(
            mentions=[
                ExtractedMention(source="スバル", english="Subaru", category="character"),
            ]
        )
        assert len(r.mentions) == 1
        assert r.mentions[0].source == "スバル"


# ---------------------------------------------------------------------------
# GlossaryClusterDecision / GlossaryClusterResponse
# ---------------------------------------------------------------------------


class TestGlossaryClusterDecision:
    def test_basic_construction(self):
        d = GlossaryClusterDecision(
            entity_id_a="c001",
            entity_id_b="c002",
            same_entity=True,
            preferred_entity_id="c001",
            preferred_canonical_english="Abel",
            reasoning="Same character.",
        )
        assert d.same_entity is True
        assert d.preferred_entity_id == "c001"

    def test_not_same_entity(self):
        d = GlossaryClusterDecision(
            entity_id_a="c001",
            entity_id_b="p001",
            same_entity=False,
            reasoning="Different entity types.",
        )
        assert d.same_entity is False
        assert d.preferred_entity_id is None
        assert d.preferred_canonical_english is None

    def test_round_trip_json(self):
        d = GlossaryClusterDecision(
            entity_id_a="c001",
            entity_id_b="c002",
            same_entity=True,
            preferred_entity_id="c001",
            preferred_canonical_english="Abel",
            reasoning="Same character with honorific suffix.",
        )
        data = d.model_dump(mode="json")
        restored = GlossaryClusterDecision(**data)
        assert restored.entity_id_a == "c001"
        assert restored.preferred_canonical_english == "Abel"


class TestGlossaryClusterResponse:
    def test_empty(self):
        r = GlossaryClusterResponse()
        assert r.decisions == []

    def test_with_decisions(self):
        r = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=True,
                    preferred_entity_id="c001",
                    preferred_canonical_english="Abel",
                    reasoning="Same character.",
                ),
            ]
        )
        assert len(r.decisions) == 1
        assert r.decisions[0].same_entity is True

    def test_round_trip_json(self):
        r = GlossaryClusterResponse(
            decisions=[
                GlossaryClusterDecision(
                    entity_id_a="c001",
                    entity_id_b="c002",
                    same_entity=False,
                    reasoning="Not the same.",
                ),
            ]
        )
        data = r.model_dump(mode="json")
        restored = GlossaryClusterResponse(**data)
        assert len(restored.decisions) == 1
        assert restored.decisions[0].same_entity is False


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
