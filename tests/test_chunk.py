"""Comprehensive tests for dao_bridge.chunk — block parsing, packing, validation.

Uses a default ChunkingConfig (target=2000, max=2400, min=400, flex=0.2)
and fixture files under ``tests/fixtures/clean/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dao_bridge.chunk import (
    Block,
    ChunkValidationError,
    chunk_blocks,
    chunk_spine_item,
    find_last_break_point_in_range,
    parse_blocks,
    validate_chunks,
)
from dao_bridge.config import ChunkingConfig
from dao_bridge.schemas import Chunk, ManifestItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "clean"


def _default_config(**overrides) -> ChunkingConfig:
    """Return a default ChunkingConfig with optional overrides."""
    return ChunkingConfig(**overrides)


def _make_sentence(n: int = 1) -> str:
    """Return *n* copies of a standard sentence (~10 tokens each)."""
    return " ".join(["The quick brown fox jumps over the lazy dog."] * n)


def _make_paragraphs(n_paras: int, sentences_per: int = 10) -> str:
    """Generate markdown with *n_paras* paragraphs of *sentences_per* sentences."""
    paras = [_make_sentence(sentences_per) for _ in range(n_paras)]
    return "\n\n".join(paras) + "\n"


def _load_fixture(name: str) -> str:
    """Load a fixture file by name."""
    p = _FIXTURES_DIR / name
    assert p.exists(), f"Fixture missing: {p}"
    return p.read_text(encoding="utf-8")


# =========================================================================
# Block parsing
# =========================================================================


class TestBlockParsing:
    """Tests for parse_blocks()."""

    def test_plain_prose_single_paragraph(self):
        md = "This is a single paragraph with several sentences. It continues here."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 1
        assert blocks[0].kind == "paragraph"
        assert blocks[0].index == 0
        assert blocks[0].token_count > 0

    def test_multiple_paragraphs(self):
        md = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 3
        for i, b in enumerate(blocks):
            assert b.kind == "paragraph"
            assert b.index == i

    def test_multiline_paragraph_with_br(self):
        """Lines joined by markdown hard line break stay in one paragraph."""
        md = "Line one  \nLine two  \nLine three"
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 1
        assert blocks[0].kind == "paragraph"
        assert "Line one" in blocks[0].text
        assert "Line two" in blocks[0].text
        assert "Line three" in blocks[0].text

    def test_heading_atx(self):
        md = "# Chapter One\n\nSome text."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 2
        assert blocks[0].kind == "heading"
        assert blocks[0].text == "# Chapter One"
        assert blocks[1].kind == "paragraph"

    def test_heading_h2(self):
        md = "## Section Two"
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 1
        assert blocks[0].kind == "heading"

    def test_heading_h3(self):
        md = "### Subsection"
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 1
        assert blocks[0].kind == "heading"

    def test_heading_flushes_paragraph(self):
        """A heading mid-stream flushes the paragraph-in-progress."""
        md = "Some text\n# Heading\nMore text"
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 3
        assert blocks[0].kind == "paragraph"
        assert blocks[0].text == "Some text"
        assert blocks[1].kind == "heading"
        assert blocks[2].kind == "paragraph"

    def test_hr_dashes(self):
        md = "Before.\n\n---\n\nAfter."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 3
        assert blocks[1].kind == "hr"

    def test_hr_asterisks(self):
        md = "Before.\n\n***\n\nAfter."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 3
        assert blocks[1].kind == "hr"

    def test_hr_underscores(self):
        md = "Before.\n\n___\n\nAfter."
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 3
        assert blocks[1].kind == "hr"

    def test_hr_with_spaces(self):
        md = "Before.\n\n- - -\n\nAfter."
        blocks = parse_blocks(md, _default_config())
        assert blocks[1].kind == "hr"

    def test_scene_break_stars(self):
        md = "Before.\n\n***\n\nAfter."
        config = _default_config()
        blocks = parse_blocks(md, config)
        # '***' matches scene_break_patterns AND is an HR
        assert blocks[1].kind == "hr"

    def test_scene_break_custom_pattern(self):
        md = "Before.\n\n◇◇◇\n\nAfter."
        config = _default_config()
        blocks = parse_blocks(md, config)
        assert blocks[1].kind == "scene_break"

    def test_scene_break_unicode_asterisks(self):
        md = "Before.\n\n＊＊＊\n\nAfter."
        config = _default_config()
        blocks = parse_blocks(md, config)
        assert blocks[1].kind == "scene_break"

    def test_scene_break_dots(self):
        md = "Before.\n\n・・・\n\nAfter."
        config = _default_config()
        blocks = parse_blocks(md, config)
        assert blocks[1].kind == "scene_break"

    def test_scene_break_spaced_asterisks(self):
        """'* * *' matches the markdown HR regex, so it's detected as hr (not scene_break).
        HRs are treated as scene breaks by the chunker for splitting purposes."""
        md = "Before.\n\n* * *\n\nAfter."
        config = _default_config()
        blocks = parse_blocks(md, config)
        # '* * *' is a valid markdown HR, so kind is 'hr' not 'scene_break'.
        assert blocks[1].kind == "hr"

    def test_mixed_content(self):
        """Parse a mix of headings, paragraphs, and scene breaks."""
        md = (
            "# Title\n\nParagraph one.\n\n* * *\n\nParagraph two.\n\n## Section\n\nParagraph three."
        )
        blocks = parse_blocks(md, _default_config())
        assert len(blocks) == 6
        assert blocks[0].kind == "heading"
        assert blocks[1].kind == "paragraph"
        assert blocks[2].kind == "hr"  # '* * *' is a valid markdown HR
        assert blocks[3].kind == "paragraph"
        assert blocks[4].kind == "heading"
        assert blocks[5].kind == "paragraph"

    def test_empty_input(self):
        blocks = parse_blocks("", _default_config())
        assert blocks == []

    def test_whitespace_only_input(self):
        blocks = parse_blocks("   \n\n  \n", _default_config())
        assert blocks == []

    def test_block_indices_sequential(self):
        md = "# A\n\nPara.\n\n---\n\nPara.\n\n# B\n\nPara."
        blocks = parse_blocks(md, _default_config())
        for i, b in enumerate(blocks):
            assert b.index == i, f"Block {i} has index {b.index}"

    def test_token_count_positive(self):
        md = "Some text with several words in it."
        blocks = parse_blocks(md, _default_config())
        assert blocks[0].token_count > 0


# =========================================================================
# Scene break normalization
# =========================================================================


class TestSceneBreakNormalization:
    def test_normalized_when_configured(self):
        """Scene breaks get replaced with normalized form."""
        md = "Before.\n\n◇◇◇\n\nAfter."
        config = _default_config(normalize_scene_breaks="* * *")
        blocks = parse_blocks(md, config)
        sb = [b for b in blocks if b.kind == "scene_break"][0]
        assert sb.text == "* * *"

    def test_hr_normalized_when_configured(self):
        """HR blocks also get normalized."""
        md = "Before.\n\n---\n\nAfter."
        config = _default_config(normalize_scene_breaks="* * *")
        blocks = parse_blocks(md, config)
        hr = [b for b in blocks if b.kind == "hr"][0]
        assert hr.text == "* * *"

    def test_original_preserved_when_null(self):
        """When normalize_scene_breaks is None, original text is preserved."""
        md = "Before.\n\n◇◇◇\n\nAfter."
        config = _default_config(normalize_scene_breaks=None)
        blocks = parse_blocks(md, config)
        sb = [b for b in blocks if b.kind == "scene_break"][0]
        assert sb.text == "◇◇◇"

    def test_hr_original_preserved_when_null(self):
        md = "Before.\n\n---\n\nAfter."
        config = _default_config(normalize_scene_breaks=None)
        blocks = parse_blocks(md, config)
        hr = [b for b in blocks if b.kind == "hr"][0]
        assert hr.text == "---"

    def test_custom_normalization_form(self):
        md = "Before.\n\n***\n\nAfter."
        config = _default_config(normalize_scene_breaks="---")
        blocks = parse_blocks(md, config)
        hr = [b for b in blocks if b.kind == "hr"][0]
        assert hr.text == "---"


# =========================================================================
# find_last_break_point_in_range
# =========================================================================


class TestFindBreakPoint:
    def test_no_breaks(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "paragraph", "text", 100),
            Block(2, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 250)
        assert result is None

    def test_single_break_in_range(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "scene_break", "* * *", 3),
            Block(2, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 250)
        assert result == 1

    def test_break_below_range(self):
        blocks = [
            Block(0, "scene_break", "* * *", 3),
            Block(1, "paragraph", "text", 100),
            Block(2, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 250)
        assert result is None  # scene break at cumulative 3, below min 50

    def test_multiple_breaks_picks_latest(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "heading", "# A", 3),
            Block(2, "paragraph", "text", 100),
            Block(3, "scene_break", "* * *", 3),
            Block(4, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 350)
        assert result == 3  # Latest break within range

    def test_heading_counts_as_break(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "heading", "# Title", 3),
            Block(2, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 250)
        assert result == 1

    def test_hr_counts_as_break(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "hr", "---", 3),
            Block(2, "paragraph", "text", 100),
        ]
        result = find_last_break_point_in_range(blocks, 50, 250)
        assert result == 1

    def test_break_above_range_excluded(self):
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "paragraph", "text", 200),
            Block(2, "heading", "# Late", 3),
        ]
        # Range is 50..200, heading is at cumulative 303
        result = find_last_break_point_in_range(blocks, 50, 200)
        assert result is None


# =========================================================================
# Greedy packing algorithm
# =========================================================================


class TestGreedyPacking:
    """Tests for chunk_blocks()."""

    def test_empty_file_zero_chunks(self):
        chunks = chunk_blocks([], _default_config(), 0, "clean/0000.md")
        assert chunks == []

    def test_single_small_block_one_chunk(self):
        """File smaller than target produces exactly one chunk."""
        md = _load_fixture("short_chapter.md")
        blocks = parse_blocks(md, _default_config())
        chunks = chunk_blocks(blocks, _default_config(), 0, "clean/0000.md")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 1

    def test_under_target_one_chunk(self):
        """File under target_tokens but with multiple blocks makes one chunk."""
        md = _load_fixture("single_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        total = sum(b.token_count for b in blocks)
        assert total < config.target_tokens  # Confirm fixture is under target
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1

    def test_above_target_splits(self):
        """File above target gets split into multiple chunks."""
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        total = sum(b.token_count for b in blocks)
        assert total > config.target_tokens
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2

    def test_scene_break_in_flex_window(self):
        """Scene break within flex window is preferred for splitting."""
        md = _load_fixture("scene_break_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2

        # Check that at least one chunk ends at a scene break.
        any_ends_at_break = any(c.ends_at_scene_break for c in chunks)
        assert any_ends_at_break, "Expected at least one chunk to end at a scene break"

    def test_oversized_single_block(self):
        """A block exceeding max_tokens still becomes its own chunk."""
        md = _load_fixture("oversized_paragraph.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        assert len(blocks) == 1
        assert blocks[0].token_count > config.max_tokens
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1
        assert chunks[0].token_count > config.max_tokens

    def test_chunk_indices_sequential_from_1(self):
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i + 1

    def test_chunk_ids_formatted_correctly(self):
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 5, "clean/005.md")
        for c in chunks:
            assert c.chunk_id.startswith("0005.")
            assert c.spine_index == 5

    def test_block_range_inclusive(self):
        """block_range is [start, end] inclusive."""
        md = _load_fixture("single_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1
        start, end = chunks[0].block_range
        assert start == 0
        assert end == len(blocks) - 1


# =========================================================================
# Remainder absorption
# =========================================================================


class TestRemainderAbsorption:
    def test_tiny_remainder_absorbed(self):
        """Final content < min_chunk_tokens is absorbed into previous chunk."""
        md = _load_fixture("tiny_remainder.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")

        # The last chunk should have extended_for_remainder set.
        # Find it — it's the chunk that absorbed the remainder.
        extended = [c for c in chunks if c.extended_for_remainder]
        assert len(extended) == 1, (
            f"Expected exactly 1 extended chunk, got {len(extended)}. Chunk count: {len(chunks)}"
        )

    def test_single_small_file_not_absorbed(self):
        """A single small file (no previous chunk) is not absorbed — just emitted."""
        md = _load_fixture("short_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1
        assert not chunks[0].extended_for_remainder


# =========================================================================
# Classification filtering
# =========================================================================


class TestClassificationFiltering:
    """Tests that non-chunkable classifications produce zero chunks."""

    def _make_item(self, classification, spine_index=0):
        return ManifestItem(
            spine_index=spine_index,
            original_href=f"text/{spine_index:04d}.xhtml",
            raw_path=f"raw/{spine_index:04d}.xhtml",
            clean_path=f"clean/{spine_index:04d}.md",
            classification=classification,
        )

    def test_illustration_not_chunkable(self):
        config = _default_config()
        item = self._make_item("illustration")
        assert item.classification not in config.chunkable_classifications

    def test_toc_auto_is_chunkable(self):
        config = _default_config()
        item = self._make_item("toc_auto")
        assert item.classification in config.chunkable_classifications

    def test_chapter_is_chunkable(self):
        config = _default_config()
        item = self._make_item("chapter")
        assert item.classification in config.chunkable_classifications

    def test_frontmatter_is_chunkable(self):
        config = _default_config()
        assert "frontmatter" in config.chunkable_classifications

    def test_backmatter_is_chunkable(self):
        config = _default_config()
        assert "backmatter" in config.chunkable_classifications

    def test_toc_authored_is_chunkable(self):
        config = _default_config()
        assert "toc_authored" in config.chunkable_classifications


# =========================================================================
# Determinism
# =========================================================================


class TestDeterminism:
    def test_same_input_produces_identical_output(self):
        """Running the chunker twice produces identical results."""
        md = _load_fixture("scene_break_chapter.md")
        config = _default_config()

        blocks1 = parse_blocks(md, config)
        chunks1 = chunk_blocks(blocks1, config, 0, "clean/0000.md")

        blocks2 = parse_blocks(md, config)
        chunks2 = chunk_blocks(blocks2, config, 0, "clean/0000.md")

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            j1 = c1.model_dump_json(indent=2)
            j2 = c2.model_dump_json(indent=2)
            assert j1 == j2, f"Chunk {c1.chunk_id} differs between runs"


# =========================================================================
# Block coverage validation
# =========================================================================


class TestBlockCoverage:
    def test_all_blocks_covered(self):
        """Every block index appears in exactly one chunk."""
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")

        covered: set[int] = set()
        for c in chunks:
            for bi in range(c.block_range[0], c.block_range[1] + 1):
                assert bi not in covered, f"Block {bi} in multiple chunks"
                covered.add(bi)

        expected = set(range(len(blocks)))
        assert covered == expected

    def test_scene_break_chapter_coverage(self):
        md = _load_fixture("scene_break_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")

        covered: set[int] = set()
        for c in chunks:
            for bi in range(c.block_range[0], c.block_range[1] + 1):
                covered.add(bi)
        assert covered == set(range(len(blocks)))

    def test_tiny_remainder_coverage(self):
        md = _load_fixture("tiny_remainder.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")

        covered: set[int] = set()
        for c in chunks:
            for bi in range(c.block_range[0], c.block_range[1] + 1):
                covered.add(bi)
        assert covered == set(range(len(blocks)))


# =========================================================================
# Validation function
# =========================================================================


class TestValidation:
    def test_valid_chunks_pass(self):
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        # Should not raise.
        validate_chunks(blocks, chunks)

    def test_empty_blocks_empty_chunks_ok(self):
        validate_chunks([], [])

    def test_blocks_but_no_chunks_fails(self):
        blocks = [Block(0, "paragraph", "text", 10)]
        with pytest.raises(ChunkValidationError, match="no chunks"):
            validate_chunks(blocks, [])

    def test_gap_in_coverage_fails(self):
        blocks = [Block(0, "paragraph", "a", 5), Block(1, "paragraph", "b", 5)]
        # Create a chunk covering only block 0.
        bad_chunk = Chunk(
            chunk_id="0000.001",
            spine_index=0,
            chunk_index=1,
            source_file="clean/0000.md",
            block_range=(0, 0),
            token_count=5,
            text="a",
        )
        with pytest.raises(ChunkValidationError, match="coverage mismatch"):
            validate_chunks(blocks, [bad_chunk])

    def test_non_sequential_indices_fails(self):
        blocks = [Block(0, "paragraph", "a", 5), Block(1, "paragraph", "b", 5)]
        bad_chunks = [
            Chunk(
                chunk_id="0000.002",
                spine_index=0,
                chunk_index=2,  # should be 1
                source_file="clean/0000.md",
                block_range=(0, 1),
                token_count=10,
                text="a\n\nb",
            ),
        ]
        with pytest.raises(ChunkValidationError, match="Expected chunk_index 1"):
            validate_chunks(blocks, bad_chunks)


# =========================================================================
# chunk_spine_item (filesystem integration)
# =========================================================================


class TestChunkSpineItem:
    def test_writes_chunk_files(self, tmp_path: Path):
        """Chunk files are written to chunks/NNNN/ directory."""
        work_dir = tmp_path / "work"
        (work_dir / "clean").mkdir(parents=True)
        (work_dir / "chunks").mkdir(parents=True)

        # Write a test markdown file.
        md = _load_fixture("two_chunk_chapter.md")
        (work_dir / "clean" / "0003.md").write_text(md, encoding="utf-8")

        item = ManifestItem(
            spine_index=3,
            original_href="text/0003.xhtml",
            raw_path="raw/0003.xhtml",
            clean_path="clean/0003.md",
            classification="chapter",
        )
        config = _default_config()
        n_chunks = chunk_spine_item(work_dir, item, config)

        assert n_chunks >= 2
        chunk_d = work_dir / "chunks" / "0003"
        assert chunk_d.exists()
        chunk_files = sorted(chunk_d.glob("*.json"))
        assert len(chunk_files) == n_chunks

        # Verify each file is valid JSON / valid Chunk.
        for cf in chunk_files:
            raw = json.loads(cf.read_text(encoding="utf-8"))
            c = Chunk(**raw)
            assert c.spine_index == 3
            assert c.source_file == "clean/0003.md"

    def test_empty_file_zero_chunks(self, tmp_path: Path):
        work_dir = tmp_path / "work"
        (work_dir / "clean").mkdir(parents=True)
        (work_dir / "chunks").mkdir(parents=True)

        (work_dir / "clean" / "0000.md").write_text("", encoding="utf-8")

        item = ManifestItem(
            spine_index=0,
            original_href="text/0000.xhtml",
            raw_path="raw/0000.xhtml",
            clean_path="clean/0000.md",
            classification="chapter",
        )
        n_chunks = chunk_spine_item(work_dir, item, _default_config())
        assert n_chunks == 0

    def test_missing_clean_file_raises(self, tmp_path: Path):
        work_dir = tmp_path / "work"
        (work_dir / "clean").mkdir(parents=True)

        item = ManifestItem(
            spine_index=0,
            original_href="text/0000.xhtml",
            raw_path="raw/0000.xhtml",
            clean_path="clean/0000.md",
            classification="chapter",
        )
        with pytest.raises(FileNotFoundError):
            chunk_spine_item(work_dir, item, _default_config())


# =========================================================================
# Fixture-driven packing tests
# =========================================================================


class TestFixturePacking:
    """Use the actual fixture files to test packing behavior."""

    def test_short_chapter_one_chunk(self):
        md = _load_fixture("short_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1
        assert not chunks[0].extended_for_remainder

    def test_single_chunk_chapter(self):
        md = _load_fixture("single_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1

    def test_two_chunk_chapter(self):
        md = _load_fixture("two_chunk_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2
        validate_chunks(blocks, chunks)

    def test_scene_break_chapter(self):
        md = _load_fixture("scene_break_chapter.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2
        validate_chunks(blocks, chunks)

    def test_oversized_paragraph(self):
        md = _load_fixture("oversized_paragraph.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1
        validate_chunks(blocks, chunks)

    def test_tiny_remainder(self):
        md = _load_fixture("tiny_remainder.md")
        config = _default_config()
        blocks = parse_blocks(md, config)
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        validate_chunks(blocks, chunks)
        # At least one chunk should be extended.
        extended = [c for c in chunks if c.extended_for_remainder]
        assert len(extended) == 1


# =========================================================================
# Edge cases for packing algorithm
# =========================================================================


class TestPackingEdgeCases:
    def test_exact_target_one_chunk(self):
        """A file whose total tokens == target produces exactly one chunk."""
        config = _default_config(target_tokens=100)
        # 10 blocks of 10 tokens each = 100 tokens total
        blocks = [Block(i, "paragraph", _make_sentence(1), 10) for i in range(10)]
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) == 1

    def test_target_plus_one_splits(self):
        """target+1 tokens triggers a split."""
        config = _default_config(target_tokens=100, min_chunk_tokens=10)
        # 11 blocks of 10 tokens = 110 tokens
        blocks = [Block(i, "paragraph", _make_sentence(1), 10) for i in range(11)]
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2

    def test_scene_break_outside_flex_window_not_used(self):
        """Scene break too early (outside flex window) is not used for splitting."""
        config = _default_config(target_tokens=200, flex_window_ratio=0.2, min_chunk_tokens=10)
        # flex_min = 200 * 0.8 = 160
        # Scene break at cumulative 50 (below flex_min)
        blocks = [
            Block(0, "paragraph", "text", 50),
            Block(1, "scene_break", "* * *", 3),  # cumulative: 53 — below 160
            Block(2, "paragraph", "text", 50),  # cumulative: 103
            Block(3, "paragraph", "text", 50),  # cumulative: 153
            Block(4, "paragraph", "text", 50),  # cumulative: 203 > 200
        ]
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        # The scene break is below flex_min, so it won't be used.
        # Split should happen at target boundary.
        assert len(chunks) >= 2
        # First chunk should NOT end at block 1 (the scene break).
        assert chunks[0].block_range[1] != 1

    def test_multiple_scene_breaks_in_flex_latest_chosen(self):
        """When multiple breaks are in the flex window, pick the latest."""
        config = _default_config(target_tokens=200, flex_window_ratio=0.3, min_chunk_tokens=10)
        # flex_min = 200 * 0.7 = 140
        blocks = [
            Block(0, "paragraph", "text", 100),
            Block(1, "scene_break", "* * *", 3),  # cumulative: 103
            Block(2, "paragraph", "text", 50),  # cumulative: 153, in flex
            Block(3, "heading", "# X", 3),  # cumulative: 156, in flex — latest
            Block(4, "paragraph", "text", 50),  # cumulative: 206 > 200
        ]
        chunks = chunk_blocks(blocks, config, 0, "clean/0000.md")
        assert len(chunks) >= 2
        # First chunk should end at block 3 (the heading, latest break in flex).
        assert chunks[0].block_range[1] == 3


# =========================================================================
# chunk_all orchestrator — targeted --spine and --force
# =========================================================================


class TestChunkAllTargetedSpine:
    """Tests for --spine overriding completed state and targeted --force."""

    @staticmethod
    def _setup(tmp_path: Path):
        """Create a two-item manifest with clean markdown files."""
        from dao_bridge.chunk import chunk_all
        from dao_bridge.config import AppConfig
        from dao_bridge.schemas import Manifest, ManifestItem as MI
        from dao_bridge.state import is_stage_completed, load_state
        from dao_bridge.workdir import ensure_dirs, manifest_path, pad_spine

        work_dir = tmp_path / "work"
        ensure_dirs(work_dir)
        (work_dir / "clean").mkdir(parents=True, exist_ok=True)
        (work_dir / "chunks").mkdir(parents=True, exist_ok=True)

        # Write clean markdown for two chapters.
        md = _make_paragraphs(5, sentences_per=20)  # ~1000 tokens, 1 chunk each
        (work_dir / "clean" / "0001.md").write_text(md, encoding="utf-8")
        (work_dir / "clean" / "0002.md").write_text(md, encoding="utf-8")

        item1 = MI(
            spine_index=1,
            original_href="text/001.xhtml",
            raw_path="raw/001.xhtml",
            clean_path="clean/0001.md",
            classification="chapter",
        )
        item2 = MI(
            spine_index=2,
            original_href="text/002.xhtml",
            raw_path="raw/002.xhtml",
            clean_path="clean/0002.md",
            classification="chapter",
        )
        manifest = Manifest(
            source_epub_path="dummy.epub",
            book_id="test",
            spine=[item1, item2],
        )
        from dao_bridge.workdir import atomic_write

        atomic_write(manifest_path(work_dir), manifest.model_dump_json(indent=2))

        config = AppConfig(source_epub="dummy.epub", work_dir=str(work_dir))
        return work_dir, manifest, config, chunk_all, load_state, is_stage_completed

    def test_spine_overrides_completed_state(self, tmp_path: Path):
        """--spine N rechunks even if the item is already completed."""
        work_dir, manifest, config, chunk_all, load_state, is_stage_completed = self._setup(
            tmp_path
        )

        # First run: chunk all.
        state = load_state(work_dir)
        chunk_all(config, manifest, state, force=False)

        assert is_stage_completed(load_state(work_dir), "chunk")

        # Run with --spine 1 (no --force). Should rechunk item 1.
        state2 = load_state(work_dir)
        result = chunk_all(config, manifest, state2, spine_filter=1)

        # Item 1 should have been rechunked (chunk_count set).
        assert result.spine[0].chunk_count is not None and result.spine[0].chunk_count > 0

    def test_spine_preserves_other_items_state(self, tmp_path: Path):
        """--spine N does not reset state for other items."""
        work_dir, manifest, config, chunk_all, load_state, _ = self._setup(tmp_path)

        state = load_state(work_dir)
        chunk_all(config, manifest, state, force=False)

        state2 = load_state(work_dir)
        chunk_all(config, manifest, state2, spine_filter=1)

        state3 = load_state(work_dir)
        assert state3.items["chunk:0002"].status == "completed"

    def test_force_with_spine_is_targeted(self, tmp_path: Path):
        """--force --spine N only resets the targeted item."""
        work_dir, manifest, config, chunk_all, load_state, _ = self._setup(tmp_path)

        state = load_state(work_dir)
        chunk_all(config, manifest, state, force=False)

        state2 = load_state(work_dir)
        chunk_all(config, manifest, state2, force=True, spine_filter=1)

        state3 = load_state(work_dir)
        assert state3.items["chunk:0002"].status == "completed"
