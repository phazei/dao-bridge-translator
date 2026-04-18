#!/usr/bin/env python3
"""Generate markdown test fixtures for chunker tests.

Produces calibrated fixture files under ``tests/fixtures/clean/`` with
known token counts.  Uses ``tiktoken.cl100k_base`` to measure tokens
and a pool of diverse English sentences for natural-looking paragraphs.

Usage:
    python scripts/create_chunk_fixtures.py [--verify]

With ``--verify``, also prints chunker output per fixture for quick
sanity-checking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "clean"

enc = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Sentence pool — diverse English prose, each roughly 10-14 tokens.
# Avoids the monotony of repeating a single sentence.
# ---------------------------------------------------------------------------

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A gentle breeze carried the scent of autumn leaves through the air.",
    "She walked along the winding path toward the distant mountains ahead.",
    "The old clock tower stood silently watching over the sleeping town.",
    "Bright stars filled the dark sky on that cold winter evening.",
    "He paused at the crossroads and stared down the dusty trail.",
    "The river wound through the valley like a silver ribbon of light.",
    "Shadows lengthened across the courtyard as the sun began to set.",
    "A flock of birds scattered from the rooftop at the sudden noise.",
    "The market square bustled with merchants shouting their daily prices.",
    "Rain drummed steadily against the windowpanes of the old cottage.",
    "She traced her finger along the edge of the faded leather map.",
    "Lanterns swung in the wind above the narrow cobblestone streets.",
    "The forest floor was carpeted with moss and fallen pine needles.",
    "A lone wolf howled somewhere beyond the frozen mountain ridge.",
    "The smell of fresh bread drifted out from the bakery on the corner.",
    "He adjusted his glasses and turned to the next chapter of his book.",
    "Fireflies danced through the tall grass on that warm summer night.",
    "The train whistle echoed through the canyon as it rounded the bend.",
    "She found a small key hidden inside the hollow of an old oak tree.",
    "Waves crashed against the rocky shore sending white spray skyward.",
    "The blacksmith hammered steadily shaping iron on the glowing anvil.",
    "A cat stretched lazily on the sunlit windowsill ignoring the world.",
    "Thunder rumbled in the distance promising yet another evening storm.",
    "He climbed the narrow staircase to the top of the lighthouse tower.",
    "The garden was overgrown but the roses still bloomed defiantly red.",
    "Dust motes drifted through the beam of light from the high window.",
    "She whispered the old incantation and the stone door swung open.",
    "Footsteps echoed through the empty hallway of the abandoned school.",
    "The compass needle spun wildly as they crossed the magnetic ridge.",
]


def _count_tokens(text: str) -> int:
    return len(enc.encode(text))


def make_paragraph(n_sentences: int, offset: int = 0) -> str:
    """Build a paragraph by cycling through the sentence pool.

    *offset* rotates the starting position so consecutive paragraphs
    don't all begin with the same sentence.
    """
    pool_size = len(SENTENCES)
    parts = [SENTENCES[(offset + i) % pool_size] for i in range(n_sentences)]
    return " ".join(parts)


def make_paragraphs(
    n_paras: int,
    sentences_per: int = 10,
    start_offset: int = 0,
) -> str:
    """Build *n_paras* paragraphs separated by blank lines."""
    paras = []
    for i in range(n_paras):
        paras.append(make_paragraph(sentences_per, offset=start_offset + i))
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# Fixture definitions
#
# Each fixture is a (filename, builder_function) pair.  The builder
# returns the full markdown string.
#
# Default chunking config for reference:
#   target_tokens  = 2000
#   max_tokens     = 2400
#   min_chunk_tokens = 400
#   flex_window_ratio = 0.2
# ---------------------------------------------------------------------------


def build_short_chapter() -> str:
    """~50-60 tokens, single paragraph.  Under min_chunk_tokens."""
    return make_paragraph(5, offset=0) + "\n"


def build_single_chunk_chapter() -> str:
    """~1500-1800 tokens, multiple paragraphs.  Under target_tokens."""
    return "# Chapter One\n\n" + make_paragraphs(5, sentences_per=30, start_offset=0) + "\n"


def build_two_chunk_chapter() -> str:
    """~4500-5500 tokens, no scene breaks.  Forces splitting without natural breaks."""
    return "# Chapter Two\n\n" + make_paragraphs(9, sentences_per=50, start_offset=5) + "\n"


def build_scene_break_chapter() -> str:
    """~5000+ tokens with scene breaks at various positions."""
    parts = ["# A Chapter with Scene Breaks\n"]
    # Section 1: ~800 tokens
    parts.append(make_paragraphs(8, sentences_per=10, start_offset=0))
    parts.append("\n\n* * *\n\n")
    # Section 2: ~1000 tokens
    parts.append(make_paragraphs(10, sentences_per=10, start_offset=1))
    parts.append("\n\n***\n\n")
    # Section 3: ~1200 tokens
    parts.append(make_paragraphs(12, sentences_per=10, start_offset=2))
    parts.append("\n\n* * *\n\n")
    # Section 4: ~1500 tokens
    parts.append(make_paragraphs(15, sentences_per=10, start_offset=3))
    parts.append("\n\n")
    # Section 5: ~500 tokens
    parts.append(make_paragraphs(5, sentences_per=10, start_offset=4))
    parts.append("\n")
    return "".join(parts)


def build_oversized_paragraph() -> str:
    """Single paragraph larger than max_tokens (2400).  ~3000+ tokens."""
    return make_paragraph(300, offset=0) + "\n"


def build_tiny_remainder() -> str:
    """Two full chunks worth of content plus a tiny trailing remainder
    that is well under min_chunk_tokens (400) and should be absorbed
    into the previous chunk.

    With ~12.4 tokens/sentence average and 10 sentences/paragraph,
    each paragraph is ~123 tokens.  The greedy packer emits a chunk
    once the *next* block would push it over ``target_tokens``, so
    a 16-paragraph section (~1970 tokens) fills a chunk cleanly.

    The remainder must end up under ``min_chunk_tokens`` (400) to
    trigger absorption.  Using 16 + 15 paragraphs leaves a remainder
    of about 3 paragraphs + 2 sentences from leftover packing, which
    lands around 250-350 tokens — safely below 400.

    Structure:
      - Heading (~2 tokens)
      - 16 paragraphs of 10 sentences (~1970 tokens) -> chunk 1
      - 15 paragraphs of 10 sentences (~1845 tokens) -> chunk 2
      - 1 tiny paragraph of 2 sentences (~25 tokens) -> absorbed into chunk 2
    """
    parts = ["# Opening\n\n"]
    # Bulk section 1: fills chunk 1 up to near target.
    parts.append(make_paragraphs(16, sentences_per=10, start_offset=0))
    parts.append("\n\n")
    # Bulk section 2: slightly under target so the greedy packer
    # consumes it all into chunk 2 before hitting the remainder.
    parts.append(make_paragraphs(15, sentences_per=10, start_offset=7))
    parts.append("\n\n")
    # Tiny remainder — well under min_chunk_tokens (400).
    parts.append(make_paragraph(2, offset=15))
    parts.append("\n")
    return "".join(parts)


FIXTURES: list[tuple[str, callable]] = [
    ("short_chapter.md", build_short_chapter),
    ("single_chunk_chapter.md", build_single_chunk_chapter),
    ("two_chunk_chapter.md", build_two_chunk_chapter),
    ("scene_break_chapter.md", build_scene_break_chapter),
    ("oversized_paragraph.md", build_oversized_paragraph),
    ("tiny_remainder.md", build_tiny_remainder),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_all() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in FIXTURES:
        text = builder()
        path = FIXTURES_DIR / name
        path.write_text(text, encoding="utf-8")
        tokens = _count_tokens(text)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        print(f"  {name:.<40s} {tokens:>5} tokens, {len(blocks):>3} blocks")


def verify_all() -> None:
    """Run the chunker on each fixture and print results."""
    # Import lazily so the script works even if the package isn't installed
    # (generate mode doesn't need the chunker).
    from dao_bridge.chunk import chunk_blocks, parse_blocks
    from dao_bridge.config import ChunkingConfig

    config = ChunkingConfig()
    print(
        f"\n  Chunker config: target={config.target_tokens}, "
        f"max={config.max_tokens}, min={config.min_chunk_tokens}, "
        f"flex={config.flex_window_ratio}\n"
    )

    for name, _ in FIXTURES:
        path = FIXTURES_DIR / name
        if not path.exists():
            print(f"  {name}: MISSING — run without --verify first")
            continue
        text = path.read_text(encoding="utf-8")
        blocks = parse_blocks(text, config)
        chunks = chunk_blocks(blocks, config, 0, f"clean/{name}")
        total_block_tokens = sum(b.token_count for b in blocks)

        print(f"  {name}")
        print(f"    blocks={len(blocks)}, block_tokens={total_block_tokens}, chunks={len(chunks)}")
        for c in chunks:
            ext = " [extended]" if c.extended_for_remainder else ""
            brk = " [ends@break]" if c.ends_at_scene_break else ""
            print(
                f"    chunk {c.chunk_index}: "
                f"tokens={c.token_count}, "
                f"range={c.block_range}"
                f"{ext}{brk}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chunk test fixtures.")
    parser.add_argument("--verify", action="store_true", help="Also run chunker and print results.")
    args = parser.parse_args()

    print("Generating fixtures:")
    generate_all()

    if args.verify:
        print("\nVerifying with chunker:")
        verify_all()

    print("Done.")


if __name__ == "__main__":
    main()
