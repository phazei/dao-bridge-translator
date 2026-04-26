"""Tests for dao_bridge.similarity — string similarity utilities."""

from __future__ import annotations

from dao_bridge.similarity import string_similarity


class TestStringSimilarity:
    """Tests for the bi-directional Jaro-Winkler string_similarity function."""

    def test_exact_match_returns_one(self):
        """Identical strings return 1.0."""
        assert string_similarity("hello", "hello") == 1.0

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert string_similarity("Hello", "hello") == 1.0

    def test_empty_string_returns_zero(self):
        """Empty string on either side returns 0.0."""
        assert string_similarity("", "hello") == 0.0
        assert string_similarity("hello", "") == 0.0
        assert string_similarity("", "") == 0.0

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        assert string_similarity("  hello  ", "hello") == 1.0

    def test_forward_jaro_winkler_prefix_match(self):
        """Forward JW catches prefix matches."""
        # "Subaru" vs "Subaru-kun" — strong prefix match.
        score = string_similarity("Subaru", "Subaru-kun")
        assert score > 0.85

    def test_reverse_jaro_winkler_suffix_match(self):
        """Reversed JW catches suffix matches (e.g. honorific additions)."""
        # "アベル" vs "アベルちゃん" — suffix addition.
        score = string_similarity("アベル", "アベルちゃん")
        assert score > 0.75

    def test_completely_different_strings_low_score(self):
        """Completely different strings have low similarity."""
        score = string_similarity("apple", "banana")
        assert score < 0.7

    def test_similar_strings_high_score(self):
        """Similar strings have high similarity."""
        # Common romanization variants.
        score = string_similarity("Petelgeuse", "Petelgeous")
        assert score > 0.85

    def test_japanese_exact_match(self):
        """Japanese strings match exactly."""
        assert string_similarity("ナツキ・スバル", "ナツキ・スバル") == 1.0

    def test_threshold_auto_attach(self):
        """Score >= 0.95 for very similar strings."""
        # Nearly identical with minor difference.
        score = string_similarity("Vincent", "Vincnet")  # typo
        assert score > 0.90

    def test_threshold_clustering_candidate(self):
        """Score in [0.75, 0.95) for moderately similar strings."""
        score = string_similarity("Abel", "Abel-chan")
        assert 0.70 < score < 1.0
