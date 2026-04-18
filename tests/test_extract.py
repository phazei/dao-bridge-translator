"""Tests for dao_bridge.extract — book_id derivation and normalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dao_bridge.extract import _derive_book_id, _normalize_for_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_book(
    identifiers: list[tuple[str, dict]] | None = None,
    titles: list[tuple[str, dict]] | None = None,
) -> MagicMock:
    """Create a mock EpubBook with controllable metadata."""
    book = MagicMock()

    def get_metadata(ns: str, key: str) -> list[tuple[str, dict]]:
        if ns == "DC" and key == "identifier":
            return identifiers or []
        if ns == "DC" and key == "title":
            return titles or []
        return []

    book.get_metadata = get_metadata
    return book


# ---------------------------------------------------------------------------
# _derive_book_id
# ---------------------------------------------------------------------------


class TestDeriveBookId:
    def test_isbn_urn_prefix_13_digit(self):
        book = _mock_book(identifiers=[("urn:isbn:978-4-04-123456-7", {})])
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"

    def test_isbn_prefix_13_digit(self):
        book = _mock_book(identifiers=[("ISBN:978-4-04-123456-7", {})])
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"

    def test_isbn_10_digit(self):
        book = _mock_book(identifiers=[("ISBN:4041234565", {})])
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-4041234565"

    def test_isbn_with_spaces(self):
        book = _mock_book(identifiers=[("ISBN: 978 4 04 123456 7", {})])
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"

    def test_isbn_case_insensitive(self):
        book = _mock_book(identifiers=[("isbn:9784041234567", {})])
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"

    def test_non_isbn_identifier_falls_through_to_title(self):
        """A UUID identifier should not be picked up as ISBN."""
        book = _mock_book(
            identifiers=[("urn:uuid:12345678-1234-1234-1234-123456789012", {})],
            titles=[("My Cool Book", {})],
        )
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "my-cool-book"

    def test_isbn_wrong_length_falls_through(self):
        """An ISBN-prefixed value with wrong digit count should be skipped."""
        book = _mock_book(
            identifiers=[("ISBN:12345", {})],
            titles=[("Fallback Title", {})],
        )
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "fallback-title"

    def test_title_fallback(self):
        book = _mock_book(titles=[("Re：ゼロから始める異世界生活　第五巻", {})])
        result = _derive_book_id(book, Path("book.epub"))
        # Should be NFKC normalized, lowercased, non-alphanum replaced with hyphens
        assert "re" in result
        assert "ゼロ" in result

    def test_filename_fallback(self):
        book = _mock_book()  # No identifiers, no titles
        result = _derive_book_id(book, Path("/books/My Novel - Vol3.epub"))
        assert result == "my-novel-vol3"

    def test_prefers_isbn_over_title(self):
        book = _mock_book(
            identifiers=[("ISBN:9784041234567", {})],
            titles=[("Should Not Use This", {})],
        )
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"

    def test_multiple_identifiers_picks_first_isbn(self):
        book = _mock_book(
            identifiers=[
                ("urn:uuid:some-uuid", {}),
                ("ISBN:9784041234567", {}),
                ("ISBN:9784041999999", {}),
            ],
        )
        result = _derive_book_id(book, Path("book.epub"))
        assert result == "isbn-9784041234567"


# ---------------------------------------------------------------------------
# _normalize_for_id
# ---------------------------------------------------------------------------


class TestNormalizeForId:
    def test_simple_ascii(self):
        assert _normalize_for_id("Hello World") == "hello-world"

    def test_special_characters_replaced(self):
        assert _normalize_for_id("book: vol.3 (special)") == "book-vol-3-special"

    def test_multiple_hyphens_collapsed(self):
        assert _normalize_for_id("a --- b") == "a-b"

    def test_leading_trailing_hyphens_stripped(self):
        assert _normalize_for_id("---hello---") == "hello"

    def test_empty_string_returns_unknown(self):
        assert _normalize_for_id("") == "unknown"

    def test_only_special_chars_returns_unknown(self):
        assert _normalize_for_id("---") == "unknown"

    def test_fullwidth_nfkc_normalized(self):
        # Fullwidth "Ａ" (U+FF21) should normalize to "a"
        result = _normalize_for_id("Ａ")
        assert result == "a"

    def test_japanese_preserved(self):
        # CJK characters match \w after NFKC and should be kept
        result = _normalize_for_id("異世界生活")
        assert result == "異世界生活"

    def test_mixed_japanese_and_ascii(self):
        result = _normalize_for_id("Re：ゼロ Vol.5")
        assert "re" in result
        assert "ゼロ" in result
        assert "vol" in result
