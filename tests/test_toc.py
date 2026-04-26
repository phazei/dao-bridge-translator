"""Tests for dao_bridge.toc -- ToC translation and OPF metadata updates."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lxml import etree

from dao_bridge.schemas import (
    Glossary,
    GlossaryEntity,
    Manifest,
    SurfaceForm,
    TocTranslationResponse,
)
from dao_bridge.toc import (
    _render_glossary_for_toc,
    apply_translated_titles,
    extract_toc_titles,
    find_toc_files,
    translate_titles,
    translate_toc,
    update_opf_metadata,
)

# ---------------------------------------------------------------------------
# Fixtures: NCX, Nav, and OPF XML builders
# ---------------------------------------------------------------------------

_NCX_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="test-uid"/>
  </head>
  <docTitle><text>Test Book</text></docTitle>
  <navMap>
{nav_points}
  </navMap>
</ncx>
"""

_NAV_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Navigation</title></head>
<body>
<nav epub:type="toc" id="toc">
  <h1>Table of Contents</h1>
  <ol>
{entries}
  </ol>
</nav>
</body>
</html>
"""

_OPF_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:language>{language}</dc:language>
    <dc:identifier id="uid">{identifier}</dc:identifier>
{extra_metadata}
  </metadata>
  <manifest>
{manifest_items}
  </manifest>
  <spine{spine_attrs}>
    <itemref idref="ch1"/>
  </spine>
</package>
"""

_CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _make_ncx(titles: list[str], hrefs: list[str] | None = None) -> str:
    """Build an NCX XML string with flat navPoints."""
    if hrefs is None:
        hrefs = [f"text/ch{i + 1}.xhtml" for i in range(len(titles))]
    points = []
    for i, (title, href) in enumerate(zip(titles, hrefs)):
        points.append(
            f'    <navPoint id="np{i}" playOrder="{i + 1}">\n'
            f"      <navLabel><text>{title}</text></navLabel>\n"
            f'      <content src="{href}"/>\n'
            f"    </navPoint>"
        )
    return _NCX_TEMPLATE.format(nav_points="\n".join(points))


def _make_nested_ncx(parts: list[tuple[str, list[str]]]) -> str:
    """Build an NCX with nested navPoints (parts containing chapters)."""
    points = []
    order = 1
    for pi, (part_title, chapter_titles) in enumerate(parts):
        children = []
        for ci, ch_title in enumerate(chapter_titles):
            children.append(
                f'        <navPoint id="np{pi}_{ci}" playOrder="{order}">\n'
                f"          <navLabel><text>{ch_title}</text></navLabel>\n"
                f'          <content src="text/ch{pi}_{ci}.xhtml"/>\n'
                f"        </navPoint>"
            )
            order += 1
        child_str = "\n".join(children)
        points.append(
            f'    <navPoint id="part{pi}" playOrder="{order}">\n'
            f"      <navLabel><text>{part_title}</text></navLabel>\n"
            f'      <content src="text/part{pi}.xhtml"/>\n'
            f"{child_str}\n"
            f"    </navPoint>"
        )
        order += 1
    return _NCX_TEMPLATE.format(nav_points="\n".join(points))


def _make_nav(titles: list[str], hrefs: list[str] | None = None) -> str:
    """Build a nav XHTML string with flat entries."""
    if hrefs is None:
        hrefs = [f"text/ch{i + 1}.xhtml" for i in range(len(titles))]
    entries = []
    for title, href in zip(titles, hrefs):
        entries.append(f'    <li><a href="{href}">{title}</a></li>')
    return _NAV_TEMPLATE.format(entries="\n".join(entries))


def _make_nested_nav(parts: list[tuple[str, list[str]]]) -> str:
    """Build a nav XHTML with nested lists."""
    entries = []
    for part_title, chapter_titles in parts:
        children = []
        for ch_title in chapter_titles:
            children.append(f'          <li><a href="text/ch.xhtml">{ch_title}</a></li>')
        child_str = "\n".join(children)
        entries.append(
            f'    <li><a href="text/part.xhtml">{part_title}</a>\n'
            f"      <ol>\n{child_str}\n"
            f"      </ol>\n"
            f"    </li>"
        )
    return _NAV_TEMPLATE.format(entries="\n".join(entries))


def _make_nav_with_markup(titles_markup: list[str]) -> str:
    """Build a nav where <a> elements contain nested markup (spans etc.)."""
    entries = []
    for i, markup in enumerate(titles_markup):
        entries.append(f'    <li><a href="text/ch{i + 1}.xhtml">{markup}</a></li>')
    return _NAV_TEMPLATE.format(entries="\n".join(entries))


def _make_opf(
    title: str = "Test Book",
    language: str = "ja",
    identifier: str = "urn:uuid:test-1234",
    ncx_id: str | None = "ncx",
    nav_href: str | None = None,
    extra_metadata: str = "",
) -> str:
    """Build an OPF XML string."""
    items = ['    <item id="ch1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>']
    if ncx_id:
        items.append(
            f'    <item id="{ncx_id}" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        )
    if nav_href:
        items.append(
            f'    <item id="nav" href="{nav_href}" '
            f'media-type="application/xhtml+xml" properties="nav"/>'
        )
    spine_attrs = f' toc="{ncx_id}"' if ncx_id else ""
    return _OPF_TEMPLATE.format(
        title=title,
        language=language,
        identifier=identifier,
        extra_metadata=extra_metadata,
        manifest_items="\n".join(items),
        spine_attrs=spine_attrs,
    )


def _make_mini_epub(
    tmp_path: Path,
    opf_dir: str = "OEBPS",
    ncx_content: str | None = None,
    nav_content: str | None = None,
    opf_content: str | None = None,
) -> Path:
    """Create a minimal EPUB ZIP for testing."""
    epub_path = tmp_path / "test.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        # mimetype (first, uncompressed).
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, "application/epub+zip")

        # container.xml
        opf_path = f"{opf_dir}/content.opf" if opf_dir else "content.opf"
        zf.writestr("META-INF/container.xml", _CONTAINER_XML.format(opf_path=opf_path))

        # OPF
        if opf_content is None:
            opf_content = _make_opf()
        zf.writestr(opf_path, opf_content)

        # NCX
        if ncx_content is not None:
            ncx_path = f"{opf_dir}/toc.ncx" if opf_dir else "toc.ncx"
            zf.writestr(ncx_path, ncx_content)

        # Nav
        if nav_content is not None:
            nav_path = f"{opf_dir}/nav.xhtml" if opf_dir else "nav.xhtml"
            zf.writestr(nav_path, nav_content)

        # A dummy XHTML chapter.
        ch_path = f"{opf_dir}/text/ch1.xhtml" if opf_dir else "text/ch1.xhtml"
        zf.writestr(ch_path, "<html><body><p>Content</p></body></html>")

    return epub_path


def _make_config(**overrides):
    """Build a minimal AppConfig for testing."""
    from dao_bridge.config import AppConfig

    defaults = {
        "source_epub": "/fake/book.epub",
        "languages": {"source": "ja", "target": "en"},
        "models": {
            "classify": {"base_url": "http://localhost:8080/v1", "api_key": "x", "model": "test"},
            "glossary": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "x",
                "model": "test-glossary",
            },
            "translate": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "x",
                "model": "test-translate",
            },
        },
        "output": {
            "epub_path": "./book.en.epub",
            "title_suffix": " (English Translation)",
            "new_identifier": False,
            "css": "original",
            "add_translation_note": True,
            "run_epubcheck": False,
        },
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_glossary(entries=None, entities=None):
    """Build a minimal Glossary.

    Accepts *entities* (list of GlossaryEntity) directly, or *entries*
    for backward-compat test helpers that build entity dicts.
    """
    return Glossary(
        entities=entities or entries or [],
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Tests: find_toc_files
# ---------------------------------------------------------------------------


class TestFindTocFiles:
    def test_both_ncx_and_nav(self, tmp_path: Path):
        ncx = _make_ncx(["Ch 1", "Ch 2"])
        nav = _make_nav(["Ch 1", "Ch 2"])
        opf = _make_opf(ncx_id="ncx", nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, nav_content=nav, opf_content=opf)

        ncx_path, nav_path = find_toc_files(str(epub), "OEBPS")
        assert ncx_path == "OEBPS/toc.ncx"
        assert nav_path == "OEBPS/nav.xhtml"

    def test_only_ncx(self, tmp_path: Path):
        ncx = _make_ncx(["Ch 1"])
        opf = _make_opf(ncx_id="ncx", nav_href=None)
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, opf_content=opf)

        ncx_path, nav_path = find_toc_files(str(epub), "OEBPS")
        assert ncx_path == "OEBPS/toc.ncx"
        assert nav_path is None

    def test_only_nav(self, tmp_path: Path):
        nav = _make_nav(["Ch 1"])
        opf = _make_opf(ncx_id=None, nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, nav_content=nav, opf_content=opf)

        ncx_path, nav_path = find_toc_files(str(epub), "OEBPS")
        assert ncx_path is None
        assert nav_path == "OEBPS/nav.xhtml"

    def test_neither(self, tmp_path: Path):
        opf = _make_opf(ncx_id=None, nav_href=None)
        epub = _make_mini_epub(tmp_path, opf_content=opf)

        ncx_path, nav_path = find_toc_files(str(epub), "OEBPS")
        assert ncx_path is None
        assert nav_path is None

    def test_opf_in_subdirectory(self, tmp_path: Path):
        """OPF in a subdirectory resolves hrefs correctly."""
        ncx = _make_ncx(["Ch 1"])
        opf = _make_opf(ncx_id="ncx")
        epub = _make_mini_epub(tmp_path, opf_dir="OEBPS", ncx_content=ncx, opf_content=opf)

        ncx_path, _ = find_toc_files(str(epub), "OEBPS")
        assert ncx_path == "OEBPS/toc.ncx"

    def test_opf_at_root(self, tmp_path: Path):
        """OPF at the ZIP root resolves hrefs correctly."""
        ncx = _make_ncx(["Ch 1"])
        opf = _make_opf(ncx_id="ncx")
        epub = _make_mini_epub(tmp_path, opf_dir="", ncx_content=ncx, opf_content=opf)

        ncx_path, _ = find_toc_files(str(epub), "")
        assert ncx_path == "toc.ncx"


# ---------------------------------------------------------------------------
# Tests: extract_toc_titles
# ---------------------------------------------------------------------------


class TestExtractTocTitles:
    def test_ncx_flat(self):
        ncx = _make_ncx(["Chapter 1", "Chapter 2", "Epilogue"])
        titles = extract_toc_titles(ncx, "ncx")
        assert titles == ["Chapter 1", "Chapter 2", "Epilogue"]

    def test_ncx_nested(self):
        ncx = _make_nested_ncx(
            [
                ("Part I", ["Chapter 1", "Chapter 2"]),
                ("Part II", ["Chapter 3"]),
            ]
        )
        titles = extract_toc_titles(ncx, "ncx")
        # NCX iterates <text> in document order: Part I children first, then Part I label, etc.
        # Actually lxml.iter is document order: Part I label, then children,
        # then Part II label, then children.
        # Let's verify the actual count.
        assert len(titles) == 5
        # All titles should be present.
        assert "Part I" in titles
        assert "Part II" in titles
        assert "Chapter 1" in titles
        assert "Chapter 2" in titles
        assert "Chapter 3" in titles

    def test_ncx_empty_text(self):
        ncx = _make_ncx(["", "Chapter 2"])
        titles = extract_toc_titles(ncx, "ncx")
        assert titles == ["", "Chapter 2"]

    def test_nav_flat(self):
        nav = _make_nav(["Chapter 1", "Chapter 2", "Epilogue"])
        titles = extract_toc_titles(nav, "nav")
        assert titles == ["Chapter 1", "Chapter 2", "Epilogue"]

    def test_nav_nested(self):
        nav = _make_nested_nav(
            [
                ("Part I", ["Chapter 1", "Chapter 2"]),
                ("Part II", ["Chapter 3"]),
            ]
        )
        titles = extract_toc_titles(nav, "nav")
        assert len(titles) == 5
        assert "Part I" in titles
        assert "Chapter 1" in titles

    def test_nav_with_internal_markup(self):
        """Titles with nested <span> elements: get_text() extracts clean text."""
        nav = _make_nav_with_markup(
            [
                '<span class="num">1</span> <span class="title">The Beginning</span>',
                "Plain Title",
            ]
        )
        titles = extract_toc_titles(nav, "nav")
        assert titles[0] == "1 The Beginning"
        assert titles[1] == "Plain Title"

    def test_invalid_toc_type(self):
        with pytest.raises(ValueError, match="Unknown toc_type"):
            extract_toc_titles("<xml/>", "invalid")


# ---------------------------------------------------------------------------
# Tests: translate_titles
# ---------------------------------------------------------------------------


class TestTranslateTitles:
    def test_correct_translation(self):
        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(
            titles=["Chapter 1: Reunion", "Chapter 2: The Hero"]
        )
        config = _make_config()
        glossary = _make_glossary()

        result = translate_titles(
            ["第一章　再会", "第二章　英雄"],
            glossary,
            mock_client,
            config,
        )
        assert result == ["Chapter 1: Reunion", "Chapter 2: The Hero"]
        mock_client.complete_json.assert_called_once()

    def test_wrong_count_raises(self):
        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(titles=["Only One"])
        config = _make_config()
        glossary = _make_glossary()

        with pytest.raises(ValueError, match="returned 1 titles, expected 2"):
            translate_titles(["Title A", "Title B"], glossary, mock_client, config)

    def test_empty_input(self):
        mock_client = MagicMock()
        config = _make_config()
        glossary = _make_glossary()

        result = translate_titles([], glossary, mock_client, config)
        assert result == []
        mock_client.complete_json.assert_not_called()

    def test_glossary_rendered_in_prompt(self):
        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(titles=["Chapter 1"])
        config = _make_config()
        glossary = _make_glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
                    source="extracted",
                ),
            ]
        )

        translate_titles(["第一章"], glossary, mock_client, config)

        # Check the system message contains the glossary entry.
        call_args = mock_client.complete_json.call_args
        messages = call_args[0][0]  # First positional arg.
        system_msg = messages[0]["content"]
        assert "スバル" in system_msg
        assert "Subaru" in system_msg

    def test_llm_error_propagates(self):
        from dao_bridge.llm_client import LLMStructuredOutputError

        mock_client = MagicMock()
        mock_client.complete_json.side_effect = LLMStructuredOutputError("LLM failed")
        config = _make_config()
        glossary = _make_glossary()

        with pytest.raises(LLMStructuredOutputError):
            translate_titles(["Title"], glossary, mock_client, config)


# ---------------------------------------------------------------------------
# Tests: apply_translated_titles
# ---------------------------------------------------------------------------


class TestApplyTranslatedTitles:
    def test_ncx_titles_replaced(self):
        ncx = _make_ncx(["第一章", "第二章"])
        result = apply_translated_titles(ncx, "ncx", ["Chapter 1", "Chapter 2"])

        # Parse result and verify -- use extract_toc_titles for consistency.
        titles = extract_toc_titles(result, "ncx")
        assert titles == ["Chapter 1", "Chapter 2"]

    def test_ncx_hrefs_preserved(self):
        ncx = _make_ncx(["Ch 1"], hrefs=["special/path.xhtml"])
        result = apply_translated_titles(ncx, "ncx", ["Translated 1"])

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://www.daisy.org/z3986/2005/ncx/"
        content = tree.find(f".//{{{ns}}}content")
        assert content is not None
        assert content.get("src") == "special/path.xhtml"

    def test_ncx_nesting_preserved(self):
        ncx = _make_nested_ncx([("Part I", ["Ch 1", "Ch 2"])])
        titles = extract_toc_titles(ncx, "ncx")
        translated = [f"T-{t}" for t in titles]
        result = apply_translated_titles(ncx, "ncx", translated)

        new_titles = extract_toc_titles(result, "ncx")
        assert new_titles == translated

    def test_nav_titles_replaced(self):
        nav = _make_nav(["第一章", "第二章"])
        result = apply_translated_titles(nav, "nav", ["Chapter 1", "Chapter 2"])

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://www.w3.org/1999/xhtml"
        a_elements = list(tree.iter(f"{{{ns}}}a"))
        # Filter to only those inside the toc nav.
        texts = [a.text for a in a_elements if a.text]
        assert "Chapter 1" in texts
        assert "Chapter 2" in texts

    def test_nav_hrefs_preserved(self):
        nav = _make_nav(["Ch 1"], hrefs=["special/path.xhtml"])
        result = apply_translated_titles(nav, "nav", ["Translated 1"])

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://www.w3.org/1999/xhtml"
        a_el = tree.find(f".//{{{ns}}}a")
        assert a_el is not None
        assert a_el.get("href") == "special/path.xhtml"

    def test_nav_internal_markup_replaced_by_flat_string(self):
        nav = _make_nav_with_markup(
            [
                '<span class="num">1</span> <span class="title">The Start</span>',
            ]
        )
        result = apply_translated_titles(nav, "nav", ["Chapter 1: The Start"])

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://www.w3.org/1999/xhtml"
        a_el = tree.find(f".//{{{ns}}}a")
        assert a_el is not None
        # Should have no child elements (spans removed), just text.
        assert len(list(a_el)) == 0
        assert a_el.text == "Chapter 1: The Start"

    def test_ncx_wrong_count_raises(self):
        ncx = _make_ncx(["Ch 1", "Ch 2"])
        with pytest.raises(ValueError, match="2 title elements but got 1"):
            apply_translated_titles(ncx, "ncx", ["Only One"])

    def test_nav_wrong_count_raises(self):
        nav = _make_nav(["Ch 1", "Ch 2"])
        with pytest.raises(ValueError, match="2 <a> elements but got 3"):
            apply_translated_titles(nav, "nav", ["A", "B", "C"])


# ---------------------------------------------------------------------------
# Tests: translate_toc (orchestrator)
# ---------------------------------------------------------------------------


class TestTranslateToc:
    def test_both_ncx_and_nav(self, tmp_path: Path):
        titles = ["第一章", "第二章"]
        ncx = _make_ncx(titles)
        nav = _make_nav(titles)
        opf = _make_opf(ncx_id="ncx", nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, nav_content=nav, opf_content=opf)

        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(
            titles=["Chapter 1", "Chapter 2"]
        )
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(
            source_epub_path=str(epub),
            book_id="test",
            opf_dir="OEBPS",
        )

        result = translate_toc(str(epub), manifest, glossary, mock_client, config)

        assert "OEBPS/toc.ncx" in result
        assert "OEBPS/nav.xhtml" in result
        # Only ONE LLM call (deduplicated).
        assert mock_client.complete_json.call_count == 1

    def test_only_ncx(self, tmp_path: Path):
        ncx = _make_ncx(["Ch 1"])
        opf = _make_opf(ncx_id="ncx", nav_href=None)
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, opf_content=opf)

        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(titles=["Chapter 1"])
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(source_epub_path=str(epub), book_id="test", opf_dir="OEBPS")

        result = translate_toc(str(epub), manifest, glossary, mock_client, config)

        assert "OEBPS/toc.ncx" in result
        assert len(result) == 1  # No nav.

    def test_only_nav(self, tmp_path: Path):
        nav = _make_nav(["Ch 1"])
        opf = _make_opf(ncx_id=None, nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, nav_content=nav, opf_content=opf)

        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(titles=["Chapter 1"])
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(source_epub_path=str(epub), book_id="test", opf_dir="OEBPS")

        result = translate_toc(str(epub), manifest, glossary, mock_client, config)

        assert "OEBPS/nav.xhtml" in result
        assert len(result) == 1

    def test_neither_returns_empty(self, tmp_path: Path):
        opf = _make_opf(ncx_id=None, nav_href=None)
        epub = _make_mini_epub(tmp_path, opf_content=opf)

        mock_client = MagicMock()
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(source_epub_path=str(epub), book_id="test", opf_dir="OEBPS")

        result = translate_toc(str(epub), manifest, glossary, mock_client, config)

        assert result == {}
        mock_client.complete_json.assert_not_called()

    def test_deduplication(self, tmp_path: Path):
        """NCX and nav share the same titles -- only unique titles sent to LLM."""
        titles = ["Chapter 1", "Chapter 2"]
        ncx = _make_ncx(titles)
        nav = _make_nav(titles)
        opf = _make_opf(ncx_id="ncx", nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, nav_content=nav, opf_content=opf)

        mock_client = MagicMock()
        mock_client.complete_json.return_value = TocTranslationResponse(
            titles=["Translated 1", "Translated 2"]
        )
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(source_epub_path=str(epub), book_id="test", opf_dir="OEBPS")

        translate_toc(str(epub), manifest, glossary, mock_client, config)

        # One call with 2 unique titles (not 4 = 2 NCX + 2 nav).
        call_args = mock_client.complete_json.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        titles_sent = json.loads(user_msg)
        assert len(titles_sent) == 2

    def test_different_titles_in_ncx_and_nav(self, tmp_path: Path):
        """NCX and nav have different titles -- all unique titles translated."""
        ncx = _make_ncx(["NCX Title 1", "Shared Title"])
        nav = _make_nav(["Nav Title 1", "Shared Title"])
        opf = _make_opf(ncx_id="ncx", nav_href="nav.xhtml")
        epub = _make_mini_epub(tmp_path, ncx_content=ncx, nav_content=nav, opf_content=opf)

        mock_client = MagicMock()
        # 3 unique titles: NCX Title 1, Shared Title, Nav Title 1
        mock_client.complete_json.return_value = TocTranslationResponse(
            titles=["T-NCX Title 1", "T-Shared Title", "T-Nav Title 1"]
        )
        config = _make_config(source_epub=str(epub))
        glossary = _make_glossary()
        manifest = Manifest(source_epub_path=str(epub), book_id="test", opf_dir="OEBPS")

        result = translate_toc(str(epub), manifest, glossary, mock_client, config)

        # Both files should be in the result.
        assert len(result) == 2
        # Verify NCX got correct translations.
        ncx_result = result["OEBPS/toc.ncx"].decode("utf-8")
        ncx_titles = extract_toc_titles(ncx_result, "ncx")
        assert ncx_titles == ["T-NCX Title 1", "T-Shared Title"]
        # Verify Nav got correct translations.
        nav_result = result["OEBPS/nav.xhtml"].decode("utf-8")
        nav_titles = extract_toc_titles(nav_result, "nav")
        assert nav_titles == ["T-Nav Title 1", "T-Shared Title"]


# ---------------------------------------------------------------------------
# Tests: update_opf_metadata
# ---------------------------------------------------------------------------


class TestUpdateOpfMetadata:
    def test_language_updated(self):
        opf = _make_opf(language="ja")
        config = _make_config()
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        lang = tree.find(f".//{{{ns}}}language")
        assert lang is not None
        assert lang.text == "en"

    def test_title_suffix_appended(self):
        opf = _make_opf(title="My Book")
        config = _make_config()
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        title = tree.find(f".//{{{ns}}}title")
        assert title is not None
        assert title.text == "My Book (English Translation)"

    def test_translation_note_added(self):
        opf = _make_opf()
        config = _make_config()
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        desc = tree.find(f".//{{{ns}}}description")
        assert desc is not None
        assert "Machine translated" in desc.text
        assert "test-translate" in desc.text

    def test_translation_note_not_added(self):
        opf = _make_opf()
        config = _make_config(
            output={
                "epub_path": "./book.en.epub",
                "add_translation_note": False,
                "title_suffix": " (EN)",
            }
        )
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        desc = tree.find(f".//{{{ns}}}description")
        assert desc is None

    def test_identifier_unchanged_by_default(self):
        opf = _make_opf(identifier="urn:isbn:1234567890")
        config = _make_config()
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        ident = tree.find(f".//{{{ns}}}identifier")
        assert ident is not None
        assert ident.text == "urn:isbn:1234567890"

    def test_identifier_replaced_with_uuid(self):
        opf = _make_opf(identifier="urn:isbn:1234567890")
        config = _make_config(
            output={
                "epub_path": "./book.en.epub",
                "title_suffix": " (EN)",
                "new_identifier": True,
            }
        )
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        ident = tree.find(f".//{{{ns}}}identifier")
        assert ident is not None
        assert ident.text.startswith("urn:uuid:")
        assert ident.text != "urn:isbn:1234567890"

    def test_existing_description_replaced(self):
        opf = _make_opf(
            extra_metadata="    <dc:description>Original description</dc:description>",
        )
        config = _make_config()
        result = update_opf_metadata(opf, config)

        tree = etree.fromstring(result.encode("utf-8"))
        ns = "http://purl.org/dc/elements/1.1/"
        descs = list(tree.iter(f"{{{ns}}}description"))
        assert len(descs) == 1
        assert "Machine translated" in descs[0].text


# ---------------------------------------------------------------------------
# Tests: glossary rendering for ToC
# ---------------------------------------------------------------------------


class TestRenderGlossaryForToc:
    def test_characters_and_places_included(self):
        glossary = _make_glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
                    source="extracted",
                ),
                GlossaryEntity(
                    entity_id="place_000001",
                    category="place",
                    canonical_english="Lugunica",
                    surface_forms=[SurfaceForm(source="ルグニカ", english="Lugunica")],
                    source="extracted",
                ),
                GlossaryEntity(
                    entity_id="concept_000001",
                    category="concept",
                    canonical_english="mana",
                    surface_forms=[SurfaceForm(source="マナ", english="mana")],
                    source="extracted",
                ),
            ]
        )
        cats = ["character", "place", "clan", "title"]
        result = _render_glossary_for_toc(glossary, categories=cats)
        assert "スバル" in result
        assert "Subaru" in result
        assert "ルグニカ" in result
        assert "Lugunica" in result
        # Concept should not be included.
        assert "マナ" not in result

    def test_empty_glossary(self):
        glossary = _make_glossary()
        result = _render_glossary_for_toc(glossary)
        assert "no glossary" in result.lower()

    def test_no_categories_includes_all(self):
        """When categories is None or empty, all entities are included."""
        glossary = _make_glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
                    source="extracted",
                ),
                GlossaryEntity(
                    entity_id="concept_000001",
                    category="concept",
                    canonical_english="mana",
                    surface_forms=[SurfaceForm(source="マナ", english="mana")],
                    source="extracted",
                ),
            ]
        )
        result = _render_glossary_for_toc(glossary, categories=None)
        assert "スバル" in result
        assert "マナ" in result

        result2 = _render_glossary_for_toc(glossary, categories=[])
        assert "スバル" in result2
        assert "マナ" in result2

    def test_custom_categories(self):
        """Custom categories list controls which entries appear."""
        glossary = _make_glossary(
            entities=[
                GlossaryEntity(
                    entity_id="character_000001",
                    category="character",
                    canonical_english="Subaru",
                    surface_forms=[SurfaceForm(source="スバル", english="Subaru")],
                    source="extracted",
                ),
                GlossaryEntity(
                    entity_id="concept_000001",
                    category="concept",
                    canonical_english="mana",
                    surface_forms=[SurfaceForm(source="マナ", english="mana")],
                    source="extracted",
                ),
            ]
        )
        result = _render_glossary_for_toc(glossary, categories=["concept"])
        assert "マナ" in result
        assert "スバル" not in result
