"""Tests for dao_bridge.rebuild -- EPUB reconstruction via modified copy."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dao_bridge.config import AppConfig
from dao_bridge.rebuild import (
    build_modified_files,
    inject_default_css_link,
    markdown_to_html,
    replace_xhtml_body,
    resolve_zip_path,
    restore_ruby_tags,
    validate_with_epubcheck,
    write_epub_modified_copy,
)
from dao_bridge.schemas import (
    Glossary,
    Manifest,
    ManifestItem,
    TocTranslationResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XHTML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
{head_content}
</head>
<body{body_attrs}>
{body_content}
</body>
</html>
"""

_CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_OPF_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>ja</dc:language>
    <dc:identifier id="uid">urn:uuid:test-1234</dc:identifier>
  </metadata>
  <manifest>
{manifest_items}
  </manifest>
  <spine>
{spine_items}
  </spine>
</package>
"""


def _make_xhtml(
    body_content: str = "<p>Original content</p>",
    head_content: str = '<meta charset="UTF-8"/>',
    body_attrs: str = "",
) -> str:
    if body_attrs:
        body_attrs = " " + body_attrs
    return _XHTML_TEMPLATE.format(
        head_content=head_content,
        body_content=body_content,
        body_attrs=body_attrs,
    )


def _make_config(**overrides) -> AppConfig:
    defaults = {
        "source_epub": "/fake/book.epub",
        "languages": {"source": "ja", "target": "en"},
        "models": {
            "classify": {"base_url": "http://localhost:8080/v1", "api_key": "x", "model": "test"},
            "glossary": {"base_url": "http://localhost:8080/v1", "api_key": "x", "model": "test"},
            "translate": {"base_url": "http://localhost:8080/v1", "api_key": "x", "model": "test"},
        },
        "output": {
            "epub_path": "./book.en.epub",
            "title_suffix": " (EN)",
            "new_identifier": False,
            "css": "original",
            "add_translation_note": False,
            "validate": False,
        },
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _make_mini_epub(
    tmp_path: Path,
    spine_items: list[tuple[str, str]] | None = None,
    opf_dir: str = "OEBPS",
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Create a minimal EPUB ZIP for testing.

    Parameters
    ----------
    spine_items:
        List of ``(href, xhtml_content)`` tuples.  The href is relative
        to the OPF directory.
    extra_files:
        Additional ZIP entries to add ``{zip_path: content_bytes}``.
    """
    if spine_items is None:
        spine_items = [("text/ch1.xhtml", _make_xhtml())]

    epub_path = tmp_path / "source.epub"

    manifest_lines = []
    spine_lines = []
    for i, (href, _) in enumerate(spine_items):
        item_id = f"item{i}"
        manifest_lines.append(
            f'    <item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>'
        )
        spine_lines.append(f'    <itemref idref="{item_id}"/>')

    opf = _OPF_TEMPLATE.format(
        manifest_items="\n".join(manifest_lines),
        spine_items="\n".join(spine_lines),
    )

    opf_path = f"{opf_dir}/content.opf" if opf_dir else "content.opf"

    with zipfile.ZipFile(epub_path, "w") as zf:
        # mimetype first, uncompressed.
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, "application/epub+zip")

        # container.xml
        zf.writestr("META-INF/container.xml", _CONTAINER_XML.format(opf_path=opf_path))

        # OPF
        zf.writestr(opf_path, opf)

        # Spine items.
        for href, content in spine_items:
            zip_path = f"{opf_dir}/{href}" if opf_dir else href
            zf.writestr(zip_path, content)

        # Extra files.
        if extra_files:
            for zp, data in extra_files.items():
                zf.writestr(zp, data)

    return epub_path


def _make_manifest(
    epub_path: str,
    spine_items: list[dict],
    opf_dir: str = "OEBPS",
) -> Manifest:
    items = []
    for s in spine_items:
        items.append(ManifestItem(**s))
    return Manifest(
        source_epub_path=epub_path,
        book_id="test",
        opf_dir=opf_dir,
        spine=items,
    )


# ---------------------------------------------------------------------------
# Tests: resolve_zip_path
# ---------------------------------------------------------------------------


class TestResolveZipPath:
    def test_opf_in_subdirectory(self):
        assert resolve_zip_path("OEBPS", "Text/chapter1.xhtml") == "OEBPS/Text/chapter1.xhtml"

    def test_opf_at_root(self):
        assert resolve_zip_path("", "chapter1.xhtml") == "chapter1.xhtml"

    def test_normalize_dotdot(self):
        assert resolve_zip_path("OEBPS", "../images/img.png") == "images/img.png"

    def test_normalize_dot(self):
        assert resolve_zip_path("OEBPS", "./Text/ch.xhtml") == "OEBPS/Text/ch.xhtml"


# ---------------------------------------------------------------------------
# Tests: markdown_to_html
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    def test_plain_paragraphs(self):
        html = markdown_to_html("First paragraph.\n\nSecond paragraph.")
        assert "<p>First paragraph.</p>" in html
        assert "<p>Second paragraph.</p>" in html

    def test_headings(self):
        html = markdown_to_html("# Heading 1\n\n## Heading 2\n\n### Heading 3")
        assert "<h1>Heading 1</h1>" in html
        assert "<h2>Heading 2</h2>" in html
        assert "<h3>Heading 3</h3>" in html

    def test_bold_and_italic(self):
        html = markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_scene_break(self):
        html = markdown_to_html("Before.\n\n* * *\n\nAfter.")
        assert "<hr/>" in html
        # Should not contain the text "* * *" in a paragraph.
        assert ">* * *<" not in html

    def test_ruby_notation(self):
        html = markdown_to_html("The word {漢字|かんじ} is here.")
        assert "<ruby>漢字<rt>かんじ</rt></ruby>" in html
        # Placeholder should not remain.
        assert "RUBYDBT" not in html

    def test_multiple_ruby_in_one_paragraph(self):
        html = markdown_to_html("{魔法|まほう} and {剣|けん} are common.")
        assert "<ruby>魔法<rt>まほう</rt></ruby>" in html
        assert "<ruby>剣<rt>けん</rt></ruby>" in html

    def test_br_self_closing(self):
        # Two trailing spaces = hard line break in markdown.
        html = markdown_to_html("Line one  \nLine two")
        assert "<br/>" in html
        assert "<br>" not in html.replace("<br/>", "")

    def test_empty_input(self):
        html = markdown_to_html("")
        assert html == ""

    def test_mixed_content(self):
        md = (
            "# Chapter 1\n\n"
            "The hero {スバル|すばる} arrived.\n\n"
            "**Bold** and *italic*.\n\n"
            "* * *\n\n"
            "Next scene."
        )
        html = markdown_to_html(md)
        assert "<h1>Chapter 1</h1>" in html
        assert "<ruby>スバル<rt>すばる</rt></ruby>" in html
        assert "<strong>Bold</strong>" in html
        assert "<em>italic</em>" in html
        assert "<hr/>" in html
        assert "<p>Next scene.</p>" in html


# ---------------------------------------------------------------------------
# Tests: restore_ruby_tags
# ---------------------------------------------------------------------------


class TestRestoreRubyTags:
    def test_single_placeholder(self):
        result = restore_ruby_tags(
            "Hello RUBYDBT0001 world",
            {"RUBYDBT0001": ("漢", "かん")},
        )
        assert result == "Hello <ruby>漢<rt>かん</rt></ruby> world"

    def test_multiple_placeholders(self):
        result = restore_ruby_tags(
            "RUBYDBT0001 and RUBYDBT0002",
            {"RUBYDBT0001": ("A", "a"), "RUBYDBT0002": ("B", "b")},
        )
        assert "<ruby>A<rt>a</rt></ruby>" in result
        assert "<ruby>B<rt>b</rt></ruby>" in result

    def test_no_placeholders(self):
        result = restore_ruby_tags("No placeholders here", {})
        assert result == "No placeholders here"


# ---------------------------------------------------------------------------
# Tests: replace_xhtml_body
# ---------------------------------------------------------------------------


class TestReplaceXhtmlBody:
    def test_simple_replacement(self):
        original = _make_xhtml(body_content="<p>Original Japanese text</p>")
        result = replace_xhtml_body(original, "Translated English text.")
        assert "Translated English text" in result
        assert "Original Japanese text" not in result

    def test_head_preserved(self):
        head = (
            '<meta charset="UTF-8"/>\n'
            '  <link rel="stylesheet" href="styles/main.css"/>\n'
            '  <link rel="stylesheet" href="styles/fonts.css"/>'
        )
        original = _make_xhtml(head_content=head)
        result = replace_xhtml_body(original, "New content.")
        assert "styles/main.css" in result
        assert "styles/fonts.css" in result

    def test_body_attributes_preserved(self):
        original = _make_xhtml(
            body_attrs='class="chapter" epub:type="bodymatter" id="ch1"',
        )
        result = replace_xhtml_body(original, "New content.")
        assert 'class="chapter"' in result
        assert "bodymatter" in result
        assert 'id="ch1"' in result

    def test_ruby_in_markdown(self):
        original = _make_xhtml()
        result = replace_xhtml_body(original, "The {漢字|かんじ} text.")
        assert "<ruby>漢字<rt>かんじ</rt></ruby>" in result

    def test_scene_break_in_markdown(self):
        original = _make_xhtml()
        result = replace_xhtml_body(original, "Before.\n\n* * *\n\nAfter.")
        assert "<hr/>" in result

    def test_headings_in_markdown(self):
        original = _make_xhtml()
        result = replace_xhtml_body(original, "# Chapter Title\n\nParagraph text.")
        assert "Chapter Title" in result
        assert "Paragraph text" in result

    def test_no_body_raises(self):
        with pytest.raises(ValueError, match="No <body> element"):
            replace_xhtml_body("<html><head/></html>", "text")


# ---------------------------------------------------------------------------
# Tests: inject_default_css_link
# ---------------------------------------------------------------------------


class TestInjectDefaultCssLink:
    def test_link_added_to_head(self):
        xhtml = _make_xhtml()
        result = inject_default_css_link(xhtml, "../dao_bridge_default.css")
        assert "dao_bridge_default.css" in result
        assert 'rel="stylesheet"' in result

    def test_existing_links_preserved(self):
        head = '<link rel="stylesheet" href="styles/original.css"/>'
        xhtml = _make_xhtml(head_content=head)
        result = inject_default_css_link(xhtml, "dao_bridge_default.css")
        assert "styles/original.css" in result
        assert "dao_bridge_default.css" in result

    def test_no_head_returns_unchanged(self):
        xhtml = "<html><body><p>text</p></body></html>"
        result = inject_default_css_link(xhtml, "style.css")
        assert result == xhtml  # Unchanged.


# ---------------------------------------------------------------------------
# Tests: write_epub_modified_copy
# ---------------------------------------------------------------------------


class TestWriteEpubModifiedCopy:
    def test_all_source_files_present(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            source_names = set()
            with zipfile.ZipFile(epub, "r") as src:
                source_names = {i.filename for i in src.infolist()}
            output_names = {i.filename for i in zf.infolist()}
            assert source_names == output_names

    def test_unmodified_files_identical(self, tmp_path: Path):
        epub = _make_mini_epub(
            tmp_path,
            extra_files={"OEBPS/images/test.png": b"PNG_DATA_HERE"},
        )
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(epub, "r") as src:
            with zipfile.ZipFile(output, "r") as dst:
                assert dst.read("OEBPS/images/test.png") == src.read("OEBPS/images/test.png")

    def test_modified_files_have_new_content(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        modified = {"OEBPS/text/ch1.xhtml": b"<html><body><p>New content</p></body></html>"}
        write_epub_modified_copy(str(epub), str(output), modified)

        with zipfile.ZipFile(output, "r") as zf:
            content = zf.read("OEBPS/text/ch1.xhtml")
            assert b"New content" in content

    def test_mimetype_is_first_entry(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            assert zf.infolist()[0].filename == "mimetype"

    def test_mimetype_uncompressed(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            mimetype_info = zf.infolist()[0]
            assert mimetype_info.compress_type == zipfile.ZIP_STORED

    def test_mimetype_no_extra_field(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            mimetype_info = zf.infolist()[0]
            assert mimetype_info.extra == b""

    def test_new_files_added(self, tmp_path: Path):
        """Files in modified_files not in source are added as new entries."""
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        modified = {"OEBPS/new_style.css": b"body { color: red; }"}
        write_epub_modified_copy(str(epub), str(output), modified)

        with zipfile.ZipFile(output, "r") as zf:
            assert "OEBPS/new_style.css" in zf.namelist()
            assert zf.read("OEBPS/new_style.css") == b"body { color: red; }"

    def test_per_entry_compress_type_preserved(self, tmp_path: Path):
        """Entries that use ZIP_DEFLATED in source remain ZIP_DEFLATED."""
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(epub, "r") as src:
            with zipfile.ZipFile(output, "r") as dst:
                for src_info in src.infolist():
                    if src_info.filename == "mimetype":
                        continue
                    dst_info = None
                    for di in dst.infolist():
                        if di.filename == src_info.filename:
                            dst_info = di
                            break
                    assert dst_info is not None, f"Missing: {src_info.filename}"
                    assert dst_info.compress_type == src_info.compress_type


# ---------------------------------------------------------------------------
# Tests: build_modified_files
# ---------------------------------------------------------------------------


class TestBuildModifiedFiles:
    def test_translated_items_replaced(self, tmp_path: Path):
        xhtml = _make_xhtml(body_content="<p>Japanese text</p>")
        epub = _make_mini_epub(tmp_path, spine_items=[("text/ch1.xhtml", xhtml)])
        config = _make_config(source_epub=str(epub))

        # Write assembled markdown.
        work = tmp_path / "work"
        work.mkdir()
        asm_dir = work / "assembled"
        asm_dir.mkdir()
        (asm_dir / "0000.md").write_text("English translation.", encoding="utf-8")

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                }
            ],
        )

        modified = build_modified_files(manifest, work, str(epub), config)
        assert "OEBPS/text/ch1.xhtml" in modified
        content = modified["OEBPS/text/ch1.xhtml"].decode("utf-8")
        assert "English translation" in content
        assert "Japanese text" not in content

    def test_non_translated_items_not_in_modified(self, tmp_path: Path):
        xhtml1 = _make_xhtml(body_content="<p>Chapter</p>")
        xhtml2 = _make_xhtml(body_content="<p>Illustration</p>")
        epub = _make_mini_epub(
            tmp_path,
            spine_items=[("text/ch1.xhtml", xhtml1), ("text/illust.xhtml", xhtml2)],
        )
        config = _make_config(source_epub=str(epub))

        work = tmp_path / "work"
        work.mkdir()
        asm_dir = work / "assembled"
        asm_dir.mkdir()
        (asm_dir / "0000.md").write_text("Translated.", encoding="utf-8")

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                },
                {
                    "spine_index": 1,
                    "original_href": "text/illust.xhtml",
                    "raw_path": "raw/0001.xhtml",
                    "chunk_count": 0,
                },
            ],
        )

        modified = build_modified_files(manifest, work, str(epub), config)
        assert "OEBPS/text/ch1.xhtml" in modified
        assert "OEBPS/text/illust.xhtml" not in modified

    def test_missing_assembled_raises(self, tmp_path: Path):
        epub = _make_mini_epub(tmp_path)
        config = _make_config(source_epub=str(epub))

        work = tmp_path / "work"
        work.mkdir()
        (work / "assembled").mkdir()
        # No assembled file written.

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                }
            ],
        )

        with pytest.raises(FileNotFoundError, match="Assembled file missing"):
            build_modified_files(manifest, work, str(epub), config)


# ---------------------------------------------------------------------------
# Tests: CSS handling
# ---------------------------------------------------------------------------


class TestCssHandling:
    def test_css_original_no_modifications(self, tmp_path: Path):
        xhtml = _make_xhtml()
        epub = _make_mini_epub(tmp_path, spine_items=[("text/ch1.xhtml", xhtml)])
        config = _make_config(
            source_epub=str(epub),
            output={"epub_path": "./out.epub", "css": "original", "title_suffix": ""},
        )

        work = tmp_path / "work"
        work.mkdir()
        (work / "assembled").mkdir()
        (work / "assembled" / "0000.md").write_text("Text.", encoding="utf-8")

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                }
            ],
        )

        modified = build_modified_files(manifest, work, str(epub), config)
        content = modified["OEBPS/text/ch1.xhtml"].decode("utf-8")
        assert "dao_bridge_default.css" not in content

    def test_css_default_link_injected(self, tmp_path: Path):
        xhtml = _make_xhtml()
        epub = _make_mini_epub(tmp_path, spine_items=[("text/ch1.xhtml", xhtml)])
        config = _make_config(
            source_epub=str(epub),
            output={"epub_path": "./out.epub", "css": "default", "title_suffix": ""},
        )

        work = tmp_path / "work"
        work.mkdir()
        (work / "assembled").mkdir()
        (work / "assembled" / "0000.md").write_text("Text.", encoding="utf-8")

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                }
            ],
        )

        modified = build_modified_files(manifest, work, str(epub), config)
        content = modified["OEBPS/text/ch1.xhtml"].decode("utf-8")
        assert "dao_bridge_default.css" in content


# ---------------------------------------------------------------------------
# Tests: validate_with_epubcheck
# ---------------------------------------------------------------------------


class TestValidateWithEpubcheck:
    @patch("dao_bridge.rebuild.shutil.which", return_value=None)
    def test_not_found_returns_true(self, mock_which):
        assert validate_with_epubcheck("/fake/book.epub") is True

    @patch("dao_bridge.rebuild.subprocess.run")
    @patch("dao_bridge.rebuild.shutil.which", return_value="/usr/bin/epubcheck")
    def test_success_returns_true(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        assert validate_with_epubcheck("/fake/book.epub") is True

    @patch("dao_bridge.rebuild.subprocess.run")
    @patch("dao_bridge.rebuild.shutil.which", return_value="/usr/bin/epubcheck")
    def test_failure_returns_false(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Errors found")
        assert validate_with_epubcheck("/fake/book.epub") is False


# ---------------------------------------------------------------------------
# Tests: Passthrough items
# ---------------------------------------------------------------------------


class TestPassthroughItems:
    def test_illustration_unchanged_in_output(self, tmp_path: Path):
        """Non-translated spine items copy through byte-identical."""
        illustration_xhtml = _make_xhtml(body_content='<img src="../images/illust.png"/>')
        chapter_xhtml = _make_xhtml(body_content="<p>Original chapter</p>")
        epub = _make_mini_epub(
            tmp_path,
            spine_items=[
                ("text/ch1.xhtml", chapter_xhtml),
                ("text/illust.xhtml", illustration_xhtml),
            ],
        )

        work = tmp_path / "work"
        work.mkdir()
        (work / "assembled").mkdir()
        (work / "assembled" / "0000.md").write_text("Translated chapter.", encoding="utf-8")

        manifest = _make_manifest(
            str(epub),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                },
                {
                    "spine_index": 1,
                    "original_href": "text/illust.xhtml",
                    "raw_path": "raw/0001.xhtml",
                    "chunk_count": 0,
                },
            ],
        )

        config = _make_config(source_epub=str(epub))
        modified = build_modified_files(manifest, work, str(epub), config)

        # Illustration should NOT be in modified (passes through unchanged at ZIP level).
        assert "OEBPS/text/illust.xhtml" not in modified

        # Write output and verify illustration is byte-identical.
        output = tmp_path / "output.epub"
        write_epub_modified_copy(str(epub), str(output), modified)

        with zipfile.ZipFile(epub, "r") as src:
            with zipfile.ZipFile(output, "r") as dst:
                assert dst.read("OEBPS/text/illust.xhtml") == src.read("OEBPS/text/illust.xhtml")


# ---------------------------------------------------------------------------
# Tests: run_rebuild_stage
# ---------------------------------------------------------------------------


class TestRunRebuildStage:
    def _setup_rebuild(
        self, tmp_path: Path, epub_path: Path | None = None
    ) -> tuple[Path, AppConfig]:
        """Set up a work directory ready for rebuild."""
        work = tmp_path / "work"
        work.mkdir()
        (work / "assembled").mkdir()
        (work / "assembled" / "0000.md").write_text("Translated text.", encoding="utf-8")
        (work / "logs").mkdir()

        if epub_path is None:
            epub_path = _make_mini_epub(tmp_path)

        config = _make_config(
            source_epub=str(epub_path),
            work_dir=str(work),
        )

        manifest = _make_manifest(
            str(epub_path),
            [
                {
                    "spine_index": 0,
                    "original_href": "text/ch1.xhtml",
                    "raw_path": "raw/0000.xhtml",
                    "chunk_count": 1,
                }
            ],
        )
        (work / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

        # State file.
        from dao_bridge.state import PipelineState, save_state, RunState

        state = PipelineState(
            run=RunState(source_epub=str(epub_path), started_at="", status="initialised")
        )
        save_state(work, state)

        # Glossary.
        glossary = Glossary(
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        (work / "glossary.json").write_text(glossary.model_dump_json(), encoding="utf-8")

        # Config file.
        import yaml

        (work / "config.yaml").write_text(
            yaml.dump(json.loads(config.model_dump_json())),
            encoding="utf-8",
        )

        return work, config

    @patch("dao_bridge.rebuild.translate_toc", return_value={})
    def test_happy_path(self, mock_toc, tmp_path: Path):
        work, config = self._setup_rebuild(tmp_path)

        from dao_bridge.rebuild import run_rebuild_stage

        run_rebuild_stage(work, config, force=False)

        # Stage should be marked completed.
        from dao_bridge.state import load_state, is_stage_completed

        state = load_state(work)
        assert is_stage_completed(state, "rebuild")

        # Output EPUB should exist.
        output_path = (work.parent / config.output.epub_path).resolve()
        assert output_path.exists()

    @patch("dao_bridge.rebuild.translate_toc", return_value={})
    def test_missing_assembled_raises(self, mock_toc, tmp_path: Path):
        work, config = self._setup_rebuild(tmp_path)
        # Remove assembled file.
        (work / "assembled" / "0000.md").unlink()

        from dao_bridge.rebuild import run_rebuild_stage

        with pytest.raises(FileNotFoundError, match="Missing assembled files"):
            run_rebuild_stage(work, config, force=False)

        # Stage should be marked failed.
        from dao_bridge.state import load_state

        state = load_state(work)
        assert state.stages.get("rebuild", {})

    @patch("dao_bridge.rebuild.translate_toc", return_value={})
    def test_missing_source_epub_raises(self, mock_toc, tmp_path: Path):
        work, config = self._setup_rebuild(tmp_path)
        # Delete source EPUB.
        Path(config.source_epub).unlink()

        from dao_bridge.rebuild import run_rebuild_stage

        with pytest.raises(FileNotFoundError, match="Source EPUB not found"):
            run_rebuild_stage(work, config, force=False)

    @patch("dao_bridge.rebuild.translate_toc", return_value={})
    def test_force_reruns(self, mock_toc, tmp_path: Path):
        work, config = self._setup_rebuild(tmp_path)

        from dao_bridge.rebuild import run_rebuild_stage

        # First run.
        run_rebuild_stage(work, config, force=False)

        # Second run without force -- should skip.
        run_rebuild_stage(work, config, force=False)
        assert mock_toc.call_count == 1  # Only called once.

        # Third run with force -- should re-run.
        run_rebuild_stage(work, config, force=True)
        assert mock_toc.call_count == 2

    @patch("dao_bridge.rebuild.translate_toc", return_value={})
    def test_already_completed_skips(self, mock_toc, tmp_path: Path):
        work, config = self._setup_rebuild(tmp_path)

        from dao_bridge.rebuild import run_rebuild_stage

        run_rebuild_stage(work, config, force=False)
        mock_toc.reset_mock()

        run_rebuild_stage(work, config, force=False)
        mock_toc.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: ZIP-level integrity
# ---------------------------------------------------------------------------


class TestZipIntegrity:
    def test_unusual_zip_structure_preserved(self, tmp_path: Path):
        """Extra directories and non-standard paths copy through."""
        epub = _make_mini_epub(
            tmp_path,
            extra_files={
                "OEBPS/some/deep/path/data.xml": b"<data/>",
                "META-INF/encryption.xml": b"<encryption/>",
            },
        )
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            assert "OEBPS/some/deep/path/data.xml" in zf.namelist()
            assert "META-INF/encryption.xml" in zf.namelist()

    def test_very_small_epub(self, tmp_path: Path):
        """EPUB with one chapter, no images, no ToC."""
        epub = _make_mini_epub(tmp_path)
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            assert "mimetype" in zf.namelist()

    def test_opf_in_subdirectory(self, tmp_path: Path):
        """EPUB with OPF in OEBPS subdirectory."""
        epub = _make_mini_epub(tmp_path, opf_dir="OEBPS")
        output = tmp_path / "output.epub"

        write_epub_modified_copy(str(epub), str(output), {})

        with zipfile.ZipFile(output, "r") as zf:
            assert "OEBPS/content.opf" in zf.namelist()
