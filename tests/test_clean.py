"""Tests for dao_bridge.clean — HTML cleaning and markdown conversion.

Covers: plain prose, ruby text variants, koboSpan handling (inside and outside
ruby), br tags, bold/italic, headings, hr, images, script/style stripping,
deeply nested divs, and whitespace normalization.
"""


from dao_bridge.clean import clean_spine_item


def _wrap_body(inner_html: str) -> str:
    """Wrap HTML fragment in a minimal XHTML document."""
    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>test</title></head>
<body>
{inner_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Plain prose
# ---------------------------------------------------------------------------


class TestPlainProse:
    def test_single_paragraph(self):
        html = _wrap_body("<p>Hello, world.</p>")
        md = clean_spine_item(html)
        assert "Hello, world." in md

    def test_multiple_paragraphs(self):
        html = _wrap_body("<p>First paragraph.</p><p>Second paragraph.</p>")
        md = clean_spine_item(html)
        assert "First paragraph." in md
        assert "Second paragraph." in md


# ---------------------------------------------------------------------------
# Ruby text
# ---------------------------------------------------------------------------


class TestRubyText:
    def test_simple_ruby(self):
        html = _wrap_body("<p><ruby>漢字<rt>かんじ</rt></ruby></p>")
        md = clean_spine_item(html)
        assert "{漢字|かんじ}" in md

    def test_ruby_with_rb(self):
        html = _wrap_body("<p><ruby><rb>漢字</rb><rt>かんじ</rt></ruby></p>")
        md = clean_spine_item(html)
        assert "{漢字|かんじ}" in md

    def test_ruby_with_rp_fallback(self):
        html = _wrap_body("<p><ruby>漢<rp>(</rp><rt>かん</rt><rp>)</rp></ruby></p>")
        md = clean_spine_item(html)
        assert "{漢|かん}" in md

    def test_multi_segment_ruby(self):
        """<ruby>法<rt>ほう</rt>衣<rt>え</rt></ruby> -> {法|ほう}{衣|え}"""
        html = _wrap_body("<p><ruby>法<rt>ほう</rt>衣<rt>え</rt></ruby></p>")
        md = clean_spine_item(html)
        assert "{法|ほう}" in md
        assert "{衣|え}" in md

    def test_ruby_surrounded_by_text(self):
        html = _wrap_body("<p>前<ruby>漢字<rt>かんじ</rt></ruby>後</p>")
        md = clean_spine_item(html)
        assert "前{漢字|かんじ}後" in md


# ---------------------------------------------------------------------------
# KoboSpan handling
# ---------------------------------------------------------------------------


class TestKoboSpan:
    def test_kobospan_inside_ruby(self):
        """Real-world pattern from Kobo-processed EPUBs."""
        html = _wrap_body(
            '<p><ruby><span class="koboSpan" id="kobo.1.1">痩</span>'
            '<rt><span class="koboSpan" id="kobo.2.1">や</span></rt></ruby>'
            '<span class="koboSpan" id="kobo.3.1">せぎすの男だった。</span></p>'
        )
        md = clean_spine_item(html)
        assert "{痩|や}" in md
        assert "せぎすの男だった。" in md

    def test_kobospan_outside_ruby(self):
        """KoboSpan wrapping plain text outside any ruby element."""
        html = _wrap_body('<p><span class="koboSpan" id="kobo.1.1">普通のテキスト</span></p>')
        md = clean_spine_item(html)
        assert "普通のテキスト" in md
        # The koboSpan wrapper should be gone, only text remains.
        assert "koboSpan" not in md
        assert "kobo.1.1" not in md

    def test_kobospan_multi_segment_ruby(self):
        """Multi-segment ruby with koboSpan wrappers in each part."""
        html = _wrap_body(
            "<p><ruby>"
            '<span class="koboSpan" id="kobo.7.1">法</span>'
            '<rt><span class="koboSpan" id="kobo.8.1">ほう</span></rt>'
            '<span class="koboSpan" id="kobo.9.1">衣</span>'
            '<rt><span class="koboSpan" id="kobo.10.1">え</span></rt>'
            "</ruby></p>"
        )
        md = clean_spine_item(html)
        assert "{法|ほう}" in md
        assert "{衣|え}" in md

    def test_kobospan_adjacent_to_ruby(self):
        """Mix of koboSpan text and ruby in the same paragraph."""
        html = _wrap_body(
            "<p>"
            '<span class="koboSpan" id="kobo.1.1">黒い装束の集団に囲まれるその男は、</span>'
            "<ruby>"
            '<span class="koboSpan" id="kobo.2.1">法</span>'
            '<rt><span class="koboSpan" id="kobo.3.1">ほう</span></rt>'
            '<span class="koboSpan" id="kobo.4.1">衣</span>'
            '<rt><span class="koboSpan" id="kobo.5.1">え</span></rt>'
            "</ruby>"
            '<span class="koboSpan" id="kobo.6.1">に身を包んでいる。</span>'
            "</p>"
        )
        md = clean_spine_item(html)
        assert "黒い装束の集団に囲まれるその男は、" in md
        assert "{法|ほう}" in md
        assert "{衣|え}" in md
        assert "に身を包んでいる。" in md
        assert "koboSpan" not in md


# ---------------------------------------------------------------------------
# BR tags
# ---------------------------------------------------------------------------


class TestBrTags:
    def test_br_produces_line_break(self):
        html = _wrap_body("<p>Line one<br/>Line two</p>")
        md = clean_spine_item(html)
        # markdownify uses two trailing spaces + newline for <br>
        assert "Line one" in md
        assert "Line two" in md


# ---------------------------------------------------------------------------
# Bold / Italic
# ---------------------------------------------------------------------------


class TestBoldItalic:
    def test_bold(self):
        html = _wrap_body("<p><b>bold text</b></p>")
        md = clean_spine_item(html)
        assert "**bold text**" in md

    def test_strong(self):
        html = _wrap_body("<p><strong>strong text</strong></p>")
        md = clean_spine_item(html)
        assert "**strong text**" in md

    def test_italic(self):
        html = _wrap_body("<p><i>italic text</i></p>")
        md = clean_spine_item(html)
        assert "*italic text*" in md

    def test_em(self):
        html = _wrap_body("<p><em>emphasis</em></p>")
        md = clean_spine_item(html)
        assert "*emphasis*" in md


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


class TestHeadings:
    def test_h1(self):
        html = _wrap_body("<h1>Chapter Title</h1>")
        md = clean_spine_item(html)
        assert "# Chapter Title" in md

    def test_h2(self):
        html = _wrap_body("<h2>Section</h2>")
        md = clean_spine_item(html)
        assert "## Section" in md

    def test_h3(self):
        html = _wrap_body("<h3>Subsection</h3>")
        md = clean_spine_item(html)
        assert "### Subsection" in md


# ---------------------------------------------------------------------------
# HR (scene break)
# ---------------------------------------------------------------------------


class TestHr:
    def test_hr_becomes_scene_break(self):
        html = _wrap_body("<p>Before</p><hr/><p>After</p>")
        md = clean_spine_item(html)
        assert "* * *" in md
        assert "Before" in md
        assert "After" in md


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImages:
    def test_img_tag(self):
        html = _wrap_body('<p><img src="images/cover.jpg" alt="Cover"/></p>')
        md = clean_spine_item(html)
        assert "![Cover](images/cover.jpg)" in md


# ---------------------------------------------------------------------------
# Script / style stripping
# ---------------------------------------------------------------------------


class TestScriptStyleStripping:
    def test_script_removed(self):
        html = _wrap_body("<script>alert('xss');</script><p>Content</p>")
        md = clean_spine_item(html)
        assert "alert" not in md
        assert "Content" in md

    def test_style_removed(self):
        html = _wrap_body("<style>body { color: red; }</style><p>Content</p>")
        md = clean_spine_item(html)
        assert "color" not in md
        assert "Content" in md


# ---------------------------------------------------------------------------
# Messy Calibre-like markup
# ---------------------------------------------------------------------------


class TestMessyMarkup:
    def test_deeply_nested_divs(self):
        html = _wrap_body(
            '<div class="calibre1">'
            '  <div class="calibre2">'
            '    <div class="calibre3">'
            "      <p>Deeply nested content.</p>"
            "    </div>"
            "  </div>"
            "</div>"
        )
        md = clean_spine_item(html)
        assert "Deeply nested content." in md

    def test_empty_divs_removed(self):
        html = _wrap_body(
            '<div class="spacer"></div>'
            '<div class="empty"><span class="nothing"></span></div>'
            "<p>Real content.</p>"
        )
        md = clean_spine_item(html)
        assert "Real content." in md
        # Should not have excessive blank lines from empty divs.
        assert "spacer" not in md


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    def test_collapses_blank_lines(self):
        html = _wrap_body("<p>Para 1</p>\n\n\n\n\n<p>Para 2</p>\n\n\n\n\n\n<p>Para 3</p>")
        md = clean_spine_item(html)
        # Should not have more than one blank line between paragraphs.
        assert "\n\n\n" not in md

    def test_trailing_whitespace_stripped(self):
        html = _wrap_body("<p>Text with trailing spaces   </p>")
        md = clean_spine_item(html)
        for line in md.split("\n"):
            if line:  # skip empty lines
                # Allow two trailing spaces for <br> line breaks.
                stripped = line.rstrip(" ")
                spaces = len(line) - len(stripped)
                assert spaces == 0 or spaces == 2
