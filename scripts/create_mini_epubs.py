#!/usr/bin/env python3
"""
Create mini test EPUBs from the full Re:Zero Vol 5 English and Japanese EPUBs.

Produces two small, structurally valid EPUBs containing:
- Cover page + tiny placeholder cover image
- 1 illustration page + tiny placeholder image
- Title page
- Table of contents (nav + ncx, trimmed)
- Prologue (full)
- Chapter 1 opening (~20-25 paragraphs)
- Chapter 3 opening (~20-25 paragraphs)

All full-size images are replaced with tiny placeholder JPEGs.
Unused CSS, JS, and content files are stripped.

Usage:
    python scripts/create_mini_epubs.py

Expects the source EPUBs in the repo root:
    ReZero-Vol5-eng.epub
    ReZero-Vol5-jp.epub

Outputs to:
    tests/fixtures/ReZero-Vol5-mini-eng.epub
    tests/fixtures/ReZero-Vol5-mini-jp.epub
"""

import struct
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Tiny placeholder JPEG generator
# ---------------------------------------------------------------------------


def make_tiny_jpeg(
    width: int = 8, height: int = 8, color: tuple[int, int, int] = (128, 128, 128)
) -> bytes:
    """Create a minimal valid JPEG file (solid color, very small).

    Uses a hand-crafted minimal JFIF with a single 8x8 MCU.
    For simplicity, we use a known-good minimal JPEG binary approach.
    """
    # Minimal valid JPEG: 1x1 pixel, grayscale-ish.  This is a known minimal
    # JPEG byte sequence that every reader accepts.  ~285 bytes.
    # We'll use struct-based construction for a truly tiny JPEG.

    # Actually the simplest approach: use a pre-built minimal JPEG.
    # This is a valid 1x1 white JPEG (107 bytes).
    return bytes(
        [
            0xFF,
            0xD8,
            0xFF,
            0xE0,
            0x00,
            0x10,
            0x4A,
            0x46,
            0x49,
            0x46,
            0x00,
            0x01,
            0x01,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x00,
            0xFF,
            0xDB,
            0x00,
            0x43,
            0x00,
            0x08,
            0x06,
            0x06,
            0x07,
            0x06,
            0x05,
            0x08,
            0x07,
            0x07,
            0x07,
            0x09,
            0x09,
            0x08,
            0x0A,
            0x0C,
            0x14,
            0x0D,
            0x0C,
            0x0B,
            0x0B,
            0x0C,
            0x19,
            0x12,
            0x13,
            0x0F,
            0x14,
            0x1D,
            0x1A,
            0x1F,
            0x1E,
            0x1D,
            0x1A,
            0x1C,
            0x1C,
            0x20,
            0x24,
            0x2E,
            0x27,
            0x20,
            0x22,
            0x2C,
            0x23,
            0x1C,
            0x1C,
            0x28,
            0x37,
            0x29,
            0x2C,
            0x30,
            0x31,
            0x34,
            0x34,
            0x34,
            0x1F,
            0x27,
            0x39,
            0x3D,
            0x38,
            0x32,
            0x3C,
            0x2E,
            0x33,
            0x34,
            0x32,
            0xFF,
            0xC0,
            0x00,
            0x0B,
            0x08,
            0x00,
            0x01,
            0x00,
            0x01,
            0x01,
            0x01,
            0x11,
            0x00,
            0xFF,
            0xC4,
            0x00,
            0x1F,
            0x00,
            0x00,
            0x01,
            0x05,
            0x01,
            0x01,
            0x01,
            0x01,
            0x01,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
            0x09,
            0x0A,
            0x0B,
            0xFF,
            0xC4,
            0x00,
            0xB5,
            0x10,
            0x00,
            0x02,
            0x01,
            0x03,
            0x03,
            0x02,
            0x04,
            0x03,
            0x05,
            0x05,
            0x04,
            0x04,
            0x00,
            0x00,
            0x01,
            0x7D,
            0x01,
            0x02,
            0x03,
            0x00,
            0x04,
            0x11,
            0x05,
            0x12,
            0x21,
            0x31,
            0x41,
            0x06,
            0x13,
            0x51,
            0x61,
            0x07,
            0x22,
            0x71,
            0x14,
            0x32,
            0x81,
            0x91,
            0xA1,
            0x08,
            0x23,
            0x42,
            0xB1,
            0xC1,
            0x15,
            0x52,
            0xD1,
            0xF0,
            0x24,
            0x33,
            0x62,
            0x72,
            0x82,
            0x09,
            0x0A,
            0x16,
            0x17,
            0x18,
            0x19,
            0x1A,
            0x25,
            0x26,
            0x27,
            0x28,
            0x29,
            0x2A,
            0x34,
            0x35,
            0x36,
            0x37,
            0x38,
            0x39,
            0x3A,
            0x43,
            0x44,
            0x45,
            0x46,
            0x47,
            0x48,
            0x49,
            0x4A,
            0x53,
            0x54,
            0x55,
            0x56,
            0x57,
            0x58,
            0x59,
            0x5A,
            0x63,
            0x64,
            0x65,
            0x66,
            0x67,
            0x68,
            0x69,
            0x6A,
            0x73,
            0x74,
            0x75,
            0x76,
            0x77,
            0x78,
            0x79,
            0x7A,
            0x83,
            0x84,
            0x85,
            0x86,
            0x87,
            0x88,
            0x89,
            0x8A,
            0x92,
            0x93,
            0x94,
            0x95,
            0x96,
            0x97,
            0x98,
            0x99,
            0x9A,
            0xA2,
            0xA3,
            0xA4,
            0xA5,
            0xA6,
            0xA7,
            0xA8,
            0xA9,
            0xAA,
            0xB2,
            0xB3,
            0xB4,
            0xB5,
            0xB6,
            0xB7,
            0xB8,
            0xB9,
            0xBA,
            0xC2,
            0xC3,
            0xC4,
            0xC5,
            0xC6,
            0xC7,
            0xC8,
            0xC9,
            0xCA,
            0xD2,
            0xD3,
            0xD4,
            0xD5,
            0xD6,
            0xD7,
            0xD8,
            0xD9,
            0xDA,
            0xE1,
            0xE2,
            0xE3,
            0xE4,
            0xE5,
            0xE6,
            0xE7,
            0xE8,
            0xE9,
            0xEA,
            0xF1,
            0xF2,
            0xF3,
            0xF4,
            0xF5,
            0xF6,
            0xF7,
            0xF8,
            0xF9,
            0xFA,
            0xFF,
            0xDA,
            0x00,
            0x08,
            0x01,
            0x01,
            0x00,
            0x00,
            0x3F,
            0x00,
            0x7B,
            0x94,
            0x11,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x1F,
            0xFF,
            0xD9,
        ]
    )


# ---------------------------------------------------------------------------
# XHTML truncation helpers
# ---------------------------------------------------------------------------


def truncate_xhtml_paragraphs(xhtml: str, max_content_paragraphs: int) -> str:
    """Truncate an XHTML chapter file to keep only the first N content paragraphs.

    A 'content paragraph' is a <p> tag that contains actual text (not just <br/> or whitespace).
    We keep the full <head>, opening tags, and close everything properly.
    """
    # Strategy: find all <p ...>...</p> spans, classify them, keep only
    # the first max_content_paragraphs that have real text content.
    # Then rebuild by keeping everything up to and including the last kept <p>,
    # and properly closing all open tags.

    import re

    # Find body start
    body_match = re.search(r"<body[^>]*>", xhtml)
    if not body_match:
        return xhtml

    head_and_open = xhtml[: body_match.end()]

    # Everything after <body...>
    body_rest = xhtml[body_match.end() :]

    # Find all <p> elements (non-greedy match for content)
    # Handle both self-contained <p>...</p> patterns
    p_pattern = re.compile(r"(<p[^>]*>)(.*?)(</p>)", re.DOTALL)

    content_para_count = 0
    last_kept_end = 0  # position in body_rest

    for m in p_pattern.finditer(body_rest):
        inner = m.group(2)
        # Strip tags to check for actual text
        text = re.sub(r"<[^>]+>", "", inner).strip()

        if not text:
            # Empty paragraph (just <br/> etc.) - keep it, don't count
            last_kept_end = m.end()
            continue

        content_para_count += 1
        last_kept_end = m.end()

        if content_para_count >= max_content_paragraphs:
            break

    # Take everything up to the last kept paragraph
    kept_body = body_rest[:last_kept_end]

    # Now close all open tags.  Find what wrapper divs/sections are open.
    # Simple approach: find all opening div/section tags before body content
    # and close them in reverse order.
    open_tags = []
    tag_pattern = re.compile(r"<(/?)(div|section|nav)(?:\s[^>]*)?\s*/?>")
    for tm in tag_pattern.finditer(kept_body):
        if tm.group(1) == "/":
            if open_tags and open_tags[-1] == tm.group(2):
                open_tags.pop()
        else:
            # Check it's not self-closing
            if not tm.group(0).endswith("/>"):
                open_tags.append(tm.group(2))

    closing = "\n" + "\n".join(f"</{tag}>" for tag in reversed(open_tags))
    closing += "\n</body>\n</html>"

    return head_and_open + kept_body + closing


# ---------------------------------------------------------------------------
# English mini EPUB builder
# ---------------------------------------------------------------------------


def build_english_mini():
    """Build the mini English EPUB."""
    src_path = REPO_ROOT / "ReZero-Vol5-eng.epub"
    out_path = FIXTURES_DIR / "ReZero-Vol5-mini-eng.epub"

    placeholder_jpg = make_tiny_jpeg()

    # Files to copy as-is from the source
    copy_as_is = {
        "mimetype",
        "META-INF/container.xml",
        "OEBPS/css/stylesheet.css",
    }

    # Files where we replace the image with a placeholder
    image_placeholders = {
        "OEBPS/images/9780316398466.jpg": placeholder_jpg,  # cover
        "OEBPS/images/Art_insert001.jpg": placeholder_jpg,  # illustration
    }

    # XHTML files to copy as-is
    xhtml_copy = {
        "OEBPS/cover.xhtml",
        "OEBPS/insert001.xhtml",  # illustration page
        "OEBPS/titlepage.xhtml",
        "OEBPS/prologue.xhtml",  # full prologue
    }

    # XHTML files to truncate
    xhtml_truncate = {
        "OEBPS/chapter001.xhtml": 20,  # first 20 content paragraphs
        "OEBPS/chapter003.xhtml": 20,  # first 20 content paragraphs
    }

    # We also need a tiny placeholder for the titlepage image
    image_placeholders["OEBPS/images/Art_tit.jpg"] = placeholder_jpg

    # Build modified OPF
    new_opf = _build_english_opf()

    # Build modified toc.xhtml (nav)
    new_toc_nav = _build_english_nav()

    # Build modified toc.ncx
    new_toc_ncx = _build_english_ncx()

    modified_files: dict[str, bytes] = {}
    modified_files["OEBPS/package.opf"] = new_opf.encode("utf-8")
    modified_files["OEBPS/toc.xhtml"] = new_toc_nav.encode("utf-8")
    modified_files["OEBPS/toc.ncx"] = new_toc_ncx.encode("utf-8")

    for img_path, img_bytes in image_placeholders.items():
        modified_files[img_path] = img_bytes

    # All files that should be in the output
    all_output_files = (
        copy_as_is
        | xhtml_copy
        | set(image_placeholders.keys())
        | set(modified_files.keys())
        | set(xhtml_truncate.keys())
    )

    with zipfile.ZipFile(src_path, "r") as src:
        with zipfile.ZipFile(out_path, "w") as dst:
            # Write mimetype FIRST, uncompressed, no extra field
            mi = zipfile.ZipInfo("mimetype")
            mi.compress_type = zipfile.ZIP_STORED
            mi.extra = b""
            dst.writestr(mi, "application/epub+zip")

            for item in src.infolist():
                if item.filename == "mimetype":
                    continue  # already written

                if item.filename in modified_files:
                    new_info = item
                    new_info.extra = item.extra  # preserve
                    dst.writestr(new_info, modified_files[item.filename])
                elif item.filename in xhtml_truncate:
                    content = src.read(item.filename).decode("utf-8")
                    truncated = truncate_xhtml_paragraphs(content, xhtml_truncate[item.filename])
                    dst.writestr(item, truncated.encode("utf-8"))
                elif item.filename in copy_as_is or item.filename in xhtml_copy:
                    dst.writestr(item, src.read(item.filename))
                # else: skip this file

    print(f"Created: {out_path}")
    print(f"  Size: {out_path.stat().st_size:,} bytes")


def _build_english_opf() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" xml:lang="en" unique-identifier="pub-id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Re:ZERO -Starting Life in Another World-, Vol. 5 (Mini Test)</dc:title>
<dc:creator id="creator01">Tappei Nagatsuki</dc:creator>
<meta refines="#creator01" property="role" scheme="marc:relators">aut</meta>
<dc:publisher>Yen On</dc:publisher>
<dc:date>2017-10-31</dc:date>
<dc:rights>Test fixture - truncated excerpts for development testing only</dc:rights>
<meta property="dcterms:modified">2025-01-01T00:00:00Z</meta>
<dc:language>en</dc:language>
<meta name="cover" content="cover-image"/>
<dc:identifier id="pub-id">urn:uuid:4c88169e-d76b-49a0-b0a1-8c0beeb2e67f</dc:identifier>
</metadata>
<manifest>
<item href="cover.xhtml" id="cover" media-type="application/xhtml+xml"/>
<item href="insert001.xhtml" id="insert001" media-type="application/xhtml+xml"/>
<item href="titlepage.xhtml" id="titlepage" media-type="application/xhtml+xml"/>
<item href="prologue.xhtml" id="prologue" media-type="application/xhtml+xml"/>
<item href="chapter001.xhtml" id="chapter001" media-type="application/xhtml+xml"/>
<item href="chapter003.xhtml" id="chapter003" media-type="application/xhtml+xml"/>
<item id="cover-image" properties="cover-image" href="images/9780316398466.jpg" media-type="image/jpeg"/>
<item href="images/Art_insert001.jpg" id="aArt_insert001" media-type="image/jpeg"/>
<item href="images/Art_tit.jpg" id="aArt_tit" media-type="image/jpeg"/>
<item id="style" href="css/stylesheet.css" media-type="text/css"/>
<item id="toc" properties="nav" href="toc.xhtml" media-type="application/xhtml+xml"/>
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
</manifest>
<spine page-progression-direction="ltr" toc="ncx">
<itemref idref="cover" linear="no"/>
<itemref idref="insert001" linear="yes"/>
<itemref idref="titlepage" linear="yes"/>
<itemref idref="toc" linear="yes"/>
<itemref idref="prologue" linear="yes"/>
<itemref idref="chapter001" linear="yes"/>
<itemref idref="chapter003" linear="yes"/>
</spine>
<guide>
<reference type="cover" title="Cover Image" href="cover.xhtml"/>
<reference type="toc" title="Table of Contents" href="toc.xhtml"/>
<reference type="text" title="Begin Reading" href="insert001.xhtml"/>
</guide>
</package>"""


def _build_english_nav() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
<meta content="text/html; charset=utf-8" http-equiv="default-style"/>
<link rel="stylesheet" href="css/stylesheet.css" type="text/css"/>
</head>
<body>
<section epub:type="bodymatter chapter">
<nav epub:type="toc">
<h1>Contents</h1>
<ol>
<li><a href="cover.xhtml">Cover</a></li>
<li><a href="insert001.xhtml">Insert</a></li>
<li><a href="titlepage.xhtml">Title Page</a></li>
<li><a href="prologue.xhtml">Prologue: And His Name Is\u2014</a></li>
<li><a href="chapter001.xhtml">Chapter 1: A Decaying Mind</a></li>
<li><a href="chapter003.xhtml">Chapter 3: A Disease Called Despair</a></li>
</ol>
</nav>
<nav epub:type="landmarks" class="hidden-tag" hidden="hidden">
<h1>Navigation</h1>
<ol>
<li><a epub:type="bodymatter" href="insert001.xhtml">Begin Reading</a></li>
<li><a epub:type="toc" href="toc.xhtml">Table of Contents</a></li>
</ol>
</nav>
</section>
</body>
</html>"""


def _build_english_ncx() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head>
<meta content="9780316398466" name="dtb:uid"/>
<meta content="1" name="dtb:depth"/>
</head>
<docTitle><text>Re:ZERO -Starting Life in Another World-, Vol. 5 (Mini Test)</text></docTitle>
<docAuthor><text>Tappei Nagatsuki</text></docAuthor>
<navMap>
<navPoint id="ncx-1" playOrder="1"><navLabel><text>Cover</text></navLabel><content src="cover.xhtml"/></navPoint>
<navPoint id="ncx-2" playOrder="2"><navLabel><text>Insert</text></navLabel><content src="insert001.xhtml"/></navPoint>
<navPoint id="ncx-3" playOrder="3"><navLabel><text>Title Page</text></navLabel><content src="titlepage.xhtml"/></navPoint>
<navPoint id="ncx-4" playOrder="4"><navLabel><text>Table of Contents</text></navLabel><content src="toc.xhtml"/></navPoint>
<navPoint id="ncx-5" playOrder="5"><navLabel><text>Prologue: And His Name Is\u2014</text></navLabel><content src="prologue.xhtml"/></navPoint>
<navPoint id="ncx-6" playOrder="6"><navLabel><text>Chapter 1: A Decaying Mind</text></navLabel><content src="chapter001.xhtml"/></navPoint>
<navPoint id="ncx-7" playOrder="7"><navLabel><text>Chapter 3: A Disease Called Despair</text></navLabel><content src="chapter003.xhtml"/></navPoint>
</navMap>
</ncx>"""


# ---------------------------------------------------------------------------
# Japanese mini EPUB builder
# ---------------------------------------------------------------------------


def build_japanese_mini():
    """Build the mini Japanese EPUB."""
    src_path = REPO_ROOT / "ReZero-Vol5-jp.epub"
    out_path = FIXTURES_DIR / "ReZero-Vol5-mini-jp.epub"

    placeholder_jpg = make_tiny_jpeg()

    # Files to copy as-is
    copy_as_is = {
        "mimetype",
        "META-INF/container.xml",
        "item/style/book-style.css",
        "item/style/style-standard.css",
        "item/style/fixed-layout-jp.css",
    }

    # Image placeholders
    image_placeholders = {
        "item/image/cover.jpg": placeholder_jpg,
        "item/image/kuchie02.jpg": placeholder_jpg,  # illustration in p-001
    }

    # XHTML to copy as-is
    xhtml_copy = {
        "item/xhtml/p-cover.xhtml",
        "item/xhtml/p-001.xhtml",  # illustration (kuchie02)
        "item/xhtml/p-titlepage.xhtml",
    }

    # XHTML to truncate
    xhtml_truncate = {
        "item/xhtml/p-006.xhtml": 25,  # ch1 - first 25 content paragraphs
        "item/xhtml/p-014.xhtml": 25,  # ch3 - first 25 content paragraphs
    }

    # Build new metadata files
    new_opf = _build_japanese_opf()
    new_nav = _build_japanese_nav()
    new_ncx = _build_japanese_ncx()
    new_toc_page = _build_japanese_toc_xhtml()

    modified_files: dict[str, bytes] = {}
    modified_files["item/standard.opf"] = new_opf.encode("utf-8")
    modified_files["item/navigation-documents.xhtml"] = new_nav.encode("utf-8")
    modified_files["item/toc.ncx"] = new_ncx.encode("utf-8")
    modified_files["item/xhtml/p-toc-001.xhtml"] = new_toc_page.encode("utf-8")

    for img_path, img_bytes in image_placeholders.items():
        modified_files[img_path] = img_bytes

    # The prologue (p-005) is small enough to include in full
    xhtml_copy.add("item/xhtml/p-005.xhtml")

    with zipfile.ZipFile(src_path, "r") as src:
        with zipfile.ZipFile(out_path, "w") as dst:
            # mimetype FIRST, uncompressed
            mi = zipfile.ZipInfo("mimetype")
            mi.compress_type = zipfile.ZIP_STORED
            mi.extra = b""
            dst.writestr(mi, "application/epub+zip")

            for item in src.infolist():
                if item.filename == "mimetype":
                    continue

                if item.filename in modified_files:
                    dst.writestr(item, modified_files[item.filename])
                elif item.filename in xhtml_truncate:
                    content = src.read(item.filename).decode("utf-8")
                    truncated = truncate_xhtml_paragraphs(content, xhtml_truncate[item.filename])
                    dst.writestr(item, truncated.encode("utf-8"))
                elif item.filename in copy_as_is or item.filename in xhtml_copy:
                    dst.writestr(item, src.read(item.filename))

    print(f"Created: {out_path}")
    print(f"  Size: {out_path.stat().st_size:,} bytes")


def _build_japanese_opf() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" xml:lang="ja" unique-identifier="bw-ecode"
  prefix="rendition: http://www.idpf.org/vocab/rendition/#
          ebpaj: http://www.ebpaj.jp/
          fixed-layout-jp: http://www.digital-comic.jp/
          kadokawa: http://www.kadokawa.co.jp/">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title id="title">\uff32\uff45\uff1a\u30bc\u30ed\u304b\u3089\u59cb\u3081\u308b\u7570\u4e16\u754c\u751f\u6d3b5 (Mini Test)</dc:title>
<meta refines="#title" property="file-as">\u30ea\u30bc\u30ed\u30ab\u30e9\u30cf\u30b8\u30e1\u30eb\u30a4\u30bb\u30ab\u30a4\u30bb\u30a4\u30ab\u30c4005</meta>
<dc:creator id="creator01">\u9577\u6708\u9054\u5e73</dc:creator>
<meta refines="#creator01" property="role" scheme="marc:relators">aut</meta>
<meta refines="#creator01" property="file-as">\u30ca\u30ac\u30c4\u30ad\u30bf\u30c3\u30da\u30a4</meta>
<dc:publisher>KADOKAWA</dc:publisher>
<dc:language>ja</dc:language>
<dc:rights>Test fixture - truncated excerpts for development testing only</dc:rights>
<meta property="dcterms:modified">2025-01-01T00:00:00Z</meta>
<meta property="rendition:layout">reflowable</meta>
<meta property="rendition:spread">auto</meta>
<dc:identifier id="bw-ecode">urn:uuid:198647d7-20df-47b5-9183-aec14335bd71</dc:identifier>
</metadata>
<manifest>
<item media-type="application/xhtml+xml" id="nav" href="navigation-documents.xhtml" properties="nav"/>
<item media-type="text/css" id="fixed-layout-jp" href="style/fixed-layout-jp.css"/>
<item media-type="text/css" id="book-style" href="style/book-style.css"/>
<item media-type="text/css" id="style-standard" href="style/style-standard.css"/>
<item media-type="image/jpeg" id="cover" href="image/cover.jpg" properties="cover-image"/>
<item media-type="image/jpeg" id="i-kuchie02" href="image/kuchie02.jpg"/>
<item media-type="application/xhtml+xml" id="p-cover" href="xhtml/p-cover.xhtml" properties="svg"/>
<item media-type="application/xhtml+xml" id="p-001" href="xhtml/p-001.xhtml"/>
<item media-type="application/xhtml+xml" id="p-titlepage" href="xhtml/p-titlepage.xhtml"/>
<item media-type="application/xhtml+xml" id="p-toc-001" href="xhtml/p-toc-001.xhtml"/>
<item media-type="application/xhtml+xml" id="p-005" href="xhtml/p-005.xhtml"/>
<item media-type="application/xhtml+xml" id="p-006" href="xhtml/p-006.xhtml"/>
<item media-type="application/xhtml+xml" id="p-014" href="xhtml/p-014.xhtml"/>
<item id="toc.ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
</manifest>
<spine page-progression-direction="rtl">
<itemref linear="yes" idref="p-cover" properties="rendition:layout-pre-paginated rendition:spread-none rendition:page-spread-center"/>
<itemref linear="yes" idref="p-001" properties="rendition:layout-pre-paginated rendition:spread-none"/>
<itemref linear="yes" idref="p-titlepage"/>
<itemref linear="yes" idref="p-toc-001"/>
<itemref linear="yes" idref="p-005"/>
<itemref linear="yes" idref="p-006"/>
<itemref linear="yes" idref="p-014"/>
</spine>
</package>"""


def _build_japanese_nav() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ja">
<head>
<meta charset="UTF-8"/>
<title>Navigation</title>
</head>
<body>
<nav epub:type="toc" id="toc">
<h1>Navigation</h1>
<ol>
<li><a href="xhtml/p-cover.xhtml">\u8868\u7d19</a></li>
<li><a href="xhtml/p-toc-001.xhtml">CONTENTS</a></li>
<li><a href="xhtml/p-005.xhtml">\u30d7\u30ed\u30ed\u30fc\u30b0\u300e\u305d\u306e\u540d\u306f\u2500\u2500\u300f</a></li>
<li><a href="xhtml/p-006.xhtml">\u7b2c\u4e00\u7ae0\u300e\u8150\u6557\u3059\u308b\u7cbe\u795e\u300f</a></li>
<li><a href="xhtml/p-014.xhtml">\u7b2c\u4e09\u7ae0\u300e\u7d76\u671b\u3068\u3044\u3046\u75c5\u300f</a></li>
</ol>
</nav>
<nav epub:type="landmarks" id="guide">
<h1>Guide</h1>
<ol>
<li><a epub:type="cover" href="xhtml/p-cover.xhtml">\u8868\u7d19</a></li>
<li><a epub:type="toc" href="xhtml/p-toc-001.xhtml">\u76ee\u6b21</a></li>
<li><a epub:type="bodymatter" href="xhtml/p-005.xhtml">\u672c\u7de8</a></li>
</ol>
</nav>
</body>
</html>"""


def _build_japanese_ncx() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="4420032460000"/></head>
<docTitle><text>\uff32\uff45\uff1a\u30bc\u30ed\u304b\u3089\u59cb\u3081\u308b\u7570\u4e16\u754c\u751f\u6d3b5</text></docTitle>
<navMap>
<navPoint>
<navLabel><text>\u8868\u7d19</text></navLabel>
<content src="xhtml/p-cover.xhtml"/>
</navPoint>
<navPoint>
<navLabel><text>CONTENTS</text></navLabel>
<content src="xhtml/p-toc-001.xhtml"/>
</navPoint>
<navPoint>
<navLabel><text>\u30d7\u30ed\u30ed\u30fc\u30b0\u300e\u305d\u306e\u540d\u306f\u2500\u2500\u300f</text></navLabel>
<content src="xhtml/p-005.xhtml"/>
</navPoint>
<navPoint>
<navLabel><text>\u7b2c\u4e00\u7ae0\u300e\u8150\u6557\u3059\u308b\u7cbe\u795e\u300f</text></navLabel>
<content src="xhtml/p-006.xhtml"/>
</navPoint>
<navPoint>
<navLabel><text>\u7b2c\u4e09\u7ae0\u300e\u7d76\u671b\u3068\u3044\u3046\u75c5\u300f</text></navLabel>
<content src="xhtml/p-014.xhtml"/>
</navPoint>
</navMap>
</ncx>"""


def _build_japanese_toc_xhtml() -> str:
    """Build the in-book ToC page (p-toc-001.xhtml) for the mini EPUB."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ja" class="vrtl">
<head>
<meta charset="UTF-8"/>
<title>\uff32\uff45\uff1a\u30bc\u30ed\u304b\u3089\u59cb\u3081\u308b\u7570\u4e16\u754c\u751f\u6d3b5</title>
<link rel="stylesheet" type="text/css" href="../style/book-style.css"/>
</head>
<body class="p-toc">
<div class="main start-2em">
<p><span class="mfont font-1em00 bold">CONTENTS</span></p>
<p><br/></p>
<div class="h-indent-5em">
<p><a href="p-005.xhtml#a000">\u30d7\u30ed\u30ed\u30fc\u30b0</a>\u300e\u305d\u306e\u540d\u306f\u2500\u2500\u300f</p>
</div>
<div class="h-indent-3em">
<p><a href="p-006.xhtml#a001">\u7b2c\u4e00\u7ae0</a>\u300e\u8150\u6557\u3059\u308b\u7cbe\u795e\u300f</p>
<p><a href="p-014.xhtml#a003">\u7b2c\u4e09\u7ae0</a>\u300e\u7d76\u671b\u3068\u3044\u3046\u75c5\u300f</p>
</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building mini English EPUB...")
    build_english_mini()
    print()
    print("Building mini Japanese EPUB...")
    build_japanese_mini()
    print()
    print("Done.")
