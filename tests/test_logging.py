"""Tests for dao_bridge.logging — file-handler markup handling."""

from __future__ import annotations

import logging

from dao_bridge.logging import _PlainMarkupFormatter


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="dao_bridge",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestPlainMarkupFormatter:
    """The file formatter undoes Rich-escaping without eating other text."""

    def _fmt(self, msg: str) -> str:
        return _PlainMarkupFormatter("%(message)s").format(_record(msg))

    def test_unescapes_escaped_label(self):
        # _context_prefix emits "\\[summary:<id>] ..." for the file stream.
        assert self._fmt("\\[summary:place_000001] LLM request start") == (
            "[summary:place_000001] LLM request start"
        )

    def test_unescaped_batch_label_untouched(self):
        # Batch IDs are not tag-like, so they are never escaped to begin with.
        assert self._fmt("[0020.b2] LLM request success") == (
            "[0020.b2] LLM request success"
        )

    def test_does_not_eat_markup_like_message_text(self):
        # A blanket render().plain would turn this into 'something  here';
        # the targeted unescape must leave non-escaped brackets intact.
        assert self._fmt("something [not a tag] here") == "something [not a tag] here"

    def test_preserves_json_array_brackets(self):
        raw = 'Raw LLM response:\n[{"name": "x"}, {"y": 1}]'
        assert self._fmt(raw) == raw

    def test_leaves_styled_tags_raw(self):
        # state.py emits [bold] intentionally; file output keeps it as today.
        assert self._fmt("Stage [bold]glossary_build[/bold] started") == (
            "Stage [bold]glossary_build[/bold] started"
        )
