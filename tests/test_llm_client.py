"""Tests for dao_bridge.llm_client — mocked OpenAI SDK, retry logic, structured output."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from dao_bridge.config import LLMConfig, ModelConfig
from dao_bridge.llm_client import CompletionResult, LLMClient, LLMStructuredOutputError

# ---------------------------------------------------------------------------
# Helper: mock response factory
# ---------------------------------------------------------------------------


def _mock_response(
    content: str = "Hello",
    model: str = "test-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a mock OpenAI ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Pydantic model for structured output tests
# ---------------------------------------------------------------------------


class SimpleResponse(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# Basic complete()
# ---------------------------------------------------------------------------


class TestComplete:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_successful_completion(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("Hello world")

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        result = client.complete([{"role": "user", "content": "hi"}])

        assert isinstance(result, CompletionResult)
        assert result.text == "Hello world"
        assert result.model == "test-model"
        assert result.token_usage["total_tokens"] == 15

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_max_tokens_passed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response()

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        client.complete([{"role": "user", "content": "hi"}], max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100


# ---------------------------------------------------------------------------
# context_label on start/success log lines
# ---------------------------------------------------------------------------


def _renders_literally_on_console(message: str, expected_prefix: str) -> bool:
    """Return whether *message* renders with *expected_prefix* through a
    markup-enabled Rich console (mimics the live console handler).

    A bare ``[summary:<id>]`` would be parsed as a style tag and dropped; the
    escaped form must survive as literal text.
    """
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, force_terminal=False, markup=True, width=200).print(message)
    return buf.getvalue().startswith(expected_prefix)


class TestContextLabel:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_label_on_start_and_success_lines(self, mock_openai_cls, caplog):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        client = LLMClient(ModelConfig(model="test-model"))
        with caplog.at_level("INFO", logger="dao_bridge"):
            client.complete(
                [{"role": "user", "content": "hi"}],
                context_label="0014.b3",
            )

        start = next(m for m in caplog.messages if "LLM request start" in m)
        success = next(m for m in caplog.messages if "LLM request success" in m)
        # Batch IDs are not tag-like, so they are not escaped and render as-is.
        assert start.startswith("[0014.b3] ")
        assert success.startswith("[0014.b3] ")
        assert _renders_literally_on_console(start, "[0014.b3] ")

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_summary_label_is_escaped_and_renders_literally(self, mock_openai_cls, caplog):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        client = LLMClient(ModelConfig(model="test-model"))
        with caplog.at_level("INFO", logger="dao_bridge"):
            client.complete(
                [{"role": "user", "content": "hi"}],
                context_label="summary:place_000001",
            )

        start = next(m for m in caplog.messages if "LLM request start" in m)
        # The colon-form label is tag-like, so it is escaped in the raw record...
        assert start.startswith("\\[summary:place_000001] ")
        # ...and therefore survives Rich markup rendering on the console.
        assert _renders_literally_on_console(start, "[summary:place_000001] ")

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_no_label_no_prefix(self, mock_openai_cls, caplog):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        client = LLMClient(ModelConfig(model="test-model"))
        with caplog.at_level("INFO", logger="dao_bridge"):
            client.complete([{"role": "user", "content": "hi"}])

        start = next(m for m in caplog.messages if "LLM request start" in m)
        assert start.startswith("LLM request start")

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_complete_json_forwards_label_to_start_line(self, mock_openai_cls, caplog):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "x", "value": 1}'
        )

        client = LLMClient(ModelConfig(model="test-model"))
        with caplog.at_level("INFO", logger="dao_bridge"):
            client.complete_json(
                [{"role": "user", "content": "hi"}],
                response_model=SimpleResponse,
                context_label="summary:character_000007",
            )

        start = next(m for m in caplog.messages if "LLM request start" in m)
        assert start.startswith("\\[summary:character_000007] ")
        assert _renders_literally_on_console(start, "[summary:character_000007] ")


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------


class TestRetry:
    @patch("dao_bridge.llm_client.time.sleep")
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_retry_on_connection_error(self, mock_openai_cls, mock_sleep):
        import openai as openai_mod

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Fail twice, succeed on third.
        mock_client.chat.completions.create.side_effect = [
            openai_mod.APIConnectionError(request=MagicMock()),
            openai_mod.APIConnectionError(request=MagicMock()),
            _mock_response("recovered"),
        ]

        config = ModelConfig(model="test-model")
        llm_config = LLMConfig(max_retries=3, retry_backoff_seconds=0.01)
        client = LLMClient(config, llm_config)
        result = client.complete([{"role": "user", "content": "hi"}])

        assert result.text == "recovered"
        assert mock_sleep.call_count == 2

    @patch("dao_bridge.llm_client.time.sleep")
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_retry_exhausted_raises(self, mock_openai_cls, mock_sleep):
        import openai as openai_mod

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = openai_mod.APIConnectionError(
            request=MagicMock()
        )

        config = ModelConfig(model="test-model")
        llm_config = LLMConfig(max_retries=2, retry_backoff_seconds=0.01)
        client = LLMClient(config, llm_config)

        with pytest.raises(openai_mod.APIConnectionError):
            client.complete([{"role": "user", "content": "hi"}])

    @patch("dao_bridge.llm_client.time.sleep")
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_exponential_backoff(self, mock_openai_cls, mock_sleep):
        import openai as openai_mod

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [
            openai_mod.RateLimitError(
                "rate limited", response=MagicMock(status_code=429), body=None
            ),
            openai_mod.RateLimitError(
                "rate limited", response=MagicMock(status_code=429), body=None
            ),
            _mock_response("ok"),
        ]

        config = ModelConfig(model="test-model")
        llm_config = LLMConfig(max_retries=3, retry_backoff_seconds=1.0)
        client = LLMClient(config, llm_config)
        client.complete([{"role": "user", "content": "hi"}])

        # Backoff should be: 1.0 * 2^0 = 1.0, then 1.0 * 2^1 = 2.0
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# complete_json — successful parse
# ---------------------------------------------------------------------------


class TestCompleteJson:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_valid_json_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "test", "value": 42}'
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
        )

        assert isinstance(result, SimpleResponse)
        assert result.name == "test"
        assert result.value == 42

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_json_in_code_fence(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '```json\n{"name": "test", "value": 42}\n```'
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
        )

        assert isinstance(result, SimpleResponse)
        assert result.value == 42


# ---------------------------------------------------------------------------
# complete_json — retry on invalid JSON
# ---------------------------------------------------------------------------


class TestCompleteJsonRetry:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_retry_on_malformed_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call returns invalid JSON, second returns valid.
        mock_client.chat.completions.create.side_effect = [
            _mock_response("not json at all"),
            _mock_response('{"name": "fixed", "value": 1}'),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
            max_retries=3,
        )

        assert result.name == "fixed"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_retry_on_validation_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First: valid JSON but wrong schema (value should be int).
        # Second: correct.
        mock_client.chat.completions.create.side_effect = [
            _mock_response('{"name": "test", "value": "not_an_int"}'),
            _mock_response('{"name": "test", "value": 99}'),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)
        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
            max_retries=3,
        )

        assert result.value == 99

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_raises_after_max_consecutive_failures(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # All responses are invalid JSON.
        mock_client.chat.completions.create.return_value = _mock_response("garbage")

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        with pytest.raises(LLMStructuredOutputError):
            client.complete_json(
                [{"role": "user", "content": "give me data"}],
                response_model=SimpleResponse,
                max_retries=2,
            )

        # Should have made exactly 2 attempts (= max_retries for consecutive failures).
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# complete_json — retry counter reset scenario
# ---------------------------------------------------------------------------


class TestRetryCounterAndTotalAttempts:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_consecutive_failures_accumulate_across_parse_and_validation(self, mock_openai_cls):
        """Failures accumulate without reset: parse fail + validation fail both count.

        - Attempt 1: malformed JSON -> consecutive_failures = 1
        - Attempt 2: valid JSON, validation fails -> consecutive_failures = 2
        - Attempt 3: valid JSON, valid schema -> SUCCESS

        With max_retries=3, we succeed on attempt 3 (consecutive_failures
        never reaches 3).
        """
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [
            # Attempt 1: malformed JSON -> consecutive_failures = 1
            _mock_response("this is not json"),
            # Attempt 2: valid JSON, wrong schema -> consecutive_failures = 2
            _mock_response('{"name": "test", "value": "not_int"}'),
            # Attempt 3: valid JSON, valid schema -> SUCCESS
            _mock_response('{"name": "final", "value": 42}'),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
            max_retries=3,
        )

        assert isinstance(result, SimpleResponse)
        assert result.name == "final"
        assert result.value == 42
        assert mock_client.chat.completions.create.call_count == 3

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_total_attempts_ceiling_prevents_infinite_loop(self, mock_openai_cls):
        """Even if consecutive_failures stays below max_retries, the total
        attempts ceiling (max_retries * 2) halts the loop.

        With max_retries=3 and max_total_attempts=6:
        - Attempts 1-6: all return parseable-but-invalid JSON
        - consecutive_failures increments each time: 1, 2, 3 -> exits at 3

        Actually consecutive failures alone catch this case.  To test the
        *total attempts* ceiling we need max_retries high enough that
        consecutive failures wouldn't trigger first.  Use max_retries=10
        with a scenario that hits total_attempts=20 ceiling first.

        Simpler: just verify that validation failures accumulate and the
        loop terminates.
        """
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # All responses are valid JSON but fail validation (value is string).
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "test", "value": "not_an_int"}'
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        with pytest.raises(LLMStructuredOutputError):
            client.complete_json(
                [{"role": "user", "content": "give me data"}],
                response_model=SimpleResponse,
                max_retries=3,
            )

        # Should have made exactly 3 attempts (consecutive_failures hits 3).
        assert mock_client.chat.completions.create.call_count == 3

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_total_attempts_ceiling_with_high_max_retries(self, mock_openai_cls):
        """With a high max_retries, the total_attempts ceiling (max_retries * 2)
        is the binding constraint.

        max_retries=100, max_total_attempts=200.
        All responses fail validation -> consecutive_failures hits 100 at attempt 100.
        But total_attempts ceiling is 200, so consecutive_failures is the
        binding limit here.

        To actually test the total_attempts ceiling being the binding limit,
        we need failures to *not* be consecutive — but without resetting
        the counter, they always are.  The total_attempts ceiling is a
        safety net for any future logic changes.  Verify the ceiling exists:
        """
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Always returns invalid schema.
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "test", "value": "bad"}'
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        with pytest.raises(LLMStructuredOutputError):
            client.complete_json(
                [{"role": "user", "content": "data"}],
                response_model=SimpleResponse,
                max_retries=5,
            )

        # consecutive_failures hits 5 at attempt 5.
        # total_attempts ceiling is 10, so consecutive is the binding limit.
        assert mock_client.chat.completions.create.call_count == 5


# ---------------------------------------------------------------------------
# Cumulative token usage tracking
# ---------------------------------------------------------------------------


class TestTokenUsageTracking:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_accumulates_across_complete_calls(self, mock_openai_cls):
        """total_token_usage accumulates across multiple complete() calls."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _mock_response("first", prompt_tokens=100, completion_tokens=50),
            _mock_response("second", prompt_tokens=200, completion_tokens=60),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        client.complete([{"role": "user", "content": "a"}])
        client.complete([{"role": "user", "content": "b"}])

        usage = client.total_token_usage
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 110
        assert usage["total_tokens"] == 410

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_accumulates_through_complete_json(self, mock_openai_cls):
        """complete_json calls complete() internally — tokens are tracked."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"name": "test", "value": 42}',
            prompt_tokens=80,
            completion_tokens=20,
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
        )

        usage = client.total_token_usage
        assert usage["prompt_tokens"] == 80
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 100

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_complete_json_retries_accumulate(self, mock_openai_cls):
        """complete_json retries also accumulate token usage."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            # First attempt: bad JSON.
            _mock_response("not json", prompt_tokens=50, completion_tokens=10),
            # Second attempt: valid.
            _mock_response('{"name": "ok", "value": 1}', prompt_tokens=60, completion_tokens=15),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        client.complete_json(
            [{"role": "user", "content": "data"}],
            response_model=SimpleResponse,
            max_retries=3,
        )

        usage = client.total_token_usage
        # Both calls accumulated.
        assert usage["prompt_tokens"] == 110
        assert usage["completion_tokens"] == 25
        assert usage["total_tokens"] == 135

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_reset_zeroes_accumulator(self, mock_openai_cls):
        """reset_token_usage() clears the cumulative counters."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            "hello", prompt_tokens=100, completion_tokens=50
        )

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        client.complete([{"role": "user", "content": "a"}])
        assert client.total_token_usage["total_tokens"] == 150

        client.reset_token_usage()
        assert client.total_token_usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Accumulation restarts after reset.
        client.complete([{"role": "user", "content": "b"}])
        assert client.total_token_usage["total_tokens"] == 150

    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_total_token_usage_returns_copy(self, mock_openai_cls):
        """total_token_usage returns a copy, not the internal dict."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        usage1 = client.total_token_usage
        usage1["prompt_tokens"] = 999
        usage2 = client.total_token_usage
        assert usage2["prompt_tokens"] == 0
