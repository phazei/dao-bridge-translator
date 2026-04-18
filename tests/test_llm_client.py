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


class TestRetryCounterReset:
    @patch("dao_bridge.llm_client.openai.OpenAI")
    def test_retry_counter_resets_on_successful_parse(self, mock_openai_cls):
        """Scenario: malformed, malformed, good, malformed, good -> 5 attempts, no error.

        The counter resets after a successful JSON parse, so:
        - Attempt 1: malformed JSON -> consecutive_failures = 1
        - Attempt 2: malformed JSON -> consecutive_failures = 2
        - Attempt 3: valid JSON, valid schema -> consecutive_failures = 0, RETURN

        But we want to test the full scenario where parse succeeds but
        validation fails, then it recovers.  Let's use the exact spec:

        - Attempt 1: malformed JSON -> consecutive_failures = 1
        - Attempt 2: malformed JSON -> consecutive_failures = 2  (max_retries=3, not yet hit)
        - Attempt 3: valid JSON, valid schema -> SUCCESS, return

        Actually the spec says: "malformed, malformed, good, malformed, good -> 5 attempts"
        The key insight: a successful JSON *parse* resets the counter, even if
        validation then fails.  So we need:

        - Attempt 1: malformed (not JSON) -> consecutive_failures = 1
        - Attempt 2: malformed (not JSON) -> consecutive_failures = 2
        - Attempt 3: valid JSON + valid model -> return
          BUT we want 5 attempts.  So the "good" must be parse-good but
          validation-bad:

        Reinterpretation:
        - Attempt 1: not valid JSON at all -> consecutive_failures = 1
        - Attempt 2: not valid JSON at all -> consecutive_failures = 2
        - Attempt 3: valid JSON, passes parse, resets counter to 0.
                     Then validation fails -> consecutive_failures = 1
        - Attempt 4: not valid JSON -> consecutive_failures = 2
        - Attempt 5: valid JSON, valid model -> SUCCESS

        With max_retries=3, we never hit the limit because the counter
        never reaches 3 consecutively.  5 API calls total.
        """
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [
            # Attempt 1: malformed JSON
            _mock_response("this is not json"),
            # Attempt 2: malformed JSON
            _mock_response("{broken json"),
            # Attempt 3: valid JSON, but wrong schema (value is string not int)
            #   -> parse succeeds (resets counter to 0), validation fails (counter -> 1)
            _mock_response('{"name": "test", "value": "not_int"}'),
            # Attempt 4: malformed JSON again -> counter -> 2
            _mock_response("still broken"),
            # Attempt 5: valid JSON, valid schema -> SUCCESS
            _mock_response('{"name": "final", "value": 42}'),
        ]

        config = ModelConfig(model="test-model")
        client = LLMClient(config)

        # max_retries=3 means we tolerate up to 3 consecutive failures.
        # The counter resets mid-sequence so we never hit 3.
        result = client.complete_json(
            [{"role": "user", "content": "give me data"}],
            response_model=SimpleResponse,
            max_retries=3,
        )

        assert isinstance(result, SimpleResponse)
        assert result.name == "final"
        assert result.value == 42
        # Exactly 5 API calls were made.
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
