"""LLM client wrapping the OpenAI SDK for multi-turn completions.

Supports any OpenAI-compatible API (llama-server, vLLM, LM Studio, Claude,
OpenAI, OpenRouter, etc.) by pointing ``base_url`` at the target.

The client operates on **messages** (not a single prompt string) so that later
pipeline stages can cleanly separate previous-context from target-text in
multi-turn conversations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import openai
from pydantic import BaseModel, ValidationError
from rich.markup import escape as _rich_escape

from dao_bridge.config import LLMConfig, ModelConfig

logger = logging.getLogger("dao_bridge")


# ---------------------------------------------------------------------------
# Result / exception types
# ---------------------------------------------------------------------------


@dataclass
class CompletionResult:
    """Wrapper around a chat-completion response."""

    text: str
    token_usage: dict = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMStructuredOutputError(Exception):
    """Raised when ``complete_json`` cannot obtain valid structured output
    after exhausting retries."""


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------

_TRANSIENT_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.BadRequestError,
)


def _error_code(exc: Exception) -> str | None:
    """Best-effort provider error code extraction."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            nested_code = error.get("code")
            if isinstance(nested_code, str) and nested_code:
                return nested_code
    return None


def _context_prefix(context_label: str | None) -> str:
    """Build a ``"[label] "`` log prefix that survives Rich markup rendering.

    The console handler has ``markup=True``, so a bare ``[summary:<id>]`` would
    be parsed as a style tag and silently dropped.  Escaping the whole bracket
    token makes Rich render it literally on the console; the file handler's
    formatter strips the escape so ``run.log`` stays clean (see
    :mod:`dao_bridge.logging`).
    """
    if not context_label:
        return ""
    return _rich_escape(f"[{context_label}]") + " "


def _should_retry(exc: Exception) -> bool:
    """Return whether this provider error is worth retrying."""
    if isinstance(exc, openai.BadRequestError):
        return False

    if isinstance(exc, openai.RateLimitError):
        return _error_code(exc) != "insufficient_quota"

    return True


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """High-level client for LLM chat completions.

    Parameters
    ----------
    config:
        Per-task model configuration (``base_url``, ``api_key``, ``model``,
        ``temperature``).
    llm_config:
        Global retry / timeout settings.
    """

    def __init__(self, config: ModelConfig, llm_config: LLMConfig | None = None) -> None:
        self.config = config
        self.llm_config = llm_config or LLMConfig()
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=self.llm_config.request_timeout_seconds,
            max_retries=0,
        )
        self._total_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Cumulative token usage
    # ------------------------------------------------------------------

    @property
    def total_token_usage(self) -> dict[str, int]:
        """Return a copy of the cumulative token usage across all calls."""
        return dict(self._total_token_usage)

    def reset_token_usage(self) -> None:
        """Zero the cumulative token usage counter."""
        self._total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # complete
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        context_label: str | None = None,
    ) -> CompletionResult:
        """Multi-turn chat completion with automatic retries on transient errors.

        Parameters
        ----------
        messages:
            OpenAI chat format:
            ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
        max_tokens:
            Optional maximum completion tokens.
        context_label:
            Optional caller-provided label (e.g. batch ID, ``summary:<id>``)
            prefixed onto the request start/success/retry log lines so the
            console and run.log show what each call is for.

        Returns
        -------
        CompletionResult
        """
        ctx = _context_prefix(context_label)
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
        }
        effective_temperature = self.config.temperature if temperature is None else temperature
        if effective_temperature is not None:
            kwargs["temperature"] = effective_temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(1, self.llm_config.max_retries + 1):
            attempt_started = time.monotonic()
            logger.info(
                "%sLLM request start (%d/%d): model=%s timeout=%.1fs messages=%d",
                ctx,
                attempt,
                self.llm_config.max_retries,
                self.config.model,
                self.llm_config.request_timeout_seconds,
                len(messages),
            )
            try:
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - attempt_started
                choice = response.choices[0]
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        self._total_token_usage[key] += usage.get(key, 0)
                logger.info(
                    "%sLLM request success (%d/%d): model=%s elapsed=%.2fs finish=%s",
                    ctx,
                    attempt,
                    self.llm_config.max_retries,
                    response.model or self.config.model,
                    elapsed,
                    choice.finish_reason or "",
                )
                return CompletionResult(
                    text=choice.message.content or "",
                    token_usage=usage,
                    model=response.model or self.config.model,
                    finish_reason=choice.finish_reason or "",
                )
            except _TRANSIENT_EXCEPTIONS as exc:
                elapsed = time.monotonic() - attempt_started
                if not _should_retry(exc):
                    logger.error(
                        "%sLLM request failed without retry (%d/%d): "
                        "model=%s elapsed=%.2fs error=%s",
                        ctx,
                        attempt,
                        self.llm_config.max_retries,
                        self.config.model,
                        elapsed,
                        exc,
                    )
                    raise
                last_error = exc
                wait = self.llm_config.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%sTransient LLM error (attempt %d/%d): "
                    "model=%s elapsed=%.2fs error=%s — retrying in %.1fs",
                    ctx,
                    attempt,
                    self.llm_config.max_retries,
                    self.config.model,
                    elapsed,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # complete_json
    # ------------------------------------------------------------------

    def complete_json(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        max_retries: int = 3,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context_label: str | None = None,
    ) -> BaseModel:
        """Chat completion that returns a validated Pydantic model.

        Injects a schema instruction into the last user message, parses the
        JSON response, and validates it.  On parse / validation failure the
        error is appended to the conversation and the call is retried.

        Both parse failures and validation failures increment the
        consecutive failure counter.  A hard ceiling on total attempts
        (``max_retries * 2``) prevents infinite loops when the model
        consistently returns parseable-but-invalid JSON.

        Parameters
        ----------
        messages:
            OpenAI chat format messages.
        response_model:
            Pydantic model class to validate against.
        max_retries:
            Maximum *consecutive* parse/validation failures before raising.
        max_tokens:
            Optional maximum completion tokens.
        context_label:
            Optional caller-provided label (e.g. chunk ID, batch ID)
            included in log messages for easier debugging.

        Raises
        ------
        LLMStructuredOutputError
            After *max_retries* consecutive failures.
        """
        ctx = _context_prefix(context_label)
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = (
            "\n\nRespond with JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Return ONLY valid JSON, no other text."
        )

        # Deep-copy messages so we can mutate safely.
        conversation: list[dict] = [dict(m) for m in messages]

        # Inject schema instruction into the last user message.
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]["role"] == "user":
                conversation[i] = dict(conversation[i])
                conversation[i]["content"] = conversation[i]["content"] + schema_instruction
                break

        consecutive_failures = 0
        total_attempts = 0
        max_total_attempts = max_retries * 2

        while consecutive_failures < max_retries and total_attempts < max_total_attempts:
            total_attempts += 1
            result = self.complete(
                conversation,
                max_tokens=max_tokens,
                temperature=temperature,
                context_label=context_label,
            )
            raw_text = result.text.strip()

            # Strip markdown code fences if present.
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                # Remove first line (```json or ```) and last line (```)
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                raw_text = "\n".join(lines).strip()

            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                consecutive_failures += 1
                error_msg = f"JSON parse error: {exc}"
                logger.warning(
                    "%scomplete_json parse failure (%d/%d): %s",
                    ctx,
                    consecutive_failures,
                    max_retries,
                    error_msg,
                )
                logger.debug(
                    "%sRaw LLM response:\n%s",
                    ctx,
                    result.text,
                )
                conversation.append({"role": "assistant", "content": result.text})
                conversation.append(
                    {
                        "role": "user",
                        "content": f"Your response was not valid JSON. Error: {error_msg}\n"
                        "Please try again with valid JSON only.",
                    }
                )
                continue

            try:
                return response_model(**parsed)
            except (ValidationError, TypeError) as exc:
                consecutive_failures += 1
                error_msg = f"Validation error: {exc}"
                logger.warning(
                    "%scomplete_json validation failure (%d/%d): %s",
                    ctx,
                    consecutive_failures,
                    max_retries,
                    error_msg,
                )
                logger.debug(
                    "%sRaw LLM response (valid JSON, failed validation):\n%s",
                    ctx,
                    raw_text,
                )
                conversation.append({"role": "assistant", "content": result.text})
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Your JSON was parseable but failed validation. "
                            f"Error: {error_msg}\n"
                            "Please fix the issues and try again with valid JSON."
                        ),
                    }
                )
                continue

        raise LLMStructuredOutputError(
            f"{ctx}Failed to get valid structured output after {total_attempts} attempts "
            f"({consecutive_failures} consecutive failures, limit {max_retries}). "
            f"Model: {self.config.model}"
        )
