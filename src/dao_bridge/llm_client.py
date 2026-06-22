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
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union, get_args, get_origin

import openai
from pydantic import BaseModel, ValidationError
from rich.markup import escape as _rich_escape

from dao_bridge.config import LLMConfig, ModelConfig

logger = logging.getLogger("dao_bridge")


def _last_balanced_json_object(text: str) -> str | None:
    """Return the last top-level balanced ``{...}`` object in *text*, or None.

    Scans from the end for a ``}``, then walks backwards to its matching ``{``
    while respecting string literals and escapes.  Used as a fallback when a
    model wraps its JSON in prose / an unclosed reasoning preamble.
    """
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    in_string = False
    i = end
    while i >= 0:
        ch = text[i]
        if in_string:
            # We are scanning backwards; count preceding backslashes to know if
            # this quote is escaped.
            if ch == '"':
                back = i - 1
                slashes = 0
                while back >= 0 and text[back] == "\\":
                    slashes += 1
                    back -= 1
                if slashes % 2 == 0:
                    in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    return text[i : end + 1]
        i -= 1
    return None


def _extract_structured_text(raw_text: str) -> str:
    """Best-effort extraction of the JSON object from a model response.

    Order of operations:
    1. Strip surrounding markdown code fences.
    2. If the result still does not start with ``{``, fall back to extracting
       the last balanced ``{...}`` object found anywhere in the text (handles a
       model wrapping its JSON in prose like "Final decision: ...").
    """
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    if not text.startswith("{"):
        candidate = _last_balanced_json_object(text)
        if candidate is not None:
            return candidate
    return text


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
    # Server-side reasoning/"thinking" text, when the model/endpoint returns it
    # in a separate ``reasoning_content`` field (e.g. LM Studio). Empty string
    # when thinking is disabled or the field is absent. The pipeline does not
    # use this for output; it is exposed for diagnostics (e.g. measuring how
    # often reasoning runs away).
    reasoning_text: str = ""


class LLMStructuredOutputError(Exception):
    """Raised when ``complete_json`` cannot obtain valid structured output
    after exhausting retries."""


class LLMValidationError(Exception):
    """Raised by a ``complete_validated`` validator to reject a response, and
    re-raised by ``complete_validated`` itself once retries are exhausted."""


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


def _example_value(annotation: object) -> object:
    """A placeholder example value for a single field annotation.

    Used to build an example *instance* of a response model (see
    :func:`_example_instruction`).  Recurses into ``list``/``set``/``tuple``
    element types and into nested :class:`~pydantic.BaseModel` subclasses so
    array- and object-valued fields produce a realistic shape.
    """
    # Unwrap Optional[...] / unions (both typing.Union and PEP 604 ``X | Y``):
    # use the first non-None member.
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _example_value(args[0]) if args else "..."

    if origin in (list, set, tuple):
        inner_args = get_args(annotation)
        inner = inner_args[0] if inner_args else str
        return [_example_value(inner)]

    if origin in (dict,):
        return {"...": "..."}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _example_instance(annotation)

    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return True

    # str, Literal, enums, datetime, and anything unrecognized -> string hint.
    return "..."


def _example_instance(model: type[BaseModel]) -> dict:
    """Build a JSON-serializable example instance of *model* from its fields."""
    return {name: _example_value(field.annotation) for name, field in model.model_fields.items()}


def _example_instruction(model: type[BaseModel]) -> str:
    """Build the output-shape instruction injected into the prompt.

    Injects a concrete example *instance* (e.g. ``{"summary": "..."}``) rather
    than the raw ``model_json_schema()`` envelope.  The schema envelope
    (``{"properties": ..., "required": ...}``) is itself valid JSON, and some
    local models echo it back verbatim — valid JSON that is missing every
    actual field, so it fails validation and forces a retry.  An example
    instance has no schema envelope to parrot: the only JSON shape shown is the
    answer shape.  The example is generated from the Pydantic model, so it
    stays in sync with the schema automatically.
    """
    example = json.dumps(_example_instance(model), ensure_ascii=False, indent=2)
    return (
        "\n\nReturn ONLY a JSON object of exactly this shape, replacing the "
        f"placeholder values with real content:\n```json\n{example}\n```\n"
        "Return ONLY valid JSON, no other text. Do not return the schema; "
        "return a filled-in object."
    )


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
        # Forward the thinking/reasoning control when configured. Sent via
        # extra_body so non-standard values (e.g. LM Studio's "none", which is
        # outside the OpenAI SDK's typed enum) pass through unvalidated.
        if self.config.reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": self.config.reasoning_effort,
            }
        # Keep-alive: forward ttl so the server (LM Studio) defers auto-unload,
        # keeping the model resident between calls (e.g. when alternating with a
        # separate QA model). Sent via extra_body as it is non-standard.
        if self.config.ttl is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "ttl": self.config.ttl,
            }

        last_error: Exception | None = None
        for attempt in range(1, self.llm_config.max_retries + 1):
            attempt_started = time.monotonic()
            logger.debug(
                "%sLLM request start\u00a0\u00a0 (%d/%d): model=%s timeout=%.1fs messages=%d",
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
                retry_note = "" if attempt == 1 else f" (retry {attempt}/{self.llm_config.max_retries})"
                logger.info(
                    "%sLLM model=%s%s elapsed=%.2fs finish=%s",
                    ctx,
                    self.config.model,
                    retry_note,
                    elapsed,
                    choice.finish_reason or "",
                )
                # ``reasoning_content`` is non-standard; access defensively.
                reasoning_text = getattr(choice.message, "reasoning_content", None) or ""
                return CompletionResult(
                    text=choice.message.content or "",
                    token_usage=usage,
                    model=response.model or self.config.model,
                    finish_reason=choice.finish_reason or "",
                    reasoning_text=reasoning_text,
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
    # complete_validated
    # ------------------------------------------------------------------

    def complete_validated(
        self,
        messages: list[dict],
        validate: Callable[[str], None],
        max_attempts: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context_label: str | None = None,
    ) -> CompletionResult:
        """Chat completion whose text must satisfy a caller-supplied contract.

        Calls :meth:`complete` (which still handles transient API errors with
        its own backoff) and runs *validate* on the returned text.  When
        *validate* raises :class:`LLMValidationError`, the response is rejected
        and the **same messages** are resubmitted — temperature variance, not
        conversation feedback, drives the model toward a compliant sample.

        Unlike :meth:`complete_json`, no corrective message is injected into the
        conversation: the validated contract here (e.g. a required boundary
        tag) is structural, and re-explaining freeform reasoning output is not
        meaningful.  Only the bounded-retry discipline is borrowed.

        Parameters
        ----------
        messages:
            OpenAI chat format messages.
        validate:
            Callable invoked with the response text.  Must raise
            :class:`LLMValidationError` to reject the response; return value is
            ignored.
        max_attempts:
            Maximum number of completions to attempt.  Defaults to the client's
            configured ``max_retries``.
        max_tokens, temperature, context_label:
            Forwarded to :meth:`complete`.

        Returns
        -------
        CompletionResult
            The first response that passes *validate*.

        Raises
        ------
        LLMValidationError
            After *max_attempts* responses all fail validation.
        """
        ctx = _context_prefix(context_label)
        attempts = max_attempts if max_attempts is not None else self.llm_config.max_retries
        attempts = max(1, attempts)

        last_error: LLMValidationError | None = None
        for attempt in range(1, attempts + 1):
            result = self.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                context_label=context_label,
            )
            try:
                validate(result.text)
                return result
            except LLMValidationError as exc:
                last_error = exc
                logger.warning(
                    "%scomplete_validated rejected response (%d/%d): %s — resubmitting",
                    ctx,
                    attempt,
                    attempts,
                    exc,
                )
                logger.debug(
                    "%sRaw LLM response (failed validation):\n%s",
                    ctx,
                    result.text,
                )

        raise LLMValidationError(
            f"{ctx}Response failed validation after {attempts} attempt(s): {last_error}"
        )

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

        Injects an output-shape instruction into the last user message, parses
        the JSON response, and validates it.  On parse / validation failure the
        error is appended to the conversation and the call is retried.

        The injected instruction is a concrete **example instance** of
        *response_model* (e.g. ``{"summary": "..."}``), generated from the
        Pydantic model so it stays in sync with the schema.  This deliberately
        avoids injecting the raw ``model_json_schema()`` envelope: that envelope
        is itself valid JSON, and some local models echo it back verbatim —
        valid JSON missing every actual field, which fails validation and
        wastes a round-trip.  See :func:`_example_instruction`.

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
        shape_instruction = _example_instruction(response_model)

        # Deep-copy messages so we can mutate safely.
        conversation: list[dict] = [dict(m) for m in messages]

        # Inject the output-shape instruction into the last user message.
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]["role"] == "user":
                conversation[i] = dict(conversation[i])
                conversation[i]["content"] = conversation[i]["content"] + shape_instruction
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
            # Extract the JSON object: strip code fences, falling back to the
            # last balanced {...} object if the model wrapped its JSON in prose.
            raw_text = _extract_structured_text(result.text)

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
