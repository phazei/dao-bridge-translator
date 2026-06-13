from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe an OpenAI-compatible chat completions endpoint."
    )
    parser.add_argument("--base-url", required=True, help="Base URL ending in /v1")
    parser.add_argument("--api-key", help="API key. Falls back to OPENAI_API_KEY")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: endpoint works",
        help="User prompt",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 text file for the user prompt. Overrides --prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the full JSON response",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming chat completions and report timing.",
    )
    return parser


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def main() -> int:
    args = build_parser().parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing API key. Use --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2

    prompt = load_prompt(args)
    url = args.base_url.rstrip("/") + "/chat/completions"

    body: dict[str, object] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
    }
    if args.stream:
        body["stream"] = True
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "dao-bridge-probe/1.0",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if args.stream:
                return handle_streaming_response(response, started)
            elapsed = time.perf_counter() - started
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print(f"Request failed after {elapsed:.2f}s: {exc}", file=sys.stderr)
        return 1

    payload = json.loads(raw)
    choice = payload["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    if args.show_raw:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "content_length": len(content),
        "reasoning_length": len(reasoning),
        "content_preview": content[:800],
        "reasoning_preview": reasoning[:800],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def handle_streaming_response(response, started: float) -> int:
    decoder = "utf-8"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    model = ""
    finish_reason = ""
    first_chunk_seconds: float | None = None

    for raw_line in response:
        line = raw_line.decode(decoder, errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        payload = json.loads(data)
        if not model:
            model = payload.get("model", "")
        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if (content or reasoning) and first_chunk_seconds is None:
            first_chunk_seconds = round(time.perf_counter() - started, 2)
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    elapsed = round(time.perf_counter() - started, 2)
    content_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    summary = {
        "stream": True,
        "elapsed_seconds": elapsed,
        "first_chunk_seconds": first_chunk_seconds,
        "model": model,
        "finish_reason": finish_reason,
        "content_length": len(content_text),
        "reasoning_length": len(reasoning_text),
        "content_preview": content_text[:800],
        "reasoning_preview": reasoning_text[:800],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
