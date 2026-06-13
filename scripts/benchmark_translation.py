"""Translation benchmark across local LM Studio models.

Runs a fixed, deliberately-difficult Chinese paragraph through several models,
N times each, via the OpenAI-compatible chat-completions endpoint.

Design notes:
- No temperature is sent. We let each model use its own default.
- We DO NOT inject any thinking toggle. Instead we DETECT reasoning output
  (the `reasoning_content` field and/or <think>...</think> spans in content)
  and report whether the model appears to be "thinking". This lets the user
  confirm whether thinking was actually disabled in the model's LM Studio /
  Jinja template config.
- Models stay loaded between runs, so runs 2 and 3 should be faster (we report
  per-run timing and tokens/sec so you can see the warm-up effect).

Usage:
    python scripts/benchmark_translation.py
    python scripts/benchmark_translation.py --runs 3 --base-url http://127.0.0.1:1234/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# The benchmark source text. Difficult across multiple axes:
# - chengyu / idioms: 吃亏是福, 门儿清, 君子爱财取之有道
# - culturally-specific simile: the abacus metaphor
# - pun / wordplay: '盘'活资产 plays on 盘活 + 算盘
# - modern internet slang: 又当又立 + literal 贞节牌坊
# - pervasive sarcasm / irony
SOURCE_TEXT = (
    "老李这人，嘴上说着“吃亏是福”，心里却比谁都门儿清。"
    "上礼拜公司裁员，他表面上替同事抱不平，背地里早把功劳簿擦得锃亮——"
    "真是又当又立，还想立块贞节牌坊。"
    "同事老王私下吐槽：“他呀，属算盘的，不拨不动，一拨叮当响。”"
    "这话传到老李耳朵里，他也不恼，只笑眯眯地回了句："
    "“君子爱财，取之有道嘛，我这叫‘盘’活资产。”"
)

SYSTEM_PROMPT = (
    "You are a professional literary translator. Translate the user's Chinese "
    "text into natural, fluent English. Preserve tone, sarcasm, idioms, "
    "metaphors, and wordplay; render them so an English reader feels the same "
    "effect. Output only the English translation, with no preamble or notes."
)

MODELS = [
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "qwen3.6-35b-a3b-mtp",
    "qwen3.6-27b",
]

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--api-key", default="lm-studio")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--models",
        nargs="*",
        default=MODELS,
        help="Override the model list.",
    )
    p.add_argument(
        "--out",
        default="scripts/benchmark_results.json",
        help="Where to write the raw JSON results.",
    )
    return p


def call_model(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SOURCE_TEXT},
        ],
        # Intentionally NO temperature. Let the model decide.
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    elapsed = time.perf_counter() - started

    payload = json.loads(raw)
    choice = payload["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    # Detect any <think>...</think> embedded in content.
    think_spans = THINK_TAG_RE.findall(content)
    visible = THINK_TAG_RE.sub("", content).strip()
    embedded_think = "".join(think_spans).strip()

    # The model is "thinking" if either a separate reasoning field is populated,
    # or a <think> block appears in the content.
    reasoning_text = reasoning or embedded_think
    thinking = bool(reasoning_text.strip())

    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    tps = (completion_tokens / elapsed) if elapsed > 0 and completion_tokens else 0.0

    return {
        "elapsed_seconds": round(elapsed, 2),
        "reported_model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "tokens_per_second": round(tps, 1),
        "thinking_detected": thinking,
        "reasoning_chars": len(reasoning_text),
        "reasoning_preview": reasoning_text[:300],
        "translation": visible,
    }


def main() -> int:
    args = build_parser().parse_args()
    results: dict[str, list[dict]] = {}

    print(f"Source text ({len(SOURCE_TEXT)} chars):\n{SOURCE_TEXT}\n")
    print("=" * 80)

    for model in args.models:
        print(f"\n### {model}")
        runs: list[dict] = []
        for i in range(1, args.runs + 1):
            try:
                r = call_model(
                    args.base_url, args.api_key, model, args.timeout, args.max_tokens
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                print(f"  run {i}: HTTP {exc.code}: {detail}")
                runs.append({"error": f"HTTP {exc.code}: {detail}"})
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  run {i}: FAILED: {exc}")
                runs.append({"error": str(exc)})
                continue

            think = "THINKING" if r["thinking_detected"] else "no-think"
            print(
                f"  run {i}: {r['elapsed_seconds']:>6}s  "
                f"{r['tokens_per_second']:>6} tok/s  "
                f"{r['usage'].get('completion_tokens', '?')} out-tok  "
                f"[{think}, reasoning={r['reasoning_chars']} chars]  "
                f"finish={r['finish_reason']}"
            )
            runs.append(r)
        results[model] = runs

        # Print the last successful translation for quality review.
        good = [r for r in runs if "translation" in r]
        if good:
            print(f"\n  --- translation (run {len(good)}) ---")
            print("  " + good[-1]["translation"].replace("\n", "\n  "))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"source": SOURCE_TEXT, "system_prompt": SYSTEM_PROMPT, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n\nRaw results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
