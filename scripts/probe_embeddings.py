#!/usr/bin/env python3
"""Probe embedding models' cosine separation on realistic glossary entity pairs.

This is the Phase 2A *acceptance-gate* measurement. The auto-merge-safety case
rests on the claim that adjacent-but-distinct entities (e.g. 准仙帝/仙帝 — a
cultivation realm just below another) score a LOWER cosine than genuine aliases
(e.g. アベル/ヴィンセント — the same person under two names). The thresholds in
GlossaryClusterConfig only mean something if the real model produces that gap.

What this script measures
-------------------------
* Multiple candidate models (MiniLM baseline + larger multilingual models).
* Two summary DENSITIES per pair:
    - "thin"  : minimal one-line summaries (a deliberate stress test; close to
                what naive Phase 1 summaries look like).
    - "dense" : richer, post-reveal *converged* summaries — what a full book
                plus the 2B compressor would accumulate. Clustering uses the
                latest/global summary, so the converged text is the correct
                production input.
* Wall-clock load + encode time per model (CPU cost visibility).

For each pair/variant the cosine is checked against the configured thresholds:
    embedding_candidate_threshold        (>= -> becomes an LLM candidate)
    embedding_auto_merge_min_cosine      (>= -> may corroborate a HIGH auto-merge)
    embedding_low_confidence_max_cosine  (<  -> embedding-only pair auto-rejected)

The acceptance gate (reported on the "dense" variant, the production config):
    * "should stay separate" pairs must fall BELOW the auto-merge floor.
    * "should merge" pairs ideally reach AT/ABOVE the candidate threshold
      (otherwise the embedding heuristic never even surfaces them).

Usage:
    python scripts/probe_embeddings.py
    python scripts/probe_embeddings.py --models BAAI/bge-m3 intfloat/multilingual-e5-large
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dao_bridge.config import GlossaryClusterConfig  # noqa: E402

# Default model line-up to compare. CPU-portable, full-precision models only.
# The bitsandbytes-quantized 4B variants
# (lainlives/Qwen3-Embedding-4B-bnb-4bit, Octen/Octen-Embedding-4B-INT8) both
# lost the bake-off and require bitsandbytes + accelerate + CUDA — pass them
# explicitly via --models to re-probe them.
DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
]


@dataclass
class Entity:
    category: str
    name: str
    source: str
    thin: str
    dense: str


@dataclass
class Probe:
    label: str
    should_merge: bool  # genuine alias (True) vs adjacent-but-distinct (False)
    a: Entity
    b: Entity


PROBES: list[Probe] = [
    Probe(
        label="准仙帝 / 仙帝 (adjacent cultivation realms — DISTINCT)",
        should_merge=False,
        a=Entity(
            category="title",
            name="Quasi-Immortal Emperor",
            source="准仙帝",
            thin="A cultivation rank below the Immortal Emperor realm.",
            dense=(
                "A cultivation rank that sits one full stage BELOW the Immortal "
                "Emperor. Practitioners at this level have not yet completed the "
                "final ascension trial and remain mortal; they cannot wield the "
                "law-rending authority that true Immortal Emperors command. Often "
                "a stepping-stone realm reached centuries before genuine "
                "immortality, if ever."
            ),
        ),
        b=Entity(
            category="title",
            name="Immortal Emperor",
            source="仙帝",
            thin="The highest cultivation realm; a true immortal sovereign.",
            dense=(
                "The supreme cultivation realm. A true immortal sovereign who has "
                "transcended mortality, rends natural law at will, and rules over "
                "all lesser cultivators including the merely quasi-immortal. "
                "Attaining this realm is the final ascension; only a handful exist "
                "in an entire era."
            ),
        ),
    ),
    Probe(
        label="アベル / ヴィンセント (fugitive name vs true name — SAME person)",
        should_merge=True,
        a=Entity(
            category="character",
            name="Abel",
            source="アベル",
            thin="A fugitive traveling under the name Abel.",
            dense=(
                "A man traveling incognito under the alias 'Abel.' Carries himself "
                "with unmistakable imperial bearing and issues commands as if born "
                "to rule. Strongly implied — and later confirmed — to be the "
                "deposed Emperor of Vollachia hiding from a usurper. Sardonic, "
                "calculating, treats subordinates as pieces on a board."
            ),
        ),
        b=Entity(
            category="character",
            name="Vincent Volakia",
            source="ヴィンセント・ヴォラキア",
            thin="The emperor of Vollachia, ruling under the name Vincent.",
            dense=(
                "The 77th Emperor of the Vollachian Empire. Sardonic, ruthlessly "
                "pragmatic ruler who was deposed and forced into hiding under an "
                "assumed identity after a coup. While in hiding he travels under a "
                "false name. The same man who issues imperial commands despite his "
                "fugitive status; reclaiming the throne is his aim."
            ),
        ),
    ),
    Probe(
        label="Huang / Da Zhuang (coincidental JW match — DISTINCT people)",
        should_merge=False,
        a=Entity(
            category="character",
            name="Huang",
            source="黄",
            thin="A minor merchant in the market district.",
            dense=(
                "A minor traveling merchant who appears briefly hawking wares in "
                "the eastern market district. No combat ability, no relation to "
                "the main cast; exists to sell the protagonist a map and vanish "
                "from the story."
            ),
        ),
        b=Entity(
            category="character",
            name="Da Zhuang",
            source="大壮",
            thin="A burly village blacksmith.",
            dense=(
                "A burly, good-natured village blacksmith and the protagonist's "
                "childhood friend. Forges the hero's first sword, stays behind to "
                "defend the village, and represents the humble life the hero "
                "leaves behind. Unrelated to any merchant."
            ),
        ),
    ),
    Probe(
        label="Petelgeuse / Petelgeous (romanisation variant — SAME entity)",
        should_merge=True,
        a=Entity(
            category="character",
            name="Petelgeuse",
            source="ペテルギウス",
            thin="A fanatical archbishop of the Witch Cult obsessed with sloth.",
            dense=(
                "A fanatical archbishop of the Witch Cult embodying the sin of "
                "Sloth. Wild-eyed, spine-contorting zealot who hijacks others' "
                "bodies and screams about diligence and love for the Witch."
            ),
        ),
        b=Entity(
            category="character",
            name="Petelgeous",
            source="ペテルギウス",
            thin="A deranged Witch Cult archbishop representing Sloth.",
            dense=(
                "A deranged Witch Cult archbishop representing the sin of Sloth. "
                "Possesses host bodies, contorts grotesquely, and raves about "
                "diligence; the same Sloth archbishop, romanised differently."
            ),
        ),
    ),
]


def _embedding_text(e: Entity, dense: bool) -> str:
    """Mirror dao_bridge.glossary_embeddings.entity_embedding_text assembly."""
    summary = e.dense if dense else e.thin
    parts = [e.category, e.name, e.source, e.name, summary]
    return ". ".join(p for p in parts if p)


# Model-family-specific input formatting. e5 REQUIRES a prefix or cosines are
# meaningless; bge-m3 and MiniLM use raw text; Qwen3 embedding works without an
# instruction for symmetric similarity but benefits from one — kept raw here for
# a fair symmetric comparison.
def _format_for_model(model_name: str, text: str) -> str:
    lower = model_name.lower()
    if "e5" in lower:
        # Symmetric similarity: both sides as "query:".
        return f"query: {text}"
    return text


def _encode(model, model_name: str, texts: list[str]):
    import numpy as np

    formatted = [_format_for_model(model_name, t) for t in texts]
    emb = model.encode(
        formatted,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(emb)


def _cosine(a, b) -> float:
    return float(a @ b)


def probe_model(model_name: str, cfg: GlossaryClusterConfig) -> bool:
    """Probe one model. Returns True if the acceptance gate passes (dense)."""
    from sentence_transformers import SentenceTransformer

    cand = cfg.embedding_candidate_threshold
    auto = cfg.embedding_auto_merge_min_cosine
    low = cfg.embedding_low_confidence_max_cosine

    print("=" * 74)
    print(f"MODEL: {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    t_load = time.perf_counter() - t0
    print(f"  load time: {t_load:6.2f}s")

    # Encode everything once, time it.
    all_texts: list[str] = []
    for p in PROBES:
        for dense in (False, True):
            all_texts.append(_embedding_text(p.a, dense))
            all_texts.append(_embedding_text(p.b, dense))
    t0 = time.perf_counter()
    embs = _encode(model, model_name, all_texts)
    t_encode = time.perf_counter() - t0
    per = t_encode / len(all_texts) * 1000
    print(f"  encode time: {t_encode:6.2f}s for {len(all_texts)} texts ({per:.1f} ms/text)")
    print(f"  thresholds: candidate>={cand:.2f}  auto>={auto:.2f}  low<{low:.2f}\n")

    idx = 0
    gate_failures: list[str] = []
    missed_aliases: list[str] = []
    for p in PROBES:
        verdict = "SHOULD MERGE" if p.should_merge else "SHOULD STAY SEPARATE"
        print(f"  {p.label}")
        print(f"      expectation: {verdict}")
        for variant, _dense in (("thin", False), ("dense", True)):
            ea, eb = embs[idx], embs[idx + 1]
            idx += 2
            c = _cosine(ea, eb)
            tier = "auto" if c >= auto else ("candidate" if c >= cand else "below-cand")
            print(f"      [{variant:5}] cosine={c:+.3f}  -> {tier}")
            if variant == "dense":
                if not p.should_merge and c >= auto:
                    gate_failures.append(f"{p.label}: {c:.3f} >= auto floor {auto:.2f}")
                if p.should_merge and c < cand:
                    missed_aliases.append(f"{p.label}: {c:.3f} < candidate {cand:.2f}")
        print()

    if gate_failures:
        print("  ACCEPTANCE GATE (dense): FAILED")
        for f in gate_failures:
            print(f"    - distinct pair too close: {f}")
    else:
        print("  ACCEPTANCE GATE (dense): PASSED (no distinct pair at/above auto floor)")
    if missed_aliases:
        print("  ALIAS RECALL: weak (these genuine merges fall below candidate threshold)")
        for m in missed_aliases:
            print(f"    - missed: {m}")
    print()
    return not gate_failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="SentenceTransformers model names to compare.",
    )
    args = parser.parse_args()
    cfg = GlossaryClusterConfig()

    results: dict[str, bool] = {}
    for model_name in args.models:
        try:
            results[model_name] = probe_model(model_name, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR loading/running {model_name}: {exc}\n")
            results[model_name] = False

    print("=" * 74)
    print("SUMMARY (acceptance gate on dense summaries)")
    for model_name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
