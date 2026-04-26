"""String similarity utilities for glossary entity linking.

Provides bi-directional Jaro-Winkler similarity, adapted from the
``context-aware-translation`` reference repository.

Suggested threshold interpretation::

    >= 0.95   Very strong candidate — auto-attach if category/evidence agree
    0.75–0.95 Candidate for clustering / LLM review
    < 0.75    Ignore unless another signal matches
"""

from __future__ import annotations

import jellyfish


def string_similarity(a: str, b: str) -> float:
    """Compute string similarity using bi-directional Jaro-Winkler.

    Forward comparison handles prefix matches (standard Jaro-Winkler
    behaviour).  Reversed comparison handles suffix matches by converting
    them into prefix matches.  Returns the maximum of both to catch both
    patterns.

    Parameters
    ----------
    a, b:
        Strings to compare.

    Returns
    -------
    float
        Similarity score in ``[0.0, 1.0]``.
    """
    if not a or not b:
        return 0.0

    a = a.lower().strip()
    b = b.lower().strip()

    if a == b:
        return 1.0

    forward = jellyfish.jaro_winkler_similarity(a, b)
    reverse = jellyfish.jaro_winkler_similarity(a[::-1], b[::-1])
    return max(forward, reverse)
