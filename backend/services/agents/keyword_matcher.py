"""Algorithmic keyword matching utilities for deterministic ATS scoring.

Provides word-boundary regex matching (not substring) to avoid false positives
like 'go' matching 'going' or 'python' matching 'pythonic'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class KeywordMatchResult:
    """Result of matching JD keywords against resume text."""
    matched: list[str]
    missing: list[str]
    match_percentage: float  # 0.0 – 100.0


def keyword_in_text(keyword: str, text: str) -> bool:
    """Check if *keyword* appears as a whole word in *text*.

    Uses word boundaries to avoid false positives like 'go' matching 'going'.
    """
    escaped = re.escape(keyword.lower())
    return bool(re.search(rf"\b{escaped}\b", text.lower()))


def calculate_keyword_match(
    text: str,
    keywords: list[str],
    *,
    categories: dict[str, list[str]] | None = None,
) -> KeywordMatchResult:
    """Calculate what percentage of *keywords* appear in *text*.

    When *categories* is provided, technical_skills and tools_platforms are
    weighted 1.5x and certifications 1.25x to reflect real ATS behaviour where
    hard-skill matches matter more than soft-skill matches.
    """
    if not keywords:
        return KeywordMatchResult(matched=[], missing=[], match_percentage=0.0)

    text_lower = text.lower()

    # Build a weight map from categories
    weights: dict[str, float] = {}
    if categories:
        high_weight_cats = {"technical_skills", "tools_platforms"}
        med_weight_cats = {"certifications", "domain_knowledge"}
        for cat, kws in categories.items():
            for kw in kws:
                if cat in high_weight_cats:
                    weights[kw.lower()] = 1.5
                elif cat in med_weight_cats:
                    weights[kw.lower()] = 1.25
                else:
                    weights[kw.lower()] = 1.0

    matched: list[str] = []
    missing: list[str] = []

    for kw in keywords:
        if keyword_in_text(kw, text_lower):
            matched.append(kw)
        else:
            missing.append(kw)

    # Weighted score
    if weights:
        total_weight = sum(weights.get(kw.lower(), 1.0) for kw in keywords)
        matched_weight = sum(weights.get(kw.lower(), 1.0) for kw in matched)
        pct = (matched_weight / total_weight) * 100 if total_weight else 0.0
    else:
        pct = (len(matched) / len(keywords)) * 100

    return KeywordMatchResult(
        matched=matched,
        missing=missing,
        match_percentage=round(pct, 1),
    )
