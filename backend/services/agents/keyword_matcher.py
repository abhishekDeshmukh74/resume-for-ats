"""Enhanced ATS keyword matching with fuzzy matching via rapidfuzz.

Provides:
- Exact word-boundary matching (primary, avoids false positives)
- Abbreviation / synonym expansion for common tech terms
- Fuzzy matching via rapidfuzz for near-matches and typos
- Section-aware scoring (keyword placement in relevant sections)
- Keyword stuffing detection
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

# ── Common tech abbreviation / synonym map ───────────────────────────────
# Maps canonical form → list of known aliases.
_TECH_SYNONYMS: dict[str, list[str]] = {
    "javascript": ["js", "es6", "ecmascript"],
    "typescript": ["ts"],
    "python": ["py"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "database": ["db", "databases"],
    "continuous integration": ["ci"],
    "continuous deployment": ["cd"],
    "ci/cd": ["cicd", "ci cd", "ci-cd"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp"],
    "microsoft azure": ["azure"],
    "kubernetes": ["k8s"],
    "postgresql": ["postgres", "psql"],
    "mongodb": ["mongo"],
    "rest api": ["restful api", "rest apis", "restful"],
    "graphql": ["gql"],
    "react.js": ["react", "reactjs"],
    "node.js": ["node", "nodejs"],
    "vue.js": ["vue", "vuejs"],
    "angular.js": ["angular", "angularjs"],
    "next.js": ["next", "nextjs"],
    "express.js": ["express", "expressjs"],
    "devops": ["dev ops", "dev-ops"],
    "frontend": ["front end", "front-end"],
    "backend": ["back end", "back-end"],
    "full stack": ["fullstack", "full-stack"],
    "application programming interface": ["api", "apis"],
    "software development life cycle": ["sdlc"],
    "object oriented programming": ["oop"],
    "user interface": ["ui"],
    "user experience": ["ux"],
    "ui/ux": ["ui ux", "ux/ui"],
    "search engine optimization": ["seo"],
    "test driven development": ["tdd"],
    "behavior driven development": ["bdd"],
    "software as a service": ["saas"],
    "platform as a service": ["paas"],
    "infrastructure as a service": ["iaas"],
}

# Build reverse lookup: any alias → set of all equivalent forms
_ALIAS_MAP: dict[str, set[str]] = {}
for _canonical, _aliases in _TECH_SYNONYMS.items():
    _group = {_canonical.lower()} | {a.lower() for a in _aliases}
    for _term in _group:
        _ALIAS_MAP.setdefault(_term, set()).update(_group)


# ── Section placement weights ────────────────────────────────────────────
# Keywords found in high-value sections contribute more to the score.
_SECTION_WEIGHTS: dict[str, float] = {
    "title": 1.4,
    "summary": 1.3,
    "skills": 1.2,
    "experience": 1.0,
    "education": 0.9,
    "certifications": 1.1,
}
_DEFAULT_SECTION_WEIGHT = 1.0


def _sentence_position_bonus(keyword: str, text: str) -> float:
    """Return an additive bonus (0.0–0.15) if *keyword* appears near the
    beginning of a sentence.  ATS parsers weight early-sentence keywords
    more heavily because they signal intent rather than filler context.

    Returns 0.15 for first-third, 0.05 for middle-third, 0.0 for last-third.
    """
    kw_lower = keyword.lower()
    escaped = re.escape(kw_lower)
    # Split on sentence boundaries (period/newline followed by space or EOL)
    sentences = re.split(r'(?<=[.!?\n])\s+', text.lower())
    best = 0.0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        m = re.search(rf"\b{escaped}\b", sentence)
        if m is None:
            continue
        pos_ratio = m.start() / max(len(sentence), 1)
        if pos_ratio < 0.33:
            best = max(best, 0.15)
        elif pos_ratio < 0.66:
            best = max(best, 0.05)
        # else: no bonus
    return best


@dataclass
class KeywordMatch:
    """Details about how a single keyword was matched."""
    keyword: str
    match_type: str  # "exact", "synonym", "fuzzy", "none"
    confidence: float  # 0.0 – 100.0
    matched_form: str | None = None  # The form found in the text


@dataclass
class KeywordMatchResult:
    """Result of matching JD keywords against resume text."""
    matched: list[str]
    missing: list[str]
    match_percentage: float  # 0.0 – 100.0
    match_details: list[KeywordMatch] = field(default_factory=list)
    exact_count: int = 0
    synonym_count: int = 0
    fuzzy_count: int = 0
    section_scores: dict[str, float] = field(default_factory=dict)
    stuffing_penalty: float = 0.0  # 0.0 – 15.0 deducted from score


def keyword_in_text(keyword: str, text: str) -> bool:
    """Check if *keyword* appears as a whole word in *text*.

    Uses word boundaries to avoid false positives like 'go' matching 'going'.
    """
    escaped = re.escape(keyword.lower())
    return bool(re.search(rf"\b{escaped}\b", text.lower()))


def _synonym_match(keyword: str, text: str) -> str | None:
    """Return the matched synonym/alias found in *text*, or ``None``."""
    kw_lower = keyword.lower()
    aliases = _ALIAS_MAP.get(kw_lower, set())
    for alias in aliases:
        if alias != kw_lower and keyword_in_text(alias, text):
            return alias
    return None


def _fuzzy_match(
    keyword: str, text: str, *, threshold: int = 80,
) -> tuple[str | None, float]:
    """Fuzzy-match *keyword* against tokens / n-grams in *text*.

    Returns ``(matched_form, confidence)`` or ``(None, 0.0)``.
    """
    words = text.lower().split()
    if not words:
        return None, 0.0

    kw_lower = keyword.lower()
    kw_word_count = len(kw_lower.split())

    if kw_word_count == 1:
        result = process.extractOne(
            kw_lower, words, scorer=fuzz.ratio, score_cutoff=threshold,
        )
    else:
        ngrams = [
            " ".join(words[i : i + kw_word_count])
            for i in range(len(words) - kw_word_count + 1)
        ]
        if not ngrams:
            return None, 0.0
        result = process.extractOne(
            kw_lower, ngrams, scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
        )

    if result:
        return result[0], result[1]
    return None, 0.0


def _detect_stuffing(text: str, keywords: list[str], *, max_freq: int = 4) -> float:
    """Return a penalty (0.0–15.0) if keywords appear suspiciously often.

    Real ATS systems penalise resumes that repeat the same keyword many times
    to game the score.  We deduct up to 15 points based on the number of
    over-represented keywords.
    """
    text_lower = text.lower()
    # Count approximate occurrences of each keyword
    violations = 0
    for kw in keywords:
        escaped = re.escape(kw.lower())
        hits = len(re.findall(rf"\b{escaped}\b", text_lower))
        if hits > max_freq:
            violations += 1

    # Each violation costs 3 points, capped at 15
    return min(violations * 3.0, 15.0)


def calculate_section_scores(
    sections: dict[str, str],
    keywords: list[str],
) -> dict[str, float]:
    """Score each resume section independently against *keywords*.

    *sections* maps section name → section text.
    Returns section name → match percentage (0–100).
    """
    scores: dict[str, float] = {}
    for name, body in sections.items():
        if not body:
            scores[name] = 0.0
            continue
        body_lower = body.lower()
        matched = sum(1 for kw in keywords if keyword_in_text(kw, body_lower))
        scores[name] = round((matched / len(keywords)) * 100, 1) if keywords else 0.0
    return scores


def calculate_keyword_match(
    text: str,
    keywords: list[str],
    *,
    categories: dict[str, list[str]] | None = None,
    sections: dict[str, str] | None = None,
    fuzzy_threshold: int = 80,
) -> KeywordMatchResult:
    """Calculate what percentage of *keywords* appear in *text*.

    Matching pipeline (in priority order):
      1. Exact word-boundary match  → confidence 100
      2. Synonym / abbreviation     → confidence 95
      3. Fuzzy match via rapidfuzz  → confidence = similarity score

    Category weighting:
      - technical_skills, tools_platforms: 1.5×
      - certifications, domain_knowledge: 1.25×
      - Others: 1.0×

    When *sections* is provided, keywords found in high-value sections
    (title, summary, skills) receive an additional placement bonus.
    """
    if not keywords:
        return KeywordMatchResult(matched=[], missing=[], match_percentage=0.0)

    text_lower = text.lower()

    # ── Build category weight map ────────────────────────────────────
    cat_weights: dict[str, float] = {}
    if categories:
        high_cats = {"technical_skills", "tools_platforms"}
        med_cats = {"certifications", "domain_knowledge"}
        for cat, kws in categories.items():
            for kw in kws:
                if cat in high_cats:
                    cat_weights[kw.lower()] = 1.5
                elif cat in med_cats:
                    cat_weights[kw.lower()] = 1.25
                else:
                    cat_weights[kw.lower()] = 1.0

    # ── Build section placement bonus map ────────────────────────────
    placement_bonus: dict[str, float] = {}
    if sections:
        for kw in keywords:
            best_weight = _DEFAULT_SECTION_WEIGHT
            kw_lower = kw.lower()
            for sec_name, sec_text in sections.items():
                if sec_text and keyword_in_text(kw_lower, sec_text.lower()):
                    w = _SECTION_WEIGHTS.get(sec_name.lower(), _DEFAULT_SECTION_WEIGHT)
                    best_weight = max(best_weight, w)
            placement_bonus[kw_lower] = best_weight

    # ── Match each keyword ───────────────────────────────────────────
    matched: list[str] = []
    missing: list[str] = []
    details: list[KeywordMatch] = []
    exact_count = synonym_count = fuzzy_count = 0

    for kw in keywords:
        # 1. Exact word-boundary match
        if keyword_in_text(kw, text_lower):
            matched.append(kw)
            details.append(KeywordMatch(kw, "exact", 100.0, kw.lower()))
            exact_count += 1
            continue

        # 2. Synonym / abbreviation match
        syn = _synonym_match(kw, text_lower)
        if syn:
            matched.append(kw)
            details.append(KeywordMatch(kw, "synonym", 95.0, syn))
            synonym_count += 1
            continue

        # 3. Fuzzy match
        form, conf = _fuzzy_match(kw, text_lower, threshold=fuzzy_threshold)
        if form:
            matched.append(kw)
            details.append(KeywordMatch(kw, "fuzzy", conf, form))
            fuzzy_count += 1
            continue

        # No match
        missing.append(kw)
        details.append(KeywordMatch(kw, "none", 0.0))

    # ── Compute weighted score ───────────────────────────────────────
    total_weight = 0.0
    matched_weight = 0.0
    for d in details:
        kw_lower = d.keyword.lower()
        base = cat_weights.get(kw_lower, 1.0)
        place = placement_bonus.get(kw_lower, _DEFAULT_SECTION_WEIGHT)
        pos = _sentence_position_bonus(kw_lower, text_lower) if d.match_type != "none" else 0.0
        w = base * (place + pos)
        total_weight += base * place  # Denominator uses base weight (no position bonus)
        if d.match_type != "none":
            # Fuzzy matches contribute proportionally to their confidence
            matched_weight += w * (d.confidence / 100.0)

    pct = (matched_weight / total_weight) * 100 if total_weight else 0.0

    # ── Stuffing penalty ─────────────────────────────────────────────
    stuffing = _detect_stuffing(text, keywords)
    pct = max(pct - stuffing, 0.0)

    # ── Section scores ───────────────────────────────────────────────
    sec_scores = calculate_section_scores(sections, keywords) if sections else {}

    return KeywordMatchResult(
        matched=matched,
        missing=missing,
        match_percentage=round(pct, 1),
        match_details=details,
        exact_count=exact_count,
        synonym_count=synonym_count,
        fuzzy_count=fuzzy_count,
        section_scores=sec_scores,
        stuffing_penalty=stuffing,
    )
