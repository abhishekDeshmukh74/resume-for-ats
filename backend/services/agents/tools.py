"""Deterministic scoring and analysis tools (non-LLM).

This module provides objective, reproducible measurements that agents use
alongside LLM calls.  By running deterministic checks first, we give the
LLM concrete numbers to anchor its evaluation and avoid pure "vibe" scoring.

Used by
~~~~~~~
* ``gap_analyzer.py``   — ``compute_keyword_coverage()`` to measure keyword
  overlap before sending to the LLM for nuanced gap analysis.
* ``scorer.py``         — ``compute_keyword_coverage()``, ``check_ats_format()``,
  ``check_bullet_quality()``, ``detect_keyword_stuffing()``, and
  ``compute_ats_score()`` to build the hybrid deterministic + LLM composite score.
* ``skills_optimizer.py``— ``normalize_skill_names()`` to dedupe / canonicalise
  skill names before and after LLM optimization.
* ``truth_guard.py``    — ``check_unsupported_claims()`` to flag skills in the
  optimized resume that don’t appear in the original.
* ``critic.py``         — ``detect_keyword_stuffing()`` to inject stuffing
  warnings into the critic review.

Scoring formula (``compute_ats_score``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    overall = (0.30 * keyword_coverage
            + 0.25 * semantic_alignment     # from LLM
            + 0.20 * section_quality        # from LLM
            + 0.15 * ats_format             # from check_ats_format()
            + 0.10 * truthfulness)          # 100 baseline, penalised by truth_guard
"""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Keyword Coverage
# ---------------------------------------------------------------------------

def compute_keyword_coverage(
    text: str,
    keywords: list[str],
    weights: dict[str, int] | None = None,
) -> dict:
    """Measure which JD keywords appear in ``text`` (case-insensitive, fuzzy).

    Uses two strategies:

    1. **Exact substring** match (lowercased) for multi-word keywords.
    2. **Fuzzy token** match via ``rapidfuzz.fuzz.ratio ≥ 85`` for single-word
       variants (e.g., "TypeScript" vs "Typescript").

    Called by:
        * ``gap_analyzer.compute_gap_node``  — provides deterministic coverage
          numbers that the LLM uses as an input signal.
        * ``scorer.score_resume``            — produces the ``keyword_match``
          component (weight 0.30) of the composite ATS score.

    Args:
        text:     The full resume text (or reconstructed text blob) to search.
        keywords: List of JD keywords to look for.
        weights:  Optional ``{keyword: int}`` importance weights (1–10 scale,
                  from ``parsed_jd["skill_weights"]``).  When provided the
                  return dict includes a ``weighted_score``.

    Returns:
        dict with keys:
            * ``covered``        — list of matched keywords
            * ``missing``        — list of unmatched keywords
            * ``coverage_pct``   — float 0–100 (unweighted)
            * ``weighted_score`` — float 0–100 (only when *weights* given)
    """
    text_lower = text.lower()
    covered: list[str] = []
    missing: list[str] = []

    for kw in keywords:
        kw_lower = kw.lower()
        # Exact substring match
        if kw_lower in text_lower:
            covered.append(kw)
            continue
        # Fuzzy token match (handles minor variations)
        words = text_lower.split()
        if any(fuzz.ratio(kw_lower, w) >= 85 for w in words):
            covered.append(kw)
        else:
            missing.append(kw)

    coverage_pct = (len(covered) / len(keywords) * 100) if keywords else 0.0

    result: dict = {
        "covered": covered,
        "missing": missing,
        "coverage_pct": round(coverage_pct, 1),
    }

    if weights:
        total_weight = sum(weights.get(kw, 1) for kw in keywords)
        covered_weight = sum(weights.get(kw, 1) for kw in covered)
        result["weighted_score"] = round(
            (covered_weight / total_weight * 100) if total_weight else 0.0, 1
        )

    return result


# ---------------------------------------------------------------------------
# Duplication Detector
# ---------------------------------------------------------------------------

def detect_keyword_stuffing(text: str, keywords: list[str], max_count: int = 3) -> list[dict]:
    """Flag keywords that appear too many times in ``text``.

    Keyword stuffing is a common problem when LLMs optimise for ATS — they
    repeat high-weight keywords 4–6 times, which looks spammy to human
    recruiters and some ATS systems.

    Called by:
        * ``scorer.score_resume``  — included in the score report so downstream
          agents know about stuffing issues.
        * ``critic.critic_node``   — injected into the LLM review prompt and
          added as ``keyword_stuffing`` issues in the critic report.

    Args:
        text:      The resume text to scan.
        keywords:  JD keywords to check.
        max_count: Threshold above which a keyword is flagged (default 3).

    Returns:
        List of ``{"keyword": str, "count": int, "max": int}`` for each
        over-represented keyword.  Empty list means no stuffing detected.
    """
    text_lower = text.lower()
    violations: list[dict] = []
    for kw in keywords:
        count = text_lower.count(kw.lower())
        if count > max_count:
            violations.append({
                "keyword": kw,
                "count": count,
                "max": max_count,
            })
    return violations


# ---------------------------------------------------------------------------
# Bullet Quality Checker
# ---------------------------------------------------------------------------

_STRONG_VERB_RE = re.compile(
    r"^(built|created|designed|developed|implemented|improved|led|managed|"
    r"reduced|increased|automated|deployed|integrated|migrated|optimized|"
    r"delivered|established|launched|maintained|refactored|resolved|scaled|"
    r"streamlined|architected|configured|coordinated|engineered|executed|"
    r"extended|modernized|spearheaded|transformed|wrote)\b",
    re.IGNORECASE,
)


def check_bullet_quality(bullets: list[str]) -> list[dict]:
    """Rate individual resume bullet points for ATS quality.

    Each bullet is evaluated on three criteria:

    * **Starts with a strong action verb** — matches a curated regex of
      ~35 strong verbs ("built", "designed", "reduced", etc.).
    * **Contains a quantifiable metric** — numbers, percentages, user counts.
    * **Appropriate length** — not too short (< 30 chars) or too long (> 250).

    Quality grades:
        * ``"good"``  — starts with verb AND has metric AND within length.
        * ``"weak"``  — has verb OR has metric, but not both.
        * ``"bad"``   — neither verb nor metric.

    Called by:
        * ``scorer.score_resume`` — the ratio of ``good`` bullets feeds into
          the ``section_quality`` input passed to the LLM rubric scorer.

    Args:
        bullets: List of bullet-point strings from experience/projects.

    Returns:
        List of per-bullet dicts with keys: ``bullet``, ``length``,
        ``starts_with_verb``, ``has_metric``, ``too_short``, ``too_long``,
        ``quality``.
    """
    results: list[dict] = []
    for b in bullets:
        stripped = b.strip().lstrip("•-– ")
        length = len(stripped)
        starts_verb = bool(_STRONG_VERB_RE.match(stripped))
        has_metric = bool(re.search(r"\d+[%xX]|\d+\+|\$\d|\d+\s*(users|clients|ms|seconds|minutes|hours|requests|%)", stripped))
        too_short = length < 30
        too_long = length > 250

        if starts_verb and has_metric and not too_short and not too_long:
            quality = "good"
        elif starts_verb or has_metric:
            quality = "weak"
        else:
            quality = "bad"

        results.append({
            "bullet": b,
            "length": length,
            "starts_with_verb": starts_verb,
            "has_metric": has_metric,
            "too_short": too_short,
            "too_long": too_long,
            "quality": quality,
        })
    return results


# ---------------------------------------------------------------------------
# ATS Format Checker
# ---------------------------------------------------------------------------

_STANDARD_HEADINGS = {
    "summary", "profile", "professional summary", "objective",
    "experience", "work experience", "professional experience", "employment",
    "education", "academic background",
    "skills", "technical skills", "core competencies",
    "certifications", "licenses",
    "projects", "personal projects", "key projects",
}


def check_ats_format(resume_text: str) -> dict:
    """Evaluate resume text for ATS-friendliness (structural checks).

    Checks performed (each deducts from a 100-point baseline):

    * **Standard headings** (−15): Fewer than 3 recognised section headings
      (SUMMARY, EXPERIENCE, SKILLS, EDUCATION, etc.).
    * **Table/box characters** (−10): Unicode box-drawing chars that confuse
      ATS parsers (``│┌┐═║`` etc.).
    * **Date formatting** (−5): Fewer than 2 recognisable date patterns.
    * **Contact info** (−5 each): Missing email or phone number.

    Called by:
        * ``scorer.score_resume`` — the ``score`` field becomes the
          ``ats_format`` component (weight 0.15) of the composite ATS score.

    Args:
        resume_text: Plain-text resume content.

    Returns:
        dict with keys: ``score`` (0–100), ``issues`` (list[str]),
        ``has_standard_headings`` (bool), ``found_headings`` (list[str]).
    """
    issues: list[str] = []
    lines = resume_text.strip().splitlines()
    score = 100

    # Check for standard headings
    found_headings = set()
    for line in lines:
        stripped = line.strip().lower().rstrip(":")
        if stripped in _STANDARD_HEADINGS:
            found_headings.add(stripped)

    has_std = len(found_headings) >= 3
    if not has_std:
        issues.append("Fewer than 3 standard ATS headings found.")
        score -= 15

    # Check for excessive special characters (tables, graphics indicators)
    special_chars = sum(1 for c in resume_text if c in "│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬")
    if special_chars > 5:
        issues.append("Contains table/box-drawing characters that may confuse ATS.")
        score -= 10

    # Check for consistent date formatting
    date_patterns = re.findall(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
        r"|\b\d{1,2}/\d{4}\b"
        r"|\b\d{4}\s*[-–]\s*(?:Present|\d{4})\b",
        resume_text,
    )
    if len(date_patterns) < 2:
        issues.append("Few recognizable dates found — ensure standard date formatting.")
        score -= 5

    # Check for contact info presence
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text))
    has_phone = bool(re.search(r"[\d\(\)\-\+\s]{7,}", resume_text))
    if not has_email:
        issues.append("No email address detected.")
        score -= 5
    if not has_phone:
        issues.append("No phone number detected.")
        score -= 5

    return {
        "score": max(0, score),
        "issues": issues,
        "has_standard_headings": has_std,
        "found_headings": list(found_headings),
    }


# ---------------------------------------------------------------------------
# Unsupported Claim Checker (deterministic layer)
# ---------------------------------------------------------------------------

def check_unsupported_claims(
    original_text: str,
    rewritten_skills: list[str],
    original_skills: list[str],
) -> list[dict]:
    """Flag skills that appear in the optimized resume but not in the original.

    Uses three detection strategies (in order):

    1. **Text search** — lowercased substring match against the full original
       resume text.
    2. **Skill-list match** — exact match against the parsed original skills.
    3. **Fuzzy fallback** — ``rapidfuzz.fuzz.ratio ≥ 85`` against original
       skills (catches "React.js" vs "ReactJS").

    Called by:
        * ``truth_guard.truth_guard_node`` — runs this *before* the LLM depth
          check to provide concrete evidence of unsupported skills that the LLM
          can use (and extend) in its analysis.

    Args:
        original_text:    The full original resume text (all sections joined).
        rewritten_skills: Flat list of skills from the optimized draft.
        original_skills:  Flat list of skills from the parsed original resume.

    Returns:
        List of ``{"type": "unsupported_skill", "value": str, "reason": str}``
        for each skill not traceable to the original.  Empty list = all clean.
    """
    original_lower = original_text.lower()
    original_skills_lower = {s.lower() for s in original_skills}
    violations: list[dict] = []

    for skill in rewritten_skills:
        skill_lower = skill.lower()
        # Check both the raw text and the parsed skill list
        in_text = skill_lower in original_lower
        in_skills = skill_lower in original_skills_lower
        # Fuzzy fallback
        fuzzy_match = any(
            fuzz.ratio(skill_lower, s) >= 85 for s in original_skills_lower
        )
        if not in_text and not in_skills and not fuzzy_match:
            violations.append({
                "type": "unsupported_skill",
                "value": skill,
                "reason": "Not found in source resume",
            })

    return violations


# ---------------------------------------------------------------------------
# Section Normalizer
# ---------------------------------------------------------------------------

def normalize_skill_names(skills: list[str]) -> list[str]:
    """De-duplicate and canonicalise common skill name variants.

    Maintains a static lookup table of ~40 common aliases (e.g., ``js`` →
    ``JavaScript``, ``k8s`` → ``Kubernetes``).  Unknown skill names are
    returned unchanged but de-duplicated.

    Called by:
        * ``skills_optimizer.optimize_skills_node`` — applied *before* the LLM
          call (so it sees clean names) and again *after* (to catch any
          remaining inconsistencies the LLM introduced).

    Args:
        skills: Raw skill names, potentially with duplicates or aliases.

    Returns:
        De-duplicated list with canonical names (order preserved).
    """
    _CANONICAL: dict[str, str] = {
        "js": "JavaScript",
        "javascript": "JavaScript",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "reactjs": "React",
        "react.js": "React",
        "react js": "React",
        "nextjs": "Next.js",
        "next.js": "Next.js",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node js": "Node.js",
        "expressjs": "Express.js",
        "express.js": "Express.js",
        "vuejs": "Vue.js",
        "vue.js": "Vue.js",
        "angularjs": "Angular",
        "angular.js": "Angular",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "mongodb": "MongoDB",
        "mongo": "MongoDB",
        "amazon web services": "AWS",
        "google cloud platform": "GCP",
        "gcp": "GCP",
        "k8s": "Kubernetes",
        "kubernetes": "Kubernetes",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        "ml": "Machine Learning",
        "machine learning": "Machine Learning",
        "dl": "Deep Learning",
        "deep learning": "Deep Learning",
        "nlp": "NLP",
        "natural language processing": "NLP",
    }

    seen: set[str] = set()
    normalized: list[str] = []
    for skill in skills:
        canonical = _CANONICAL.get(skill.lower().strip(), skill.strip())
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            normalized.append(canonical)
    return normalized


# ---------------------------------------------------------------------------
# Composite ATS Score
# ---------------------------------------------------------------------------

def compute_ats_score(
    keyword_coverage_pct: float,
    semantic_score: float,
    section_quality_score: float,
    ats_format_score: float,
    truthfulness_score: float,
) -> dict:
    """Compute the weighted composite ATS score.

    Formula::

        overall = (0.30 × keyword_coverage
                 + 0.25 × semantic_alignment
                 + 0.20 × section_quality
                 + 0.15 × ats_format
                 + 0.10 × truthfulness)

    All inputs are on a 0–100 scale.  The output ``overall_score`` is also
    0–100.

    Called by:
        * ``scorer.score_resume`` — combines deterministic scores (keyword,
          format) with LLM rubric scores (semantic, section quality) and a
          truthfulness baseline.

    Args:
        keyword_coverage_pct:  From ``compute_keyword_coverage.weighted_score``.
        semantic_score:        LLM-evaluated semantic alignment (0–100).
        section_quality_score: LLM-evaluated section quality (0–100).
        ats_format_score:      From ``check_ats_format.score``.
        truthfulness_score:    100 at baseline; penalised by truth guard violations.

    Returns:
        dict with ``overall_score`` (float) and ``breakdown`` (dict of
        individual dimension scores).
    """
    overall = (
        0.30 * keyword_coverage_pct
        + 0.25 * semantic_score
        + 0.20 * section_quality_score
        + 0.15 * ats_format_score
        + 0.10 * truthfulness_score
    )
    return {
        "overall_score": round(overall, 1),
        "breakdown": {
            "keyword_match": round(keyword_coverage_pct, 1),
            "semantic_match": round(semantic_score, 1),
            "section_quality": round(section_quality_score, 1),
            "ats_format": round(ats_format_score, 1),
            "truthfulness": round(truthfulness_score, 1),
        },
    }
