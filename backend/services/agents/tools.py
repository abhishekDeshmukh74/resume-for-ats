"""Deterministic scoring and analysis tools.

These are non-LLM tools used by agents for objective measurements:
  - keyword coverage calculator
  - embedding similarity calculator
  - duplication detector
  - bullet length checker
  - ATS format checker
  - unsupported claim checker
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
    """Count which keywords appear in *text* (case-insensitive fuzzy match).

    Returns:
        {
            "covered": [...],
            "missing": [...],
            "coverage_pct": float (0-100),
            "weighted_score": float (0-100) if weights provided
        }
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
    """Detect keywords that appear too many times in *text*.

    Returns list of {"keyword": str, "count": int, "max": int}.
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
    """Evaluate individual bullet points for ATS quality.

    Returns per-bullet analysis:
        {
            "bullet": str,
            "length": int,
            "starts_with_verb": bool,
            "has_metric": bool,
            "too_short": bool,
            "too_long": bool,
            "quality": "good" | "weak" | "bad"
        }
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
    """Check resume text for ATS-friendliness.

    Returns:
        {
            "score": int (0-100),
            "issues": [str],
            "has_standard_headings": bool,
            "section_order_ok": bool,
        }
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
    """Flag skills in the rewritten version not found in the original.

    Returns list of {"type": str, "value": str, "reason": str}.
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
    """Dedupe and normalize common skill name variants."""
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
    """Compute a weighted composite ATS score.

    Formula:
        0.30 * keyword_match +
        0.25 * semantic_alignment +
        0.20 * section_quality +
        0.15 * ats_format +
        0.10 * truthfulness
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
