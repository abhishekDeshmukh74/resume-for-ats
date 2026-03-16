"""Agent 8 — ATS Scoring Agent.

Hybrid scorer that combines:
  1. Deterministic keyword coverage (exact match)
  2. LLM rubric evaluation (semantic match, quality, alignment)
  3. ATS format checker (structural)
  4. Truthfulness score

Scores resume on multiple dimensions, not a single LLM vibe-check.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState
from backend.services.agents.tools import (
    check_ats_format,
    check_bullet_quality,
    compute_ats_score,
    compute_keyword_coverage,
    detect_keyword_stuffing,
)

logger = logging.getLogger(__name__)

_SCORE_SYSTEM = """You are an ATS scoring specialist using a structured rubric.

You receive a resume, JD analysis, and deterministic analysis results.
Score the resume on these dimensions (0-100 each):

1. SEMANTIC MATCH: How well does the resume's overall narrative align with the JD?
   Not just keyword presence, but contextual relevance of experience.

2. SECTION QUALITY: Rate the quality of bullets, summary, and skills section.
   - Clarity: Are bullets specific and clear?
   - Impact: Do bullets show measurable results?
   - Alignment: Does each section support the target role?

3. ROLE ALIGNMENT: Does the candidate's trajectory match the target role?
   - Seniority fit
   - Domain fit
   - Responsibility overlap

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "semantic_score": 80,
  "section_quality_score": 75,
  "role_alignment_score": 85,
  "impact_clarity": 78,
  "weak_sections": ["summary", "skills"],
  "strong_sections": ["experience"],
  "recommendations": [
    "Add more quantifiable metrics to experience bullets",
    "Summary should emphasize cloud experience more"
  ],
  "missing_high_weight_terms": ["Prisma", "Jest"]
}"""


def _build_resume_text(parsed_resume: dict) -> str:
    """Reconstruct readable text from parsed resume structure."""
    parts: list[str] = []
    basics = parsed_resume.get("basics", {})
    if basics.get("name"):
        parts.append(basics["name"])
    if parsed_resume.get("summary"):
        parts.append(f"\nSummary:\n{parsed_resume['summary']}")

    skills = parsed_resume.get("skills", {})
    if isinstance(skills, dict):
        parts.append("\nSkills:")
        for cat, items in skills.items():
            if isinstance(items, list) and items:
                parts.append(f"  {cat}: {', '.join(items)}")

    for exp in parsed_resume.get("experience", []):
        parts.append(f"\n{exp.get('title', '')} at {exp.get('company', '')}")
        for bullet in exp.get("bullets", []):
            parts.append(f"  - {bullet}")

    for proj in parsed_resume.get("projects", []):
        parts.append(f"\nProject: {proj.get('name', '')}")
        for bullet in proj.get("bullets", []):
            parts.append(f"  - {bullet}")

    for edu in parsed_resume.get("education", []):
        parts.append(f"\n{edu.get('degree', '')} — {edu.get('institution', '')}")

    return "\n".join(parts)


def _collect_bullets(parsed_resume: dict) -> list[str]:
    """Collect all bullet points from experience and projects."""
    bullets: list[str] = []
    for exp in parsed_resume.get("experience", []):
        bullets.extend(exp.get("bullets", []))
    for proj in parsed_resume.get("projects", []):
        bullets.extend(proj.get("bullets", []))
    return bullets


def score_resume(parsed_resume: dict, parsed_jd: dict, is_baseline: bool = False) -> dict:
    """Score a resume against JD using hybrid deterministic + LLM approach."""
    resume_text = _build_resume_text(parsed_resume)

    # ─── Deterministic Scoring ────────────────────────────────────────
    all_keywords = list(dict.fromkeys(
        parsed_jd.get("must_have_skills", [])
        + parsed_jd.get("good_to_have_skills", [])
        + parsed_jd.get("ats_keywords", [])
    ))
    weights = parsed_jd.get("skill_weights", {})

    keyword_result = compute_keyword_coverage(resume_text, all_keywords, weights)
    keyword_pct = keyword_result.get("weighted_score", keyword_result["coverage_pct"])

    format_result = check_ats_format(resume_text)
    format_score = format_result["score"]

    bullets = _collect_bullets(parsed_resume)
    bullet_results = check_bullet_quality(bullets)
    good_bullets = sum(1 for b in bullet_results if b["quality"] == "good")
    bullet_quality_pct = (good_bullets / len(bullet_results) * 100) if bullet_results else 50.0

    stuffing = detect_keyword_stuffing(resume_text, all_keywords)

    # ─── LLM Rubric Scoring ──────────────────────────────────────────
    llm_data = invoke_llm_json([
        {"role": "system", "content": _SCORE_SYSTEM},
        {"role": "user", "content": (
            f"## Resume\n\n{resume_text}\n\n"
            f"## JD Analysis\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Deterministic Results\n"
            f"Keyword Coverage: {keyword_pct:.1f}%\n"
            f"Covered: {', '.join(keyword_result['covered'])}\n"
            f"Missing: {', '.join(keyword_result['missing'])}\n"
            f"Format Score: {format_score}\n"
            f"Bullet Quality: {bullet_quality_pct:.1f}% good\n"
            f"Keyword Stuffing: {len(stuffing)} issues\n\n"
            "Provide semantic, section quality, and role alignment scores."
        )},
    ])

    semantic_score = llm_data.get("semantic_score", 50)
    section_quality = llm_data.get("section_quality_score", 50)

    # Truthfulness: 100 baseline (penalized later by truth guard)
    truthfulness = 100.0

    # ─── Composite Score ─────────────────────────────────────────────
    composite = compute_ats_score(
        keyword_coverage_pct=keyword_pct,
        semantic_score=semantic_score,
        section_quality_score=section_quality,
        ats_format_score=format_score,
        truthfulness_score=truthfulness,
    )

    result = {
        **composite,
        "keyword_coverage": keyword_result,
        "format_check": format_result,
        "bullet_quality": {
            "total": len(bullet_results),
            "good": good_bullets,
            "details": bullet_results,
        },
        "keyword_stuffing": stuffing,
        "semantic_score": semantic_score,
        "section_quality_score": section_quality,
        "role_alignment_score": llm_data.get("role_alignment_score", 50),
        "weak_sections": llm_data.get("weak_sections", []),
        "strong_sections": llm_data.get("strong_sections", []),
        "recommendations": llm_data.get("recommendations", []),
        "missing_high_weight_terms": llm_data.get("missing_high_weight_terms", []),
    }

    label = "Baseline" if is_baseline else "Final"
    logger.info("%s ATS Score: %s (keyword=%.1f, semantic=%s, format=%s).",
                label, composite["overall_score"], keyword_pct, semantic_score, format_score)

    return result


def baseline_score_node(state: ResumeGraphState) -> dict:
    """Node: score the original resume BEFORE optimization."""
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})

    result = score_resume(parsed_resume, parsed_jd, is_baseline=True)
    return {"baseline_score": result}


def final_score_node(state: ResumeGraphState) -> dict:
    """Node: score the optimized resume AFTER all changes."""
    draft = state.get("draft_resume", {})
    parsed_jd = state.get("parsed_jd", {})

    if not draft:
        logger.warning("Final Scorer: no draft resume to score.")
        return {"final_score": {"overall_score": 0}}

    # Apply truth guard penalty
    truth_report = state.get("truth_report", {})
    result = score_resume(draft, parsed_jd, is_baseline=False)

    # Penalize for truth violations
    high_violations = len([
        v for v in truth_report.get("violations", [])
        if v.get("severity") == "high"
    ])
    if high_violations > 0:
        penalty = min(20, high_violations * 5)
        result["overall_score"] = max(0, result["overall_score"] - penalty)
        result["breakdown"]["truthfulness"] = max(0, 100 - high_violations * 15)
        result["truth_penalty"] = penalty

    return {"final_score": result}
