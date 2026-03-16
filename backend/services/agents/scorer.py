"""Agent 8 — ATS Scoring Agent (hybrid deterministic + LLM).

This agent is called twice:
    1. **Baseline score** (``baseline_score_node``) — scores the *original*
       parsed resume before any optimisation.
    2. **Final score** (``final_score_node``) — scores the *optimised* draft
       after all changes and truth verification.

Scoring approach (hybrid):
    Pure LLM scoring is unreliable because LLMs tend to produce optimistic
    vibe-checks.  Instead, the score is composed from:

    * **Deterministic keyword coverage** (``compute_keyword_coverage``) —
      exact + fuzzy match against JD keywords, optionally weighted.
    * **Deterministic format check** (``check_ats_format``) — heading
      presence, box characters, date formatting, contact info.
    * **Deterministic bullet quality** (``check_bullet_quality``) — strong
      verb usage, metric presence, length checks.
    * **Deterministic stuffing check** (``detect_keyword_stuffing``) —
      flagged in the report for downstream agents.
    * **LLM semantic score** — how well the resume’s narrative aligns with
      the JD beyond keyword presence.
    * **LLM section quality** — clarity, impact, and alignment of each
      section.

    Composite formula (``compute_ats_score``)::

        overall = (0.30 × keyword_coverage
                 + 0.25 × semantic_alignment
                 + 0.20 × section_quality
                 + 0.15 × ats_format
                 + 0.10 × truthfulness)

    Truthfulness starts at 100 and is penalised in the final score based on
    ``truth_report.violations`` (5 points per high-severity violation,
    capped at 20).

Graph position (baseline):
    ``compute_gap`` → **baseline_score** → ``optimize_summary``

Graph position (final):
    ``critic`` [passed] → **final_score** → ``export``

State reads (baseline):
    ``parsed_resume``, ``parsed_jd``

State reads (final):
    ``draft_resume``, ``parsed_jd``, ``truth_report``

State writes:
    ``baseline_score`` or ``final_score`` — dict with ``overall_score``,
    ``breakdown``, ``keyword_coverage``, ``format_check``, ``bullet_quality``,
    ``keyword_stuffing``, ``semantic_score``, ``section_quality_score``,
    ``role_alignment_score``, ``weak_sections``, ``strong_sections``,
    ``recommendations``, ``missing_high_weight_terms``.
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
    """Reconstruct human-readable text from a parsed resume dict.

    Used to produce a text blob that the deterministic tools and the LLM
    scorer can evaluate.  Includes name, summary, skills (categorised),
    experience bullets, project bullets, and education.

    Args:
        parsed_resume: Structured resume dict (from ``intake_parser`` or
                       ``draft_resume`` after merging).

    Returns:
        Multi-line string suitable for keyword matching and LLM evaluation.
    """
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
    """Extract all bullet points from experience and projects sections.

    Args:
        parsed_resume: Structured resume dict.

    Returns:
        Flat list of bullet strings for quality analysis.
    """
    bullets: list[str] = []
    for exp in parsed_resume.get("experience", []):
        bullets.extend(exp.get("bullets", []))
    for proj in parsed_resume.get("projects", []):
        bullets.extend(proj.get("bullets", []))
    return bullets


def score_resume(parsed_resume: dict, parsed_jd: dict, is_baseline: bool = False) -> dict:
    """Score a resume against a JD using the hybrid deterministic + LLM approach.

    This is the shared scoring function called by both ``baseline_score_node``
    and ``final_score_node``.  It runs all deterministic tools first, then
    passes their results to the LLM for semantic and quality evaluation,
    and finally computes the weighted composite score.

    Args:
        parsed_resume: Structured resume dict to score.
        parsed_jd:     Structured JD analysis dict.
        is_baseline:   If ``True``, labels log output as "Baseline";
                       otherwise "Final".

    Returns:
        Comprehensive score dict with ``overall_score``, ``breakdown``,
        ``keyword_coverage``, ``format_check``, ``bullet_quality``,
        ``keyword_stuffing``, ``semantic_score``, ``section_quality_score``,
        ``role_alignment_score``, ``weak_sections``, ``strong_sections``,
        ``recommendations``, ``missing_high_weight_terms``.
    """
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
    """LangGraph node: score the *original* resume BEFORE optimisation.

    Reads the raw parsed resume (not yet optimised) and the JD analysis,
    then runs ``score_resume(is_baseline=True)``.

    The baseline score is:
        * Displayed to the user as the "before" score.
        * Available to the critic agent as context for evaluating improvement.
        * Stored in the pipeline run for analytics.

    Args:
        state: Pipeline state; reads ``parsed_resume``, ``parsed_jd``.

    Returns:
        ``{"baseline_score": dict}`` with the full score breakdown.
    """
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})

    result = score_resume(parsed_resume, parsed_jd, is_baseline=True)
    return {"baseline_score": result}


def final_score_node(state: ResumeGraphState) -> dict:
    """LangGraph node: score the *optimised* resume AFTER all changes.

    Reads the merged draft resume and applies ``score_resume(is_baseline=False)``.
    Additionally applies a truthfulness penalty based on the truth guard report:
    −5 points per high-severity violation (capped at −20).

    The final score is:
        * Displayed to the user as the "after" score.
        * Returned in the API response as ``ats_score``.
        * Stored in the pipeline run for analytics.

    Args:
        state: Pipeline state; reads ``draft_resume``, ``parsed_jd``,
               ``truth_report``.

    Returns:
        ``{"final_score": dict}`` with the full score breakdown, plus
        ``truth_penalty`` if any violations were found.
    """
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
