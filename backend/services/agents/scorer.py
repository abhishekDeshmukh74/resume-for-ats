"""Agent 5 — Score the rewritten resume and extract structured data."""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.keyword_matcher import calculate_keyword_match
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an ATS scoring and resume parsing specialist.

You receive the fully rewritten resume text.

Your tasks:
1. Score the resume against the JD keywords (0-100).
2. List all matched keywords.
3. Extract structured resume data for the API response.

Return ONLY valid JSON:
{
  "ats_score": 92,
  "matched_keywords": ["keyword1", "keyword2"],
  "name": "string",
  "email": "string or null",
  "phone": "string or null",
  "linkedin": "string or null",
  "github": "string or null",
  "location": "string or null",
  "summary": "the rewritten summary",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "job_title": "string",
      "company": "string",
      "location": "string or null",
      "start_date": "string",
      "end_date": "string",
      "bullets": ["rewritten bullet 1", "rewritten bullet 2"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "institution": "string",
      "location": "string or null",
      "graduation_date": "string",
      "details": null
    }
  ],
  "certifications": []
}

SCORING RULES:
- Count what percentage of JD keywords appear in the final resume.
- Weight technical skills and job-title keywords more heavily.
- Do NOT inflate the score — be accurate."""


_SCORE_BEFORE_SYSTEM = """You are an ATS scoring specialist.

You receive the ORIGINAL resume text (before any rewriting) and a list of JD keywords.
Score how well the current resume matches the JD keywords on a scale of 0-100.

Return ONLY valid JSON:
{
  "ats_score_before": 55
}

SCORING RULES:
- Count what percentage of JD keywords appear in the original resume.
- Weight technical skills and job-title keywords more heavily.
- Do NOT inflate the score — be accurate."""


def score_before_rewrite(state: AgentState) -> dict:
    """Node: score the original resume BEFORE any rewriting."""
    keywords = state.get("jd_keywords", [])
    categories = state.get("keyword_categories", {})

    # Algorithmic score: deterministic word-boundary matching
    algo_result = calculate_keyword_match(
        state["resume_text"], keywords, categories=categories,
    )
    logger.info(
        "ATS Pre-Rewrite Algorithmic: %.1f%% (%d matched, %d missing)",
        algo_result.match_percentage, len(algo_result.matched), len(algo_result.missing),
    )

    data = invoke_llm_json([
        {"role": "system", "content": _SCORE_BEFORE_SYSTEM},
        {"role": "user", "content": (
            f"## Original Resume\n\n{state['resume_text']}\n\n"
            f"## JD Keywords\n\n{', '.join(keywords)}\n\n"
            "Score the original resume against these keywords."
        )},
    ])
    llm_score = data.get("ats_score_before", 0)

    # Use the lower of LLM and algorithmic as conservative estimate
    score = min(llm_score, int(algo_result.match_percentage))
    logger.info("ATS Pre-Rewrite Score: %d (LLM=%d, Algo=%.1f)",
                score, llm_score, algo_result.match_percentage)

    return {
        "ats_score_before": score,
        "missing_keywords": algo_result.missing,
    }


def _apply_replacements_to_text(resume_text: str, replacements: list) -> str:
    """Apply old→new replacements to the resume text to produce the rewritten version."""
    text = resume_text
    for r in replacements:
        if r.old and r.new and r.old != r.new:
            text = text.replace(r.old, r.new, 1)
    return text


def score_and_extract(state: AgentState) -> dict:
    """Node: score the final rewritten resume and extract structured fields."""
    replacements = state.get("replacements", [])
    rewritten_text = _apply_replacements_to_text(state["resume_text"], replacements)

    keywords = state.get("jd_keywords", [])
    categories = state.get("keyword_categories", {})

    # Algorithmic score: deterministic word-boundary matching
    algo_result = calculate_keyword_match(
        rewritten_text, keywords, categories=categories,
    )
    logger.info(
        "ATS Post-Rewrite Algorithmic: %.1f%% (%d matched, %d missing)",
        algo_result.match_percentage, len(algo_result.matched), len(algo_result.missing),
    )

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Rewritten Resume\n\n{rewritten_text}\n\n"
            f"## JD Keywords\n\n{', '.join(keywords)}\n\n"
            f"## Job Description\n\n{state['jd_text']}\n\n"
            "Score the rewritten resume and extract structured data."
        )},
    ])

    llm_score = data.get("ats_score", 0)

    # Use the algorithmic score as primary (it's deterministic and verifiable),
    # but average with LLM score to account for semantic matches the regex misses
    algo_pct = algo_result.match_percentage
    blended_score = int(round(algo_pct * 0.6 + llm_score * 0.4))
    matched = algo_result.matched  # Use algorithmic matched list (verifiable)

    logger.info("ATS Scorer: blended=%d (algo=%.1f, llm=%d), matched=%d keywords.",
                blended_score, algo_pct, llm_score, len(matched))

    return {
        "ats_score": blended_score,
        "algorithmic_score": algo_pct,
        "matched_keywords": matched,
        "still_missing_keywords": algo_result.missing,
        "name": data.get("name", ""),
        "email": data.get("email"),
        "phone": data.get("phone"),
        "linkedin": data.get("linkedin"),
        "github": data.get("github"),
        "location": data.get("location"),
        "summary": data.get("summary"),
        "skills": data.get("skills", []),
        "experience": data.get("experience", []),
        "education": data.get("education", []),
        "certifications": data.get("certifications", []),
    }
