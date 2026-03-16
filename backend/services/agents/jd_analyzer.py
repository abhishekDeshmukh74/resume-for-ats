"""Agent 2 — JD Analyzer Agent.

Parses the raw job description text and extracts structured hiring signals
that all downstream agents rely on.

Responsibilities:
    * Identify the target role family (e.g., "Full Stack", "Backend").
    * Separate must-have skills from nice-to-have skills.
    * Extract ATS keywords, domain keywords, and impact patterns.
    * Infer seniority level from language cues ("5+ years", "senior").
    * Assign importance weights (1–10) to each skill based on JD emphasis.
    * Normalise skill names for ATS searchability.

Graph position:
    ``parse_resume`` → **analyze_jd** → ``compute_gap``

State reads:
    ``raw_jd_text``

State writes:
    ``parsed_jd`` — dict with keys: ``target_role``, ``must_have_skills``,
    ``good_to_have_skills``, ``domain_keywords``, ``responsibilities``,
    ``seniority_signals``, ``ats_keywords``, ``impact_patterns``,
    ``skill_weights``.

Downstream consumers:
    * ``gap_analyzer``       — compares ``parsed_jd`` against ``parsed_resume``.
    * ``scorer``             — uses ``ats_keywords`` and ``skill_weights`` for
      keyword coverage scoring.
    * ``summary_optimizer``, ``skills_optimizer``, ``bullet_rewriter`` — use
      JD signals to guide ATS-aligned rewrites.
    * ``critic``             — uses JD context to evaluate keyword stuffing and
      role alignment.
"""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json, sanitize_input
from backend.services.agents.state import ResumeGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a job description analysis specialist for ATS optimisation.

Given a job description, extract ALL important hiring signals and structure them.

═══ RULES ═══

1. Extract specific technologies, frameworks, and methodologies mentioned.
2. Separate must-have skills from nice-to-have skills accurately.
3. Infer the role family (e.g., "Full Stack", "Backend", "Data Engineer").
4. Infer seniority level from language cues (e.g., "5+ years", "senior", "lead").
5. Collect ATS-relevant keywords that an ATS system would search for.
6. Identify action verbs and business impact themes.
7. Assign weights to skills (1-10 scale) based on emphasis in the JD.
   Skills mentioned in requirements get 8-10, in nice-to-have get 5-7,
   mentioned once in passing get 1-4.
8. Normalize skill names (e.g., "javascript" → "JavaScript").

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "target_role": "Senior Full Stack Engineer",
  "must_have_skills": ["React", "TypeScript", "Node.js"],
  "good_to_have_skills": ["AWS", "GraphQL", "LangGraph"],
  "domain_keywords": ["enterprise", "scalable systems", "collaboration"],
  "responsibilities": ["Design and implement frontend features", "..."],
  "seniority_signals": ["5+ years", "senior", "lead a team"],
  "ats_keywords": ["React", "TypeScript", "Node.js", "AWS", "CI/CD"],
  "impact_patterns": ["performance", "scalability", "ownership"],
  "skill_weights": {
    "React": 10,
    "TypeScript": 9,
    "Node.js": 8,
    "AWS": 7,
    "GraphQL": 5
  }
}"""


def analyze_jd_node(state: ResumeGraphState) -> dict:
    """LangGraph node: extract structured hiring signals from the job description.

    Sends the sanitised JD text to the LLM with a system prompt that defines
    the exact output schema.  Logs a summary of extracted signal counts.

    Args:
        state: Pipeline state; reads ``raw_jd_text``.

    Returns:
        ``{"parsed_jd": dict}`` with keys ``target_role``,
        ``must_have_skills``, ``good_to_have_skills``, ``domain_keywords``,
        ``responsibilities``, ``seniority_signals``, ``ats_keywords``,
        ``impact_patterns``, ``skill_weights``.
    """
    jd_text = sanitize_input(state["raw_jd_text"])

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"## Job Description\n\n{jd_text}"},
    ])

    must_have = data.get("must_have_skills", [])
    good_to_have = data.get("good_to_have_skills", [])
    weights = data.get("skill_weights", {})

    logger.info(
        "JD Analyzer: role=%s, %d must-have, %d nice-to-have, %d weighted skills.",
        data.get("target_role", "?"),
        len(must_have),
        len(good_to_have),
        len(weights),
    )

    return {"parsed_jd": data}
