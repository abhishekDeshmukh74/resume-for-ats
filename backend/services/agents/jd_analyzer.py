"""Agent 2 — JD Analyzer Agent.

Parses the job description and extracts structured hiring signals:
required skills, nice-to-have skills, domain keywords, seniority signals,
ATS keywords, impact patterns, and skill weights.
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
    """Node: parse JD into structured hiring signals."""
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
