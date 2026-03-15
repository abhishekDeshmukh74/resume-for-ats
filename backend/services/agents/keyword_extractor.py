"""Agent 1 — Extract and categorise keywords from the job description."""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a keyword extraction specialist for ATS (Applicant Tracking System) optimisation.

Given a job description, extract ALL important keywords and categorise them.

Return ONLY valid JSON:
{
  "keywords": ["keyword1", "keyword2", ...],
  "categories": {
    "technical_skills": ["Python", "React", ...],
    "soft_skills": ["leadership", "collaboration", ...],
    "tools_platforms": ["AWS", "Docker", ...],
    "domain_knowledge": ["microservices", "CI/CD", ...],
    "certifications": ["AWS Certified", ...],
    "action_verbs": ["architected", "optimised", ...]
  },
  "required_skills": ["Python", "React", "AWS"],
  "preferred_skills": ["Kubernetes", "Terraform"]
}

RULES:
- Extract 30-60 unique keywords/phrases.
- Include specific technologies, frameworks, methodologies mentioned.
- Include implied skills (e.g. if "full stack" is mentioned, include both frontend and backend terms).
- Normalise casing (e.g. "javascript" → "JavaScript").
- Do NOT duplicate keywords across categories.
- Include important action verbs from the JD.
- "required_skills" = skills explicitly stated as required, mandatory, or must-have, plus any
  skills mentioned multiple times or in core responsibilities. These are the highest priority.
- "preferred_skills" = skills stated as preferred, nice-to-have, bonus, or mentioned only once
  in a secondary context.
- Every keyword in required_skills and preferred_skills MUST also appear in the main "keywords" list.
- If the JD doesn't clearly distinguish required vs preferred, treat skills in the top
  responsibilities and qualifications as required, and everything else as preferred."""


def extract_keywords(state: AgentState) -> dict:
    """Node: extract JD keywords and categories."""
    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"## Job Description\n\n{state['jd_text']}"},
    ])
    keywords = list(set(data.get("keywords", [])))
    categories = data.get("categories", {})
    required = data.get("required_skills", [])
    preferred = data.get("preferred_skills", [])

    logger.info("Keyword extractor: %d unique keywords in %d categories (%d required, %d preferred).",
                len(keywords), len(categories), len(required), len(preferred))

    return {
        "jd_keywords": keywords,
        "keyword_categories": categories,
        "required_keywords": required,
        "preferred_keywords": preferred,
    }
