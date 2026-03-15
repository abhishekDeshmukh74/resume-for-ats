"""Agent 2 — Analyse resume structure and identify gaps vs JD keywords."""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a resume analysis specialist.

Given a resume and a set of target JD keywords, you will:
1. Identify the resume's sections (summary, skills, each experience entry, education).
2. Map which keywords are ALREADY present in each section.
3. Identify which keywords are MISSING and suggest WHERE to place them.
4. For each missing keyword, suggest the SPECIFIC section and bullet point where it fits most naturally.

Return ONLY valid JSON:
{
  "sections": {
    "summary": "the exact summary text from the resume",
    "skills": "the exact skills text from the resume",
    "experience": [
      {
        "company": "company name",
        "title": "job title",
        "text": "the exact bullet points text for this role"
      }
    ],
    "education": "the exact education text"
  },
  "present_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["keyword3", "keyword4"],
  "gap_analysis": "Detailed analysis with SPECIFIC placement instructions for each missing keyword. Group by section. Example: SKILLS SECTION: Add Python, Docker, Kubernetes. SUMMARY: Add 'cloud architecture' and 'microservices'. EXPERIENCE bullet 1 at Company X: Add 'CI/CD' and 'agile'. Be as specific as possible about WHERE each keyword should go.",
  "placement_map": {
    "skills": ["keyword3", "keyword4"],
    "summary": ["keyword5"],
    "experience_bullets": {"Company X bullet 1": ["keyword6"]}
  }
}

RULES:
- The "text" fields must be EXACT verbatim copies from the resume.
- Be VERY specific about which missing keywords should go into which section/bullet.
- The skills section is the highest-impact place for technical keywords — prioritise it.
- Do NOT suggest adding keywords that don't make sense for the candidate's actual experience.
- Prioritise high-impact keywords (job title match, core technical skills, key methodologies).
- For the gap_analysis field, write actionable instructions the rewriter can follow directly."""


def analyse_resume(state: AgentState) -> dict:
    """Node: analyse resume sections and identify keyword gaps."""
    keywords_str = ", ".join(state.get("jd_keywords", []))

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Original Resume\n\n{state['resume_text']}\n\n"
            f"## Target Keywords\n\n{keywords_str}"
        )},
    ])
    sections = data.get("sections", {})
    gap = data.get("gap_analysis", "")
    missing = data.get("missing_keywords", [])

    logger.info("Resume analyser: %d present, %d missing keywords. Sections: %s",
                len(data.get("present_keywords", [])), len(missing),
                list(sections.keys()))

    return {
        "resume_sections": sections,
        "gap_analysis": gap,
    }
