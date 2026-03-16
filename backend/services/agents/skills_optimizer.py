"""Agent 7 — Skills Section Optimizer Agent.

ATS often parses the skills section heavily. This agent:
  - Normalizes skill naming
  - Dedupes variants
  - Aligns terminology with JD
  - Organizes into meaningful groups

Rules: only include proven skills, normalize for ATS, reflect JD wording.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState
from backend.services.agents.tools import normalize_skill_names

logger = logging.getLogger(__name__)

_SYSTEM = """You are a skills section optimization specialist for ATS systems.

You receive the candidate's parsed skills, JD signals, and gap analysis.
Optimize the skills section for maximum ATS parsability.

═══ RULES ═══

1. ONLY include skills the candidate actually has (present in their resume).
2. Normalize naming for ATS searchability:
   - "JS" → "JavaScript"
   - "ReactJS" → "React"
   - "k8s" → "Kubernetes"
3. Deduplicate variants (don't list both "JavaScript" and "JS").
4. Use JD terminology when the candidate has the skill:
   - If JD says "TypeScript" and resume says "TS", use "TypeScript".
5. Organize into meaningful groups that ATS systems parse well.
6. Order skills within each group by relevance to the JD (most relevant first).
7. Do NOT add skills the candidate doesn't have.
8. Keep the section clean — no prose, no sentences, just skill names.

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "skills": {
    "languages": ["JavaScript", "TypeScript", "Python"],
    "frontend": ["React", "Next.js", "Tailwind CSS"],
    "backend": ["Node.js", "Express.js", "PostgreSQL"],
    "cloud": ["AWS", "Docker", "Kubernetes"],
    "ai": ["LangChain", "OpenAI API"],
    "tools": ["Git", "Jira", "CI/CD"]
  },
  "added_from_resume": ["skills already in resume but reorganized"],
  "removed_duplicates": ["JS (kept JavaScript)", "ReactJS (kept React)"],
  "jd_aligned": ["TypeScript (was TS)", "Kubernetes (was k8s)"]
}"""


def optimize_skills_node(state: ResumeGraphState) -> dict:
    """Node: optimize skills section for ATS alignment."""
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})
    gap_report = state.get("gap_report", {})

    current_skills = parsed_resume.get("skills", {})

    # Pre-normalize with deterministic tool
    all_skills = []
    if isinstance(current_skills, dict):
        for category_skills in current_skills.values():
            if isinstance(category_skills, list):
                all_skills.extend(category_skills)
    elif isinstance(current_skills, list):
        all_skills = current_skills

    normalized = normalize_skill_names(all_skills)

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Current Skills (pre-normalized)\n\n{json.dumps(normalized)}\n\n"
            f"## Original Skills Structure\n\n{json.dumps(current_skills, indent=2)}\n\n"
            f"## JD Signals\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Gap Report\n\n"
            f"Covered keywords: {', '.join(gap_report.get('covered_keywords', []))}\n"
            f"Underrepresented: {', '.join(gap_report.get('underrepresented_keywords', []))}\n"
            f"Cannot claim: {', '.join(gap_report.get('cannot_claim', []))}\n\n"
            "Optimize the skills section. Only include PROVEN skills."
        )},
    ])

    skills = data.get("skills", current_skills)

    # Post-process: apply deterministic normalization to each category
    if isinstance(skills, dict):
        for category, items in skills.items():
            if isinstance(items, list):
                skills[category] = normalize_skill_names(items)

    logger.info("Skills Optimizer: %d categories, %d total skills.",
                len(skills) if isinstance(skills, dict) else 0,
                sum(len(v) for v in skills.values()) if isinstance(skills, dict) else 0)

    return {"optimized_skills": skills}
