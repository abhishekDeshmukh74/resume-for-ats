"""Agent 5 — Bullet Rewriter Agent.

Rewrites experience and project bullets so they are:
  - ATS keyword aligned
  - Achievement-based
  - Concise
  - Metric-driven (if metrics already exist)
  - Clearer and more role-aligned

Pattern: Action + Tech + Scope + Result

Rules: keep facts intact, do not add fake numbers.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert ATS bullet point rewriter.

You receive experience/project bullets from a resume, along with gap analysis
and JD signals. Rewrite the bullets to maximize ATS alignment while keeping
facts intact.

═══ PATTERN ═══

Action + Tech + Scope + Result

Example bad:
  "Worked on internal AI tool using React and Python"

Example better:
  "Built internal GenAI productivity tool using React and Python services,
   improving enterprise knowledge retrieval workflows across multiple
   internal data sources"

═══ HARD RULES ═══

1. NEVER invent facts, metrics, companies, or tools not in the original.
2. NEVER add fake numbers or percentages.
3. PRESERVE all existing metrics and numbers exactly.
4. If the original bullet has a metric, keep it. If it doesn't, do NOT add one.
5. Use strong verbs: built, created, designed, developed, improved, led,
   reduced, increased, automated, deployed, integrated, migrated, optimized.
6. AVOID AI-sounding words: spearheaded, orchestrated, synergized, leveraged,
   revolutionized, utilized.
7. Integrate JD keywords NATURALLY into the core sentence.
   Do NOT tack keywords onto the end with connectors like:
   "with focus on", "with expertise in", "ensuring", "leveraging".
8. Each keyword should appear at MOST 2 times across all bullets.
9. Spread keywords EVENLY across bullets — don't stuff one bullet.
10. Keep bullets concise — 1 to 2 lines maximum.
11. For each bullet, explain WHY the rewrite improves ATS alignment.

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "experience": [
    {
      "company": "Company Name",
      "title": "Job Title",
      "start": "date",
      "end": "date",
      "bullets": [
        {
          "original": "exact original bullet text",
          "rewritten": "ATS-optimized rewrite",
          "explanation": "Why this rewrite improves ATS alignment"
        }
      ]
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "bullets": [
        {
          "original": "exact original bullet",
          "rewritten": "rewritten bullet",
          "explanation": "reason"
        }
      ]
    }
  ]
}"""


def optimize_experience_node(state: ResumeGraphState) -> dict:
    """Node: rewrite experience and project bullets for ATS optimization."""
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})
    gap_report = state.get("gap_report", {})

    experience = parsed_resume.get("experience", [])
    projects = parsed_resume.get("projects", [])

    if not experience and not projects:
        logger.info("Bullet Rewriter: no experience or projects to rewrite.")
        return {"optimized_experience": []}

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Current Experience\n\n{json.dumps(experience, indent=2)}\n\n"
            f"## Current Projects\n\n{json.dumps(projects, indent=2)}\n\n"
            f"## JD Signals\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Gap Report\n\n"
            f"Underrepresented: {', '.join(gap_report.get('underrepresented_keywords', []))}\n"
            f"Reframe opportunities: {json.dumps(gap_report.get('reframe_opportunities', []), indent=2)}\n"
            f"Priority additions: {json.dumps(gap_report.get('priority_additions', []), indent=2)}\n\n"
            "Rewrite all bullets following the rules. Include explanations."
        )},
    ])

    optimized_exp = data.get("experience", [])
    optimized_proj = data.get("projects", [])

    # Flatten the rewritten bullets for downstream use
    result = {
        "experience": optimized_exp,
        "projects": optimized_proj,
    }

    total_bullets = sum(
        len(e.get("bullets", [])) for e in optimized_exp
    ) + sum(
        len(p.get("bullets", [])) for p in optimized_proj
    )

    logger.info("Bullet Rewriter: rewrote %d total bullets.", total_bullets)

    return {"optimized_experience": result}
