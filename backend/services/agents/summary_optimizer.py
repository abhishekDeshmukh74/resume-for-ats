"""Agent 6 — Summary Optimizer Agent.

Creates a sharp summary section matching:
  - Target role
  - Years of experience
  - Strongest matching skills
  - Domain alignment

Rules: no buzzword salad, no generic phrases, include top 4-6 match signals.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert resume summary writer for ATS optimisation.

You receive the parsed resume, JD signals, and gap analysis. Write a sharp,
focused professional summary.

═══ STRUCTURE ═══

1. Years of experience + role alignment
2. Strongest matching tech stack (top 3-4)
3. Domain / problem types solved
4. Role alignment statement

═══ HARD RULES ═══

1. NO buzzword salad. No "results-driven professional" or "dynamic leader".
2. NO generic filler. Every word must convey specific information.
3. Include the top 4-6 strongest match signals from the JD.
4. Keep it 2-4 sentences maximum.
5. NEVER claim experience the candidate doesn't have.
6. Use the candidate's ACTUAL years of experience (count from their earliest role).
7. Match the target role title if the candidate's experience supports it.
8. Include domain keywords from the JD naturally.
9. Write in third person without pronouns (no "I", "he", "she").

═══ EXAMPLE ═══

"Full Stack Engineer with 6+ years building scalable web applications using
React, TypeScript, and Node.js. Experienced in designing real-time
collaboration systems, REST API architecture, and cloud deployments on AWS.
Focused on performance optimization and developer tooling for enterprise
engineering teams."

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "summary": "The optimized summary text",
  "signals_used": ["React", "TypeScript", "6+ years", "scalable systems"],
  "explanation": "Why these signals were chosen and how they align with the JD"
}"""


def optimize_summary_node(state: ResumeGraphState) -> dict:
    """Node: write an ATS-optimized summary section."""
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})
    gap_report = state.get("gap_report", {})

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Parsed Resume\n\n{json.dumps(parsed_resume, indent=2)}\n\n"
            f"## JD Signals\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Gap Report\n\n"
            f"Covered: {', '.join(gap_report.get('covered_keywords', []))}\n"
            f"Underrepresented: {', '.join(gap_report.get('underrepresented_keywords', []))}\n"
            f"Reframe opportunities: {json.dumps(gap_report.get('reframe_opportunities', []), indent=2)}\n\n"
            "Write an optimized summary. Follow the rules strictly."
        )},
    ])

    summary = data.get("summary", "")
    logger.info("Summary Optimizer: produced %d-char summary with %d signals.",
                len(summary), len(data.get("signals_used", [])))

    return {"optimized_summary": summary}
