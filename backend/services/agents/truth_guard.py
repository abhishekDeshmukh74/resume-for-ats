"""Agent 3 — Resume Truth Guard Agent.

The most important safety agent. Checks whether rewritten content is
supported by the source resume text.

Rejects: invented tools, fake metrics, fake ownership claims,
fake domain experience, fake titles.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState
from backend.services.agents.tools import check_unsupported_claims

logger = logging.getLogger(__name__)

_SYSTEM = """You are a resume truth verification specialist.

You receive:
- The ORIGINAL parsed resume (source of truth)
- The OPTIMIZED draft (summary, skills, experience bullets)

Your job: verify that EVERY claim in the optimized version is supported by
the original resume.

═══ WHAT TO CHECK ═══

1. TOOLS/TECHNOLOGIES: Every technology mentioned in the optimized version
   must appear (or be clearly implied) in the original resume.
2. METRICS: No invented numbers, percentages, or statistics.
3. OWNERSHIP CLAIMS: If the optimized version says "led" or "architected",
   the original must support that level of ownership.
4. DOMAIN EXPERIENCE: No fake domain claims.
5. JOB TITLES: Must match the original exactly.
6. COMPANY NAMES: Must match the original exactly.
7. DATES: Must match the original exactly.
8. SKILLS: Only skills present in or strongly implied by the original.

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "supported": true/false,
  "violations": [
    {
      "type": "unsupported_skill|fake_metric|fake_ownership|fake_domain|wrong_title|invented_tool",
      "value": "the specific claim",
      "location": "summary|skills|experience bullet at Company X",
      "reason": "Why this is not supported by the original resume",
      "severity": "high|medium|low"
    }
  ],
  "warnings": [
    {
      "type": "borderline_claim",
      "value": "the claim",
      "reason": "This is technically supportable but stretched"
    }
  ],
  "summary": "Brief overall assessment of truthfulness"
}"""


def truth_guard_node(state: ResumeGraphState) -> dict:
    """Node: verify optimized content against original resume."""
    parsed_resume = state.get("parsed_resume", {})
    draft = state.get("draft_resume", {})

    if not draft:
        logger.warning("Truth Guard: no draft resume to check.")
        return {"truth_report": {"supported": True, "violations": [], "warnings": []}}

    # Deterministic skill check first
    original_skills = []
    skills_data = parsed_resume.get("skills", {})
    if isinstance(skills_data, dict):
        for v in skills_data.values():
            if isinstance(v, list):
                original_skills.extend(v)
    elif isinstance(skills_data, list):
        original_skills = skills_data

    draft_skills = []
    draft_skills_data = draft.get("skills", {})
    if isinstance(draft_skills_data, dict):
        for v in draft_skills_data.values():
            if isinstance(v, list):
                draft_skills.extend(v)
    elif isinstance(draft_skills_data, list):
        draft_skills = draft_skills_data

    # Build original text for comprehensive check
    original_text_parts = [parsed_resume.get("summary", "")]
    original_text_parts.extend(original_skills)
    for exp in parsed_resume.get("experience", []):
        original_text_parts.extend(exp.get("bullets", []))
    for proj in parsed_resume.get("projects", []):
        original_text_parts.extend(proj.get("bullets", []))
        original_text_parts.extend(proj.get("stack", []))
    original_text = " ".join(str(p) for p in original_text_parts if p)

    deterministic_violations = check_unsupported_claims(
        original_text, draft_skills, original_skills,
    )

    # LLM-based deep truth check
    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Original Parsed Resume\n\n{json.dumps(parsed_resume, indent=2)}\n\n"
            f"## Optimized Draft Resume\n\n{json.dumps(draft, indent=2)}\n\n"
            f"## Deterministic Skill Violations\n\n{json.dumps(deterministic_violations, indent=2)}\n\n"
            "Verify ALL claims in the optimized draft against the original. "
            "Be thorough and skeptical."
        )},
    ])

    # Merge deterministic violations into LLM findings
    llm_violations = data.get("violations", [])
    existing_values = {v.get("value", "").lower() for v in llm_violations}
    for dv in deterministic_violations:
        if dv["value"].lower() not in existing_values:
            llm_violations.append({
                "type": dv["type"],
                "value": dv["value"],
                "location": "skills",
                "reason": dv["reason"],
                "severity": "high",
            })

    data["violations"] = llm_violations
    data["supported"] = len([v for v in llm_violations if v.get("severity") == "high"]) == 0

    logger.info(
        "Truth Guard: supported=%s, %d violations (%d high severity), %d warnings.",
        data["supported"],
        len(llm_violations),
        len([v for v in llm_violations if v.get("severity") == "high"]),
        len(data.get("warnings", [])),
    )

    return {"truth_report": data}
