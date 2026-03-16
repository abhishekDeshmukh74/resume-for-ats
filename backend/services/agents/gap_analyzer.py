"""Agent 4 — Gap Analysis Agent.

Compares parsed resume against parsed JD to identify:
  - Keywords already covered
  - Keywords that exist but are under-expressed
  - Keywords that are truly missing and cannot be claimed
  - Reframe opportunities (truthful rephrasing)

This agent is brutally honest — not everything missing should be forced in.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState
from backend.services.agents.tools import compute_keyword_coverage

logger = logging.getLogger(__name__)

_SYSTEM = """You are a resume gap analysis specialist.

You receive a parsed resume (JSON) and a parsed job description (JSON).
Your job is to compare them brutally honestly.

═══ RULES ═══

1. Identify which JD keywords are ALREADY covered in the resume.
2. Identify keywords that EXIST in the resume but are under-expressed
   (e.g., mentioned once in passing when they should be prominent).
3. Identify keywords that are genuinely MISSING and CANNOT be truthfully claimed.
4. Identify reframe opportunities: existing experience that can be rephrased
   to better match JD language, without lying.
5. Be honest — if a skill is truly missing, say so. Do not suggest faking it.
6. Some missing things should be:
   - Omitted entirely
   - Marked as actual gaps
   - Recommended for cover letter mention instead

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "covered_keywords": ["React", "Node.js", "REST APIs"],
  "underrepresented_keywords": ["TypeScript", "performance optimization"],
  "missing_keywords": ["Prisma", "Azure DevOps"],
  "cannot_claim": ["Prisma", "Azure DevOps"],
  "reframe_opportunities": [
    {
      "original": "Existing React work description",
      "suggestion": "Can be phrased as scalable frontend component development",
      "target_keyword": "scalable systems"
    }
  ],
  "cover_letter_suggestions": ["Prisma — mention willingness to learn"],
  "priority_additions": [
    {
      "keyword": "TypeScript",
      "where": "skills section and experience bullets",
      "reason": "Already used but not prominently mentioned"
    }
  ]
}"""


def compute_gap_node(state: ResumeGraphState) -> dict:
    """Node: compare parsed resume against parsed JD for gaps."""
    parsed_resume = state.get("parsed_resume", {})
    parsed_jd = state.get("parsed_jd", {})

    # Deterministic keyword coverage as input signal
    all_jd_keywords = (
        parsed_jd.get("must_have_skills", [])
        + parsed_jd.get("good_to_have_skills", [])
        + parsed_jd.get("ats_keywords", [])
    )
    # Dedupe
    all_jd_keywords = list(dict.fromkeys(all_jd_keywords))

    # Build full resume text from parsed structure for coverage check
    resume_text_parts = []
    if parsed_resume.get("summary"):
        resume_text_parts.append(parsed_resume["summary"])
    skills = parsed_resume.get("skills", {})
    if isinstance(skills, dict):
        for category_skills in skills.values():
            if isinstance(category_skills, list):
                resume_text_parts.extend(category_skills)
    for exp in parsed_resume.get("experience", []):
        resume_text_parts.extend(exp.get("bullets", []))
    for proj in parsed_resume.get("projects", []):
        resume_text_parts.extend(proj.get("bullets", []))
        resume_text_parts.extend(proj.get("stack", []))

    resume_text_blob = " ".join(resume_text_parts)
    coverage = compute_keyword_coverage(
        resume_text_blob,
        all_jd_keywords,
        weights=parsed_jd.get("skill_weights"),
    )

    # LLM-based nuanced gap analysis
    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Parsed Resume\n\n{json.dumps(parsed_resume, indent=2)}\n\n"
            f"## Parsed JD\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Deterministic Keyword Coverage\n\n"
            f"Covered: {', '.join(coverage['covered'])}\n"
            f"Missing: {', '.join(coverage['missing'])}\n"
            f"Coverage: {coverage['coverage_pct']}%\n\n"
            "Provide your detailed gap analysis."
        )},
    ])

    # Merge deterministic coverage into the gap report
    data["deterministic_coverage"] = coverage

    logger.info(
        "Gap Analysis: %d covered, %d underrepresented, %d missing, %d reframe opportunities.",
        len(data.get("covered_keywords", [])),
        len(data.get("underrepresented_keywords", [])),
        len(data.get("missing_keywords", [])),
        len(data.get("reframe_opportunities", [])),
    )

    return {"gap_report": data}
