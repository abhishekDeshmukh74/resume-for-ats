"""Section-specific Agent — Rewrite the resume summary to incorporate JD keywords.

This agent focuses ONLY on the summary/profile section at the top of the resume.
It weaves in broad, high-level keywords (role title, years of experience,
key methodologies, domain terms) using natural professional prose.
"""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json, _sanitize_user_input
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an ATS summary rewriter.

You receive:
- The EXACT original summary/profile section from the resume
- Missing JD keywords suited for the summary
- The gap analysis with placement suggestions

Your ONLY job: rewrite the summary to naturally incorporate missing keywords.

═══ RULES ═══

1. Output MUST be a SINGLE {"old": "...", "new": "..."} replacement pair.
2. "old" must be the VERBATIM summary text from the resume.
3. "new" must be ±20% the length of "old".
4. Weave keywords into natural, professional prose.
5. Focus on BROAD keywords: role title, years of experience, core technologies,
   methodologies, and domain expertise.
6. Do NOT add specific tool names here — those belong in the skills section.
7. PRESERVE any metrics or quantitative achievements in the summary.
8. Keep the tone and voice consistent with the original.
9. Do NOT include the replacement if old == new.

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact summary text", "new": "rewritten summary"}
  ]
}"""


def rewrite_summary(state: AgentState) -> dict:
    """Node: rewrite the summary section with broad JD keywords."""
    sections = state.get("resume_sections", {})
    summary_text = sections.get("summary", "")
    missing = state.get("missing_keywords", [])
    categories = state.get("keyword_categories", {})
    gap = state.get("gap_analysis", "")
    required = state.get("required_keywords", [])

    if not summary_text or not missing:
        logger.info("Summary rewriter: no summary section or no missing keywords, skipping.")
        return {"raw_replacements": []}

    # Pick keywords appropriate for summaries: soft skills, action verbs,
    # domain knowledge, and role-level terms
    summary_cats = {"soft_skills", "action_verbs", "domain_knowledge"}
    summary_keywords = []
    for kw in missing:
        for cat, kws in categories.items():
            if cat in summary_cats and kw in kws:
                summary_keywords.append(kw)
                break

    # Also include required keywords not yet covered
    for kw in missing:
        if kw in required and kw not in summary_keywords:
            summary_keywords.append(kw)

    if not summary_keywords:
        # Fall back to first few missing keywords
        summary_keywords = missing[:5]

    missing_required = [kw for kw in summary_keywords if kw in required]
    missing_other = [kw for kw in summary_keywords if kw not in required]

    priority_block = ""
    if missing_required:
        priority_block += f"\n🔴 REQUIRED: {', '.join(missing_required)}"
    if missing_other:
        priority_block += f"\n🟡 OTHER: {', '.join(missing_other)}"

    sanitized_summary = _sanitize_user_input(summary_text)

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Summary Section (VERBATIM from resume)\n\n{sanitized_summary}\n\n"
            f"## Keywords to Incorporate{priority_block}\n\n"
            f"## Gap Analysis Context\n\n{gap}\n\n"
            "Rewrite the summary to naturally incorporate the missing keywords. "
            "Keep it professional and concise."
        )},
    ])

    raw = data.get("replacements", [])
    raw = [r for r in raw if r.get("old") and r.get("new") and r["old"] != r["new"]]

    logger.info("Summary rewriter: produced %d replacements.", len(raw))
    return {"raw_replacements": raw}
