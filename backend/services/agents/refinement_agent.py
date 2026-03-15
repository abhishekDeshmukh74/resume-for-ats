"""Refinement Agent — Targeted second-pass keyword injection for missing keywords.

This agent runs only when the first rewrite pass scores below the target (90).
It focuses exclusively on injecting the still-missing keywords into the resume
text, working with the ALREADY-rewritten text (i.e., after replacements are applied).
"""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an ATS keyword injection specialist.

The resume has already been rewritten once, but some JD keywords are STILL MISSING.
Your ONLY job is to inject the remaining missing keywords into the resume text.

You receive:
- The resume text WITH first-pass replacements already applied
- The STILL-MISSING keywords that MUST be added
- Which of those are REQUIRED (highest priority)

═══ YOUR TASK ═══

Generate additional {"old": "...", "new": "..."} replacement pairs that inject
the still-missing keywords. Focus on:

1. SKILLS SECTION — The easiest and most impactful place to add missing technical keywords.
   Find the existing skills/technologies line and add the missing ones.

2. SUMMARY — Weave in 2-3 broad missing keywords (role title, methodology, domain).

3. EXPERIENCE BULLETS — Pick the most relevant bullet for each missing keyword and
   naturally incorporate it.

═══ RULES ═══

1. "old" MUST be VERBATIM text from the CURRENT resume (after first-pass rewrites).
2. "new" must be ±20% length of "old".
3. PRESERVE ALL METRICS AND NUMBERS — never remove quantitative impact.
4. Do NOT re-add keywords that are already present — only focus on MISSING ones.
5. Maintain natural language — no keyword stuffing.
6. Each keyword should appear at MOST 2 times total.
7. Do NOT change section headers, job titles, company names, dates.
8. Do NOT include replacements where old == new.

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact text from resume", "new": "text with missing keyword injected"}
  ]
}"""


def _apply_replacements_to_text(resume_text: str, replacements: list) -> str:
    """Apply old→new replacements to produce current resume text."""
    text = resume_text
    for r in replacements:
        old = r.old if hasattr(r, "old") else r.get("old", "")
        new = r.new if hasattr(r, "new") else r.get("new", "")
        if old and new and old != new:
            text = text.replace(old, new, 1)
    return text


def refine_rewrite(state: AgentState) -> dict:
    """Node: targeted second-pass to inject still-missing keywords."""
    replacements = state.get("replacements", [])
    still_missing = state.get("still_missing_keywords", [])
    required = state.get("required_keywords", [])
    categories = state.get("keyword_categories", {})

    if not still_missing:
        logger.info("Refinement: no missing keywords, skipping.")
        return {"rewrite_pass": 1}

    # Build the current resume text with first-pass replacements applied
    current_text = _apply_replacements_to_text(state["resume_text"], replacements)

    # Separate missing keywords by priority
    missing_required = [kw for kw in still_missing if kw in required]
    missing_other = [kw for kw in still_missing if kw not in required]

    keywords_block = "\n".join(
        f"- {cat}: {', '.join(kws)}"
        for cat, kws in categories.items()
    )

    priority_block = ""
    if missing_required:
        priority_block += f"\n🔴 REQUIRED (must add): {', '.join(missing_required)}"
    if missing_other:
        priority_block += f"\n🟡 OTHER (should add): {', '.join(missing_other)}"

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Current Resume (after first rewrite pass)\n\n{current_text}\n\n"
            f"## All JD Keywords by Category\n\n{keywords_block}\n\n"
            f"## STILL-MISSING Keywords (inject these){priority_block}\n\n"
            f"Total missing: {len(still_missing)} keywords.\n"
            f"Generate replacements to inject as many as possible. "
            f"Focus especially on skills lines and summary.\n"
            "Return only the JSON object."
        )},
    ])

    raw = data.get("replacements", [])
    raw = [r for r in raw if r.get("old") and r.get("new") and r["old"] != r["new"]]

    logger.info("Refinement agent: produced %d additional replacements for %d missing keywords.",
                len(raw), len(still_missing))

    return {
        "raw_replacements": raw,
        "rewrite_pass": 1,
    }
