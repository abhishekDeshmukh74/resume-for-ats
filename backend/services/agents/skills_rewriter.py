"""Section-specific Agent — Rewrite the skills section with comma-separated keywords.

This agent focuses ONLY on skills lines (e.g. "Frontend: React, Next.js, Redux").
It appends missing technical keywords in the same comma-separated format — never
as prose or sentences.
"""

from __future__ import annotations

import logging
import re

from backend.services.agents.llm import invoke_llm_json, _sanitize_user_input
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

# Matches "Category: skill1, skill2, ..." lines
_SKILLS_LINE_RE = re.compile(r"^[A-Za-z][^:\n]{0,40}:\s*.+,", re.MULTILINE)

_SYSTEM = """You are an ATS skills-section rewriter.

You receive:
- The EXACT skills section text from the resume
- Missing JD keywords that should be added to the skills section
- The keyword categories from the JD

Your ONLY job: add missing keywords to the existing skills lines as
comma-separated items.

═══ STRICT FORMAT RULES ═══

1. Output MUST be {"old": "...", "new": "..."} replacement pairs.
2. "old" must be a VERBATIM skills line from the resume.
3. "new" must keep the EXACT same format: "Category: Skill1, Skill2, Skill3"
4. ONLY append comma-separated keywords. NEVER add:
   - Sentences or phrases ("with expertise in...", "and proficiency in...")
   - Adjectives or qualifiers ("advanced Python", "expert-level React")
   - Trailing prose of any kind
5. Place each keyword in the most relevant category line.
6. If a keyword doesn't fit any existing category, append it to the closest match.
7. Do NOT create new category lines — only modify existing ones.
8. Do NOT remove any existing skills from the lines.

═══ EXAMPLES ═══

GOOD:
  old: "Frontend: React, Next.js, Redux, HTML, CSS"
  new: "Frontend: React, Next.js, Redux, HTML, CSS, TypeScript, Tailwind CSS"

BAD:
  old: "Frontend: React, Next.js, Redux, HTML, CSS"
  new: "Frontend: React, Next.js, Redux, HTML, CSS, with expertise in TypeScript"

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact skills line", "new": "same line with keywords appended"}
  ]
}"""


def rewrite_skills(state: AgentState) -> dict:
    """Node: add missing keywords to skills lines as comma-separated items."""
    sections = state.get("resume_sections", {})
    skills_text = sections.get("skills", "")
    missing = state.get("missing_keywords", [])
    categories = state.get("keyword_categories", {})
    required = state.get("required_keywords", [])

    if not skills_text or not missing:
        logger.info("Skills rewriter: no skills section or no missing keywords, skipping.")
        return {"raw_replacements": []}

    # Filter to technical keywords most suited for a skills section
    technical_cats = {"technical_skills", "tools_platforms", "domain_knowledge", "certifications"}
    technical_missing = []
    for kw in missing:
        for cat, kws in categories.items():
            if cat in technical_cats and kw in kws:
                technical_missing.append(kw)
                break
        else:
            # Keywords not in any category still get considered
            technical_missing.append(kw)

    if not technical_missing:
        logger.info("Skills rewriter: no technical keywords missing from skills section.")
        return {"raw_replacements": []}

    missing_required = [kw for kw in technical_missing if kw in required]
    missing_other = [kw for kw in technical_missing if kw not in required]

    priority_block = ""
    if missing_required:
        priority_block += f"\n🔴 REQUIRED: {', '.join(missing_required)}"
    if missing_other:
        priority_block += f"\n🟡 OTHER: {', '.join(missing_other)}"

    categories_block = "\n".join(
        f"- {cat}: {', '.join(kws)}" for cat, kws in categories.items()
    )

    sanitized_skills = _sanitize_user_input(skills_text)

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Skills Section (VERBATIM from resume)\n\n{sanitized_skills}\n\n"
            f"## JD Keyword Categories\n\n{categories_block}\n\n"
            f"## Missing Keywords to Add{priority_block}\n\n"
            "Add these keywords to the appropriate skills lines. "
            "Comma-separated ONLY — no sentences, no prose."
        )},
    ])

    raw = data.get("replacements", [])
    raw = [r for r in raw if r.get("old") and r.get("new") and r["old"] != r["new"]]

    # Programmatic safety: strip any trailing prose the LLM may have added
    _TRAILING_PROSE = re.compile(
        r",?\s+(?:with|and|including|such as|plus)\s+"
        r"(?:experience|expertise|proficiency|knowledge|"
        r"a focus|focus|emphasis|background|skills?\b)"
        r".+$",
        re.IGNORECASE,
    )
    for r in raw:
        cleaned = _TRAILING_PROSE.sub("", r["new"]).rstrip(",; ")
        if cleaned != r["new"]:
            logger.debug("Skills rewriter: stripped prose from: %r → %r", r["new"], cleaned)
            r["new"] = cleaned

    logger.info("Skills rewriter: produced %d replacements.", len(raw))
    return {"raw_replacements": raw}
