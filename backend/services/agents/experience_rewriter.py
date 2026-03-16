"""Section-specific Agent — Rewrite experience bullets to incorporate JD keywords.

This agent focuses ONLY on experience/work-history bullet points.  It rewrites
individual bullets to weave in missing keywords while strictly preserving all
metrics, numbers, and quantitative impact.
"""

from __future__ import annotations

import logging
import re

from backend.services.agents.llm import invoke_llm_json, _sanitize_user_input
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an ATS experience-section rewriter.

You receive:
- The EXACT experience section text from the resume (each role with its bullets)
- Missing JD keywords that should be woven into experience bullets
- The gap analysis with placement suggestions

Your ONLY job: rewrite experience bullets to incorporate missing keywords.

═══ RULES ═══

1. Output MUST be {"old": "...", "new": "..."} replacement pairs.
2. "old" must be a VERBATIM bullet point — the actual bullet content describing
   work done. NEVER use lines that are job headers (company name, job title,
   or date range). A job header line contains a date range like
   "January 2023 – Present" or "Dec 2022 – Mar 2024" — these must be skipped.
   NEVER use bare section headers like "Experience" or "Work History" as "old".
   Each "old" must start with an action verb or bullet marker.
3. "new" must be ±20% the length of "old".
4. ONE replacement per bullet point. Do NOT combine multiple bullets.

5. PRESERVE ALL METRICS AND NUMBERS:
   - NEVER remove percentages, counts, throughput, latency, user counts, time savings.
   - If the original says "reducing time by 40%" or "processing 1M contacts/day",
     those numbers MUST appear verbatim in "new".
   - Pattern: describe WHAT was done (add keywords here) → HOW → IMPACT (keep numbers).

6. KEYWORD DISTRIBUTION:
   - Spread keywords evenly across bullets — 2-3 keywords per bullet maximum.
   - Each keyword at MOST 2 times across all replacements.
   - Use synonyms and variations in different bullets.
   - Pick the most relevant bullet for each keyword.

7. NATURAL LANGUAGE:
   - The rewrite must read naturally, not like keyword stuffing.
   - No filler phrases: "ensuring customer success", "while ensuring mentorship",
     "applying expertise", "ensuring collaboration and communication".
   - Only add keywords that genuinely describe the work in that bullet.

8. Do NOT change:
   - Job titles, company names, dates, locations
   - Degree names, institution names
   - Contact information

9. Do NOT include replacements where old == new.

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact bullet text", "new": "rewritten bullet with keywords"}
  ]
}"""


def rewrite_experience(state: AgentState) -> dict:
    """Node: rewrite experience bullets with JD keywords."""
    sections = state.get("resume_sections", {})
    experience = sections.get("experience", [])
    missing = state.get("missing_keywords", [])
    categories = state.get("keyword_categories", {})
    gap = state.get("gap_analysis", "")
    required = state.get("required_keywords", [])
    preferred = state.get("preferred_keywords", [])

    if not experience or not missing:
        logger.info("Experience rewriter: no experience section or no missing keywords, skipping.")
        return {"raw_replacements": []}

    # Build the experience text block from the analyser's structured output
    exp_text_parts = []
    for entry in experience:
        if isinstance(entry, dict):
            company = entry.get("company", "Unknown")
            title = entry.get("title", "")
            text = entry.get("text", "")
            exp_text_parts.append(f"### {title} at {company}\n{text}")
        elif isinstance(entry, str):
            exp_text_parts.append(entry)
    experience_text = "\n\n".join(exp_text_parts)

    if not experience_text.strip():
        # Fall back to full resume text if sections weren't parsed well
        experience_text = state.get("resume_text", "")

    # Filter to keywords appropriate for experience bullets
    # (technical skills, tools, domain knowledge, action verbs)
    exp_cats = {"technical_skills", "tools_platforms", "domain_knowledge", "action_verbs"}
    exp_keywords = []
    for kw in missing:
        for cat, kws in categories.items():
            if cat in exp_cats and kw in kws:
                exp_keywords.append(kw)
                break
    # Also include required keywords not already covered
    for kw in missing:
        if kw in required and kw not in exp_keywords:
            exp_keywords.append(kw)

    if not exp_keywords:
        exp_keywords = missing

    missing_required = [kw for kw in exp_keywords if kw in required]
    missing_preferred = [kw for kw in exp_keywords if kw in preferred]
    missing_other = [kw for kw in exp_keywords if kw not in required and kw not in preferred]

    priority_block = ""
    if missing_required:
        priority_block += f"\n🔴 MUST ADD (required): {', '.join(missing_required)}"
    if missing_preferred:
        priority_block += f"\n🟡 SHOULD ADD (preferred): {', '.join(missing_preferred)}"
    if missing_other:
        priority_block += f"\n🟢 NICE TO ADD: {', '.join(missing_other)}"

    categories_block = "\n".join(
        f"- {cat}: {', '.join(kws)}" for cat, kws in categories.items()
    )

    sanitized_exp = _sanitize_user_input(experience_text)

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Experience Section (VERBATIM from resume)\n\n{sanitized_exp}\n\n"
            f"## Full Resume (for context)\n\n{_sanitize_user_input(state['resume_text'])}\n\n"
            f"## JD Keywords by Category\n\n{categories_block}\n\n"
            f"## Missing Keywords to Add{priority_block}\n\n"
            f"## Gap Analysis\n\n{gap}\n\n"
            "Rewrite experience bullets to incorporate missing keywords. "
            "Keep all metrics and numbers. One replacement per bullet."
        )},
    ])

    raw = data.get("replacements", [])
    raw = [r for r in raw if r.get("old") and r.get("new") and r["old"] != r["new"]]

    # Reject replacements where "old" is a bare section heading, job-title line,
    # or a job header containing a date range.
    _DATE_RANGE_RE = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+\d{4}",
        re.IGNORECASE,
    )

    def _is_real_bullet(old: str) -> bool:
        stripped = old.strip().lstrip("\u2022-\u2013\u2014* ")
        # Reject lines that contain a date (job header / tenure line)
        if _DATE_RANGE_RE.search(old):
            return False
        return len(stripped.split()) >= 5

    rejected = [r for r in raw if not _is_real_bullet(r["old"])]
    if rejected:
        logger.debug(
            "Experience rewriter: rejected %d header/title replacements: %s",
            len(rejected),
            [r["old"][:60] for r in rejected],
        )
    raw = [r for r in raw if _is_real_bullet(r["old"])]

    logger.info("Experience rewriter: produced %d replacements.", len(raw))
    return {"raw_replacements": raw}
