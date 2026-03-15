"""Agent 4 — Quality Assurance: validate replacements & remove keyword duplication."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from backend.models import TextReplacement
from backend.services.agents.llm import get_llm, parse_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a QA reviewer for ATS resume optimisation.

You receive:
- The original resume text
- A list of proposed {"old", "new"} replacements
- The full JD keyword list

Your job is to review and FIX every replacement to ensure quality:

═══ CHECKS ═══

1. VERBATIM OLD TEXT: Each "old" must appear EXACTLY in the original resume.
   If it doesn't match, fix it to match the resume text exactly.

2. KEYWORD DUPLICATION: Count how many times each JD keyword appears across ALL
   "new" texts combined. If any keyword appears more than 2 times total:
   - Replace excess occurrences with synonyms or rephrase
   - Example: if "microservices" appears 5 times, keep it in the 2 most relevant
     bullets and use "distributed services", "service-oriented", etc. in others

3. METRIC PRESERVATION: Every number, percentage, throughput figure, latency,
   user count, and time-saving from the "old" text MUST appear in the "new" text.
   If a replacement drops metrics (e.g. changes "reducing triage time by 40%" to
   "ensuring reliability"), restore the original metrics. Numbers make recruiters
   stop scrolling — never dilute them.

4. NO FILLER PADDING: Remove vague trailing phrases that were not in the original,
   such as "ensuring customer success", "while ensuring mentorship and code reviews",
   "applying expertise", "ensuring collaboration and communication".
   Only add keywords that genuinely describe the actual work.

5. LENGTH: Each "new" text should be within ±20% of its "old" text length.
   If too long, cut filler words. If too short, add relevant detail.

6. NATURALNESS: The "new" text must read naturally, not like keyword stuffing.

7. NO IDENTICAL PAIRS: Remove any replacement where old == new.

6. SKILLS SECTION — COMMA-SEPARATED KEYWORDS ONLY:
   Any replacement whose "old" text matches a skills-line pattern
   (e.g. "Category: Skill1, Skill2, Skill3") must keep the "new" text in the
   SAME format: comma-separated keywords with NO trailing prose, filler phrases,
   or sentences (no "with experience in...", "and expertise in...", etc.).
   GOOD: "Frontend: React, Next.js, Tailwind CSS, Redux, HTML, CSS"
   BAD:  "Frontend: React, Next.js, Tailwind CSS, with experience in Clerk."

═══ RESPONSE FORMAT ═══

{
  "replacements": [
    {"old": "verified exact text", "new": "deduplicated rewrite"}
  ],
  "fixes_applied": ["description of each fix made"]
}

Return ONLY valid JSON."""


def qa_and_deduplicate(state: AgentState) -> dict:
    """Node: validate old text accuracy, remove keyword duplication."""
    llm = get_llm()

    raw = state.get("raw_replacements", [])
    keywords = state.get("jd_keywords", [])

    if not raw:
        logger.warning("QA agent: no raw replacements to review.")
        return {"replacements": []}

    # Pre-check: flag keyword duplication for the LLM
    all_new_text = " ".join(r.get("new", "") for r in raw).lower()
    freq = Counter()
    for kw in keywords:
        count = all_new_text.count(kw.lower())
        if count > 2:
            freq[kw] = count

    duplication_note = ""
    if freq:
        duplication_note = "\n\nKEYWORD OVERUSE DETECTED (fix these):\n" + "\n".join(
            f"- \"{kw}\" appears {c} times (max 2 allowed)"
            for kw, c in freq.most_common(20)
        )

    resp = llm.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Original Resume\n\n{state['resume_text']}\n\n"
            f"## JD Keywords\n\n{', '.join(keywords)}\n\n"
            f"## Proposed Replacements\n\n{json.dumps(raw, indent=2)}"
            f"{duplication_note}\n\n"
            "Review and fix all replacements. Return the corrected JSON."
        )},
    ])

    data = parse_llm_json(resp.content)
    reviewed = data.get("replacements", [])
    fixes = data.get("fixes_applied", [])

    if fixes:
        logger.info("QA agent: applied %d fixes: %s", len(fixes), "; ".join(fixes[:5]))

    # Final programmatic dedup safety net
    _SKILLS_LINE = re.compile(r"^[A-Za-z][^:]{0,40}:\s*.+")
    _TRAILING_PROSE = re.compile(r",?\s+(?:with|and|including|such as|plus)\b.+$", re.IGNORECASE)

    def _enforce_skills_format(old: str, new: str) -> str:
        """Strip trailing prose from skills-line replacements."""
        if not _SKILLS_LINE.match(old.strip()):
            return new
        # Check that old is a skills line (no full stop, mostly comma-separated)
        old_clean = old.strip()
        looks_like_skills = old_clean.count(",") >= 1 and not old_clean.endswith(".")
        if not looks_like_skills:
            return new
        # Strip any prose appended after the last real keyword
        cleaned = _TRAILING_PROSE.sub("", new.strip()).rstrip(",; ").rstrip(".")
        if cleaned != new.strip():
            logger.debug("QA post-filter: stripped prose from skills line: %r → %r", new, cleaned)
        return cleaned

    final: list[TextReplacement] = []
    seen_old: set[str] = set()
    for r in reviewed:
        old = r.get("old", "").strip()
        new = r.get("new", "").strip()
        if not old or not new or old == new:
            continue
        if old in seen_old:
            continue
        seen_old.add(old)
        new = _enforce_skills_format(old, new)
        if old == new:
            continue
        final.append(TextReplacement(old=old, new=new))

    logger.info("QA agent: %d → %d final replacements.", len(raw), len(final))

    return {"replacements": final}
