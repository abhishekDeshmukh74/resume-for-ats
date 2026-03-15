"""Agent 3 — Rewrite resume sections to incorporate JD keywords."""

from __future__ import annotations

import logging

from backend.services.agents.llm import get_llm, parse_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert ATS resume rewriter.

You receive:
- The EXACT original resume text
- Categorised JD keywords
- A gap analysis telling you which keywords to add where

Your job is to produce a JSON array of {"old": "...", "new": "..."} replacements
that transform the resume to score above 90% on ATS keyword matching.

═══ CRITICAL REPLACEMENT RULES ═══

1. "old" MUST be a VERBATIM, character-for-character copy from the Original Resume.
   Do NOT fix typos, change spacing, add/remove punctuation, or rephrase.
   Copy the EXACT text including line breaks.

2. "new" must be approximately the SAME LENGTH as "old" (±20%).
   The text must fit in the same physical space on the PDF.
   If you add a keyword, remove filler words to compensate.

3. ONE replacement per bullet point. ONE replacement for the summary.
   Do NOT combine multiple bullets.

4. KEYWORD DISTRIBUTION — THIS IS CRITICAL:
   - Spread keywords EVENLY across all bullets and sections.
   - Each keyword should appear at MOST 2 times in the entire resume.
   - Do NOT stuff the same keyword into every bullet.
   - Use SYNONYMS and VARIATIONS (e.g. "microservices" in one bullet,
     "service-oriented architecture" in another).
   - The summary gets broad keywords (role title, years, key tech).
   - Each experience bullet gets 2-3 specific keywords relevant to THAT bullet.
   - Skills section gets the comprehensive keyword list.

5. Do NOT include replacements where old == new.

6. PRESERVE ALL METRICS AND NUMBERS:
   - NEVER remove or dilute quantitative impact (percentages, counts, throughput,
     latency, user counts, time savings, etc.).
   - If the original says "reducing time by 40%" or "processing 1M contacts/day
     at 1K events/sec", those numbers MUST appear verbatim in "new".
   - Do NOT replace concrete metrics with vague phrases like "optimizing efficiency",
     "ensuring reliability", or "process optimization".
   - Preferred pattern: "Built X → using Y → impact Z" where Z keeps the original
     numbers. Add JD keywords into the X/Y parts, never strip the Z.

7. Do NOT pad with empty filler phrases:
   - NO "ensuring customer success", "leveraging CI/CD pipelines" (unless the
     original already mentions them), "ensuring collaboration and communication",
     "while ensuring mentorship", "applying expertise" etc.
   - Only add keywords that genuinely describe the work in that bullet.

8. Do NOT change:
   - Section headers, job titles, company names, dates
   - Degree names, institution names
   - Contact information

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact text from resume", "new": "ATS-optimised rewrite"}
  ]
}"""


def rewrite_sections(state: AgentState) -> dict:
    """Node: generate old→new replacement pairs."""
    llm = get_llm()

    categories = state.get("keyword_categories", {})
    gap = state.get("gap_analysis", "")

    keywords_block = "\n".join(
        f"- {cat}: {', '.join(kws)}"
        for cat, kws in categories.items()
    )

    resp = llm.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Original Resume\n\n{state['resume_text']}\n\n"
            f"## JD Keywords by Category\n\n{keywords_block}\n\n"
            f"## Gap Analysis\n\n{gap}\n\n"
            "Now generate the replacements array. Remember:\n"
            "- VERBATIM 'old' text from the resume above\n"
            "- 'new' text ±20% same length\n"
            "- Each keyword appears at MOST 2 times across ALL replacements\n"
            "- Spread keywords evenly, use synonyms/variations\n"
            "- KEEP ALL NUMBERS AND METRICS from the original (%, counts, latency, throughput, time savings)\n"
            "- Do NOT replace metrics with vague filler like 'ensuring reliability' or 'process optimization'\n"
            "Return only the JSON object."
        )},
    ])

    data = parse_llm_json(resp.content)
    raw = data.get("replacements", [])

    # Filter out identical replacements
    raw = [r for r in raw if r.get("old") and r.get("new") and r["old"] != r["new"]]

    logger.info("Rewriter agent: produced %d raw replacements.", len(raw))

    return {"raw_replacements": raw}
