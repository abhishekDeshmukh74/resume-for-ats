"""Agent 4 — Quality Assurance: validate replacements & remove keyword duplication."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from backend.models import TextReplacement
from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import AgentState

logger = logging.getLogger(__name__)

# ── AI phrase blacklist (inspired by Resume-Matcher) ──────────────────────
#
# These phrases sound AI-generated and may trigger recruiter suspicion.
# We replace them with simpler, human-sounding alternatives.
_AI_PHRASE_REPLACEMENTS: dict[str, str] = {
    # Action verb inflation (inspired by Resume-Matcher's 100+ list)
    "spearheaded": "led",
    "orchestrated": "coordinated",
    "synergized": "collaborated",
    "leveraged": "used",
    "revolutionized": "transformed",
    "pioneered": "introduced",
    "catalyzed": "initiated",
    "operationalized": "implemented",
    "architected": "designed",
    "effectuated": "completed",
    "endeavored": "worked",
    "facilitated": "helped",
    "utilized": "used",
    "championed": "promoted",
    "streamlined": "improved",
    "empowered": "enabled",
    "galvanized": "motivated",
    "conceptualized": "planned",
    "ideated": "brainstormed",
    "instrumentalized": "applied",
    "actualized": "achieved",
    "optimized": "improved",
    "strategized": "planned",
    "synthesized": "combined",
    "envisioned": "planned",
    "instituted": "established",
    "promulgated": "shared",
    "evangelized": "advocated",
    "amplified": "increased",
    "calibrated": "adjusted",
    "crystallized": "clarified",
    "galvanised": "motivated",
    "helmed": "led",
    "inaugurated": "started",
    "matriculated": "enrolled",
    "proliferated": "expanded",
    "quarterbacked": "led",
    "shepherded": "guided",
    "trailblazed": "started",
    "alchemized": "transformed",
    "turbo-charged": "accelerated",
    "turbocharged": "accelerated",
    # Buzzword nouns/adjectives
    "synergy": "collaboration",
    "paradigm shift": "change",
    "best-in-class": "top-performing",
    "world-class": "high-quality",
    "cutting-edge": "modern",
    "bleeding-edge": "modern",
    "game-changing": "innovative",
    "holistic": "comprehensive",
    "actionable": "practical",
    "disruptive": "innovative",
    "thought leadership": "expertise",
    "thought leader": "expert",
    "deep dive": "analysis",
    "ecosystem": "environment",
    "paradigm": "approach",
    "state-of-the-art": "advanced",
    "mission-critical": "essential",
    "next-generation": "advanced",
    "transformative": "significant",
    "groundbreaking": "notable",
    "seamlessly": "smoothly",
    "robust": "strong",
    "scalable": "flexible",
    "unprecedented": "significant",
    "best of breed": "leading",
    "core competency": "key skill",
    "value proposition": "benefit",
    "low-hanging fruit": "quick win",
    # Filler phrases
    "in order to": "to",
    "for the purpose of": "to",
    "at the end of the day": "",
    "moving forward": "",
    "going forward": "",
    "on a daily basis": "daily",
    "in a timely manner": "promptly",
    "due to the fact that": "because",
    "with respect to": "about",
    "in the context of": "in",
    "with a view to": "to",
    "in terms of": "in",
    "as a means of": "to",
    "in an effort to": "to",
    "it is worth noting that": "",
    "it should be noted that": "",
    "needless to say": "",
    "as a matter of fact": "",
    "in light of the fact that": "because",
    "by virtue of": "through",
    "with regard to": "about",
    "in the event that": "if",
    "on the grounds that": "because",
}


def _clean_ai_phrases(text: str, jd_text: str = "") -> str:
    """Replace AI-sounding phrases with simpler alternatives.

    Phrases that appear in the job description are protected — if the JD
    uses "cutting-edge" or "spearheaded", keep them as-is since the ATS
    may match on those exact words.
    """
    jd_lower = jd_text.lower()
    cleaned = text
    for phrase, replacement in _AI_PHRASE_REPLACEMENTS.items():
        # Protect phrases that appear in the job description
        if jd_lower and phrase.lower() in jd_lower:
            continue
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        cleaned = pattern.sub(replacement, cleaned)
    # Clean up double spaces from removals
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned

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
    raw = state.get("raw_replacements", [])
    keywords = state.get("jd_keywords", [])
    jd_text = state.get("jd_text", "")

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

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Original Resume\n\n{state['resume_text']}\n\n"
            f"## JD Keywords\n\n{', '.join(keywords)}\n\n"
            f"## Proposed Replacements\n\n{json.dumps(raw, indent=2)}"
            f"{duplication_note}\n\n"
            "Review and fix all replacements. Return the corrected JSON."
        )},
    ])
    reviewed = data.get("replacements", [])
    fixes = data.get("fixes_applied", [])

    if fixes:
        logger.info("QA agent: applied %d fixes: %s", len(fixes), "; ".join(fixes[:5]))

    # Final programmatic dedup safety net
    _SKILLS_LINE = re.compile(r"^[A-Za-z][^:]{0,40}:\s*.+")
    # Only match trailing prose that starts with a conjunction and continues
    # as a sentence (contains a verb/gerund or is 5+ words). This avoids
    # stripping legit keyword phrases like "Docker, with Kubernetes, Terraform".
    _TRAILING_PROSE = re.compile(
        r",?\s+(?:with|and|including|such as|plus)\s+"
        r"(?:experience|expertise|proficiency|knowledge|"
        r"a focus|focus|emphasis|background|"
        r"(?:\w+ing)\b)"  # gerund verb form (e.g. "ensuring", "building")
        r".+$",
        re.IGNORECASE,
    )

    def _enforce_skills_format(old: str, new: str) -> str:
        """Strip trailing prose from skills-line replacements."""
        if not _SKILLS_LINE.match(old.strip()):
            return new
        # Check that old is a skills line (no full stop, mostly comma-separated)
        old_clean = old.strip()
        looks_like_skills = old_clean.count(",") >= 1 and not old_clean.endswith(".")
        if not looks_like_skills:
            return new
        # Strip only sentence-like prose appended after keywords
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
        # Clean AI-sounding phrases from new text (JD-aware: protects phrases in JD)
        new = _clean_ai_phrases(new, jd_text)
        # Master alignment: reject replacements that fabricate companies or dates
        new = _guard_master_alignment(old, new, state["resume_text"])
        if old == new:
            continue
        final.append(TextReplacement(old=old, new=new))

    logger.info("QA agent: %d → %d final replacements.", len(raw), len(final))

    return {"replacements": final}


# ── Master-resume alignment guard (inspired by Resume-Matcher) ───────────

_COMPANY_RE = re.compile(
    r"\b(?:at|@)\s+([A-Z][A-Za-z0-9& .-]{1,40})\b"
)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4}\b"
    r"|\b\d{4}\s*[-–—]\s*(?:Present|\d{4})\b",
    re.IGNORECASE,
)


def _guard_master_alignment(old: str, new: str, resume_text: str) -> str:
    """Prevent the LLM from fabricating company names or altering dates.

    If the new text introduces company names not in the original resume, or
    changes date ranges, fall back to the original text for those segments.
    Inspired by Resume-Matcher's 4-gate diff validation.
    """
    # Check for fabricated company references
    original_companies = {m.group(1).strip().lower() for m in _COMPANY_RE.finditer(resume_text)}
    new_companies = {m.group(1).strip().lower() for m in _COMPANY_RE.finditer(new)}
    fabricated = new_companies - original_companies
    if fabricated:
        logger.warning("QA alignment: blocked fabricated company refs: %s", fabricated)
        return old  # Reject the entire replacement

    # Check for altered dates — dates in "new" must exist in "old" or resume
    old_dates = set(_DATE_RE.findall(old))
    resume_dates = set(_DATE_RE.findall(resume_text))
    allowed_dates = old_dates | resume_dates
    new_dates = set(_DATE_RE.findall(new))
    fabricated_dates = new_dates - allowed_dates
    if fabricated_dates:
        logger.warning("QA alignment: blocked fabricated dates: %s", fabricated_dates)
        return old  # Reject the entire replacement

    return new
