"""Agent 9 — Critic / Reviewer Agent.

Acts as the final quality gate before the resume is scored and exported.
Reviews the optimised draft like a **skeptical senior recruiter AND an ATS
auditor simultaneously**.

Checks performed:
    * **Keyword stuffing** — deterministic pre-check via
      ``detect_keyword_stuffing()`` + LLM review.
    * **Repetition** — same achievements or keywords repeated excessively.
    * **Believability** — would a recruiter find this resume credible?
    * **Bullet specificity** — concrete actions and results, not vague claims.
    * **Role focus consistency** — coherent story about what the person does.
    * **Human readability** — does it read naturally, or AI-generated?
    * **Formatting consistency** — dates, bullet styles, section order.
    * **Length** — appropriately concise, no padding.

Pass/fail logic:
    * Any ``severity == "high"`` issue → ``passed = False``.
    * When failed, the ``revision_instructions`` dict tells the
      ``rewrite_router`` which section(s) to re-optimise.

Graph position:
    ``truth_guard`` [passed] → **critic** → conditional(
        passed  → ``final_score``,
        failed  → ``rewrite_router``
    )

State reads:
    ``draft_resume``, ``parsed_jd``, ``baseline_score``, ``truth_report``

State writes:
    ``critic_report`` — dict with keys: ``passed`` (bool),
    ``overall_quality``, ``issues`` (list), ``strengths`` (list),
    ``revision_instructions`` (dict with nullable ``summary``, ``skills``,
    ``experience`` keys).

Downstream consumers:
    * ``_after_critic`` conditional edge — checks ``passed`` to decide routing.
    * ``_after_rewrite_router`` — reads ``revision_instructions`` to pick
      which optimizer to route to.
    * ``final_score_node`` — receives the critic report for context.
"""

from __future__ import annotations

import json
import logging

from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState
from backend.services.agents.tools import detect_keyword_stuffing

logger = logging.getLogger(__name__)

_SYSTEM = """You are a skeptical senior recruiter and ATS auditor.

You receive an optimized resume draft, the JD analysis, and scoring data.
Review it critically as if you're deciding whether to pass it to a hiring manager.

═══ WHAT TO CHECK ═══

1. KEYWORD STUFFING: Is the resume packed with keywords that make it unreadable?
2. REPETITION: Are the same achievements or keywords repeated too much?
3. BELIEVABILITY: Would a recruiter find this resume credible?
4. BULLET SPECIFICITY: Are bullets concrete with actions and results, or vague?
5. ROLE FOCUS: Is there a consistent story about what this person does?
6. HUMAN READABILITY: Does it read naturally, or does it sound AI-generated?
7. FORMATTING CONSISTENCY: Are dates, bullet styles, and section order consistent?
8. LENGTH: Is it appropriately concise? No unnecessary padding?

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "passed": true/false,
  "overall_quality": "excellent|good|needs_work|poor",
  "issues": [
    {
      "type": "keyword_stuffing|repetition|unbelievable|vague_bullet|inconsistent_focus|unnatural_language|formatting|too_long",
      "location": "summary|skills|experience at Company X|projects",
      "description": "Specific description of the issue",
      "severity": "high|medium|low",
      "fix_suggestion": "How to fix this specific issue"
    }
  ],
  "strengths": ["Strong technical alignment", "Clear metrics in bullets"],
  "revision_instructions": {
    "summary": "instruction if summary needs revision, null otherwise",
    "skills": "instruction if skills need revision, null otherwise",
    "experience": "instruction if experience needs revision, null otherwise"
  }
}"""


def critic_node(state: ResumeGraphState) -> dict:
    """LangGraph node: review optimised resume as skeptical recruiter + ATS auditor.

    Workflow:
        1. Collect all JD keywords (must-have + good-to-have + ATS), deduped.
        2. Build a text blob from the draft resume for stuffing analysis.
        3. Run ``detect_keyword_stuffing()`` deterministically.
        4. Send the draft, JD analysis, baseline score, truth report, and
           stuffing issues to the LLM for a comprehensive quality review.
        5. Append any deterministic stuffing issues into the LLM’s ``issues``
           list.
        6. Auto-fail if any issue has ``severity == "high"``.

    Args:
        state: Pipeline state; reads ``draft_resume``, ``parsed_jd``,
               ``baseline_score``, ``truth_report``.

    Returns:
        ``{"critic_report": dict}`` with ``passed``, ``overall_quality``,
        ``issues``, ``strengths``, ``revision_instructions``.
    """
    draft = state.get("draft_resume", {})
    parsed_jd = state.get("parsed_jd", {})
    baseline = state.get("baseline_score", {})
    truth_report = state.get("truth_report", {})

    if not draft:
        logger.warning("Critic: no draft resume to review.")
        return {"critic_report": {"passed": True, "issues": []}}

    # Deterministic keyword stuffing check
    all_keywords = list(dict.fromkeys(
        parsed_jd.get("must_have_skills", [])
        + parsed_jd.get("good_to_have_skills", [])
        + parsed_jd.get("ats_keywords", [])
    ))

    # Build text for stuffing check
    text_parts = [draft.get("summary", "")]
    skills = draft.get("skills", {})
    if isinstance(skills, dict):
        for items in skills.values():
            if isinstance(items, list):
                text_parts.extend(items)
    for exp in draft.get("experience", []):
        text_parts.extend(exp.get("bullets", []))
    resume_text = " ".join(str(p) for p in text_parts if p)

    stuffing_issues = detect_keyword_stuffing(resume_text, all_keywords)

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Optimized Resume Draft\n\n{json.dumps(draft, indent=2)}\n\n"
            f"## JD Analysis\n\n{json.dumps(parsed_jd, indent=2)}\n\n"
            f"## Baseline Score\n\n{json.dumps(baseline, indent=2)}\n\n"
            f"## Truth Guard Report\n\n{json.dumps(truth_report, indent=2)}\n\n"
            f"## Keyword Stuffing Detection\n\n{json.dumps(stuffing_issues, indent=2)}\n\n"
            "Review this resume critically. Be thorough."
        )},
    ])

    # Inject stuffing issues into critic findings
    issues = data.get("issues", [])
    if stuffing_issues:
        for si in stuffing_issues:
            issues.append({
                "type": "keyword_stuffing",
                "location": "throughout",
                "description": f"'{si['keyword']}' appears {si['count']} times (max {si['max']})",
                "severity": "medium",
                "fix_suggestion": f"Reduce occurrences of '{si['keyword']}' to {si['max']} or fewer",
            })

    data["issues"] = issues

    # Auto-fail on high-severity issues
    high_issues = [i for i in issues if i.get("severity") == "high"]
    if high_issues:
        data["passed"] = False

    logger.info(
        "Critic: passed=%s, %d issues (%d high), quality=%s.",
        data.get("passed", True),
        len(issues),
        len(high_issues),
        data.get("overall_quality", "?"),
    )

    return {"critic_report": data}
