"""Cover letter, suggested job title, and LinkedIn outreach generator."""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json, _sanitize_user_input

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert career coach and professional writer.

Given a candidate's resume text and a target job description, generate:
1. A tailored cover letter (3–4 paragraphs, professional but not robotic).
2. The best-fit job title the candidate should use when applying.
3. A short LinkedIn connection/outreach message (2–3 sentences) the candidate
   can send to the hiring manager or recruiter.

RULES:
- The cover letter must reference specific achievements from the resume that
  align with the JD requirements.  Do NOT fabricate accomplishments.
- Keep the cover letter under 400 words.
- The LinkedIn message should feel personal, not templated.
- Do NOT include placeholder brackets like [Company Name] — use the actual
  company name if it can be inferred from the JD, otherwise use a generic
  phrase like "your team".

Return ONLY valid JSON:
{
  "cover_letter": "Dear Hiring Manager,\\n\\n...",
  "suggested_job_title": "Senior Software Engineer",
  "linkedin_message": "Hi [Name], I noticed..."
}"""


def generate_cover_letter(
    resume_text: str,
    jd_text: str,
    company_name: str | None = None,
) -> dict:
    """Generate a cover letter, suggested job title, and LinkedIn outreach.

    Returns dict with keys: cover_letter, suggested_job_title, linkedin_message.
    """
    resume_text = _sanitize_user_input(resume_text)
    jd_text = _sanitize_user_input(jd_text)

    company_hint = ""
    if company_name:
        company_name = _sanitize_user_input(company_name)
        company_hint = f"\n\n## Company\n{company_name}"

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"## Resume\n\n{resume_text}\n\n"
            f"## Job Description\n\n{jd_text}"
            f"{company_hint}\n\n"
            "Generate a cover letter, suggested job title, and LinkedIn message."
        )},
    ])

    result = {
        "cover_letter": data.get("cover_letter", ""),
        "suggested_job_title": data.get("suggested_job_title", ""),
        "linkedin_message": data.get("linkedin_message", ""),
    }
    logger.info(
        "Cover letter generated (%d chars), title=%r",
        len(result["cover_letter"]),
        result["suggested_job_title"],
    )
    return result
