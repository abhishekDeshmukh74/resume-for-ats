import json
import logging
import os
from groq import Groq, RateLimitError
from backend.models import ResumeData

logger = logging.getLogger(__name__)

_clients: list[Groq] = []
_current_idx: int = 0


def _get_groq_api_keys() -> list[str]:
    """Read GROQ API keys from env (comma-separated GROQ_API_KEYS, fallback GROQ_API_KEY)."""
    keys_csv = os.environ.get("GROQ_API_KEYS", "")
    if keys_csv:
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GROQ_API_KEY", "")
    if single.strip():
        return [single.strip()]
    raise RuntimeError(
        "Neither GROQ_API_KEYS nor GROQ_API_KEY environment variable is set."
    )


def _init_clients() -> None:
    global _clients
    if not _clients:
        keys = _get_groq_api_keys()
        _clients = [Groq(api_key=k) for k in keys]
        logger.info("Initialised %d Groq client(s)", len(_clients))


def _get_client() -> Groq:
    _init_clients()
    return _clients[_current_idx]


def _rotate_client() -> bool:
    """Rotate to the next API key. Returns False if all keys exhausted."""
    global _current_idx
    _init_clients()
    next_idx = (_current_idx + 1) % len(_clients)
    if next_idx == _current_idx:
        return False  # only one key, can't rotate
    _current_idx = next_idx
    logger.info("Rotated to Groq API key #%d", _current_idx + 1)
    return True


_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) resume optimiser.
You receive the EXACT TEXT extracted from a candidate's PDF resume and a target job description (JD).
Your goal is to rewrite the resume so it scores **above 90 %** on ATS keyword matching.

═══ CRITICAL: THE "replacements" ARRAY DRIVES THE PDF REWRITE ═══

The "replacements" array is what actually changes the PDF. If the "old" values
are wrong, NOTHING gets changed and the user receives their original resume.

REPLACEMENT RULES:
  • "old" must be a VERBATIM, character-for-character copy from the Original Resume.
    Do NOT fix typos, change spacing, add/remove punctuation, or rephrase.
  • ONE replacement per bullet point. Do NOT combine multiple bullets into one replacement.
  • ONE replacement for the summary paragraph.
  • ONE replacement per skills line you change.
  • "new" must be approximately the SAME LENGTH as "old" (±20%).
    The text must fit in the same physical space on the PDF.
    If you need more words, use shorter synonyms. Do not make "new" much longer than "old".
  • Do NOT include replacements where old == new.

What to rewrite:
  • EVERY experience bullet — weave in JD keywords, action verbs, technologies.
  • The professional summary — tailor to the target role.
  • Skills lines — reorder, add JD-relevant aliases (e.g. "JS" → "JavaScript").

What NOT to change (never include these as "old"):
  • Section headers (EXPERIENCE, EDUCATION, SKILLS, etc.)
  • Job titles, company names, employment dates
  • Degree names, institution names, graduation dates
  • Certifications, contact info

═══ RESPONSE FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact verbatim text from resume", "new": "ATS-optimised rewrite of similar length"}
  ],
  "name": "string",
  "email": "string or null",
  "phone": "string or null",
  "linkedin": "string or null",
  "github": "string or null",
  "location": "string or null",
  "summary": "rewritten summary",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "job_title": "string",
      "company": "string",
      "location": "string or null",
      "start_date": "string",
      "end_date": "string",
      "bullets": ["rewritten bullet"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "institution": "string",
      "location": "string or null",
      "graduation_date": "string",
      "details": ["string"] or null
    }
  ],
  "certifications": [
    {
      "name": "string",
      "issuer": "string or null",
      "date": "string or null"
    }
  ],
  "ats_score": 90,
  "matched_keywords": ["keyword1", "keyword2"]
}

═══ STRUCTURE RULES ═══

• Same number of experience entries, bullets per role, education entries.
• Do NOT add, remove, merge entries or fabricate experience.
• ats_score: integer 0-100, must reflect keyword coverage AFTER your rewrites.
• matched_keywords: top JD keywords now present in the resume.

Return ONLY valid JSON."""


def generate_resume(resume_text: str, jd_text: str) -> ResumeData:
    """Call Groq to generate a tailored resume with explicit replacement pairs.
    Automatically rotates API keys on rate-limit errors."""

    user_prompt = (
        f"## Original Resume\n\n{resume_text}\n\n"
        f"## Job Description\n\n{jd_text}\n\n"
        "Tailor the resume to achieve an ATS keyword-match score ABOVE 90%.\n\n"
        "CRITICAL INSTRUCTIONS FOR REPLACEMENTS:\n"
        "1. Every 'old' value MUST be an EXACT copy-paste from the Original Resume "
        "text above — character for character, including all punctuation and spacing.\n"
        "2. Rewrite EVERY bullet point to incorporate JD keywords.\n"
        "3. Rewrite the summary to target this specific role.\n"
        "4. Reorder and augment skills with JD-relevant terms.\n"
        "5. The 'replacements' array is what actually changes the PDF — if it is "
        "empty or the 'old' values don't match, the user gets their original resume.\n\n"
        "Return only the JSON object."
    )

    _init_clients()
    tried = 0
    last_err: Exception | None = None

    while tried < len(_clients):
        client = _get_client()
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError as exc:
            last_err = exc
            tried += 1
            logger.warning("Rate-limited on key #%d: %s", _current_idx + 1, exc)
            if not _rotate_client():
                break
    else:
        raise RuntimeError(
            f"All {len(_clients)} Groq API key(s) are rate-limited."
        ) from last_err

    raw = completion.choices[0].message.content
    data = json.loads(raw)

    resume = ResumeData(**data)

    # Log replacement stats
    n = len(resume.replacements)
    identical = sum(1 for r in resume.replacements if r.old == r.new)
    logger.info("AI returned %d replacements (%d identical/skipped).", n, identical)
    if n == 0:
        logger.warning("AI returned ZERO replacements — resume will be unchanged.")

    return resume
