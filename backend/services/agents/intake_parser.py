"""Agent 1 — Intake / Parser Agent.

Reads uploaded resume text (extracted from PDF or LaTeX by ``parser.py`` /
``latex_parser.py``) and converts it into a structured JSON dict.

Responsibilities:
    * Normalise messy PDF-extracted text (spacing artefacts, merged lines).
    * Split sections (summary, skills, experience, projects, education, certs).
    * Extract individual bullet points.
    * Detect and remove duplicate entries.
    * Categorise skills into groups (languages, frontend, backend, etc.).
    * Preserve original facts *exactly* — do NOT invent missing fields.

Graph position:
    Entry point → first node in the pipeline.

State reads:
    ``raw_resume_text``

State writes:
    ``parsed_resume`` — structured dict with keys: ``basics``, ``summary``,
    ``skills`` (categorised dict), ``experience``, ``projects``, ``education``,
    ``certifications``.

Downstream consumers:
    Every subsequent agent reads ``parsed_resume``.  The structure it produces
    is the *canonical internal representation* of the resume for the entire
    pipeline.
"""

from __future__ import annotations

import logging

from backend.services.agents.llm import invoke_llm_json, sanitize_input
from backend.services.agents.state import ResumeGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a resume parsing specialist.

You receive raw resume text (extracted from PDF or LaTeX). Your job is to parse it
into a clean, structured JSON object.

═══ RULES ═══

1. Extract EXACTLY what is in the resume. Do NOT invent or infer missing fields.
2. If a field is not present, use null or empty array/string.
3. Normalize section names but preserve all content.
4. Split bullet points correctly — each action item should be a separate bullet.
5. Detect and remove duplicate entries.
6. Preserve original facts, numbers, dates, and metrics exactly.
7. Skills should be individual items, not comma-separated groups within one item.
8. For skills categorization, put each skill in the most appropriate category.
   If unsure, use "tools" as a catch-all.

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "basics": {
    "name": "string",
    "email": "string or null",
    "phone": "string or null",
    "linkedin": "string or null",
    "github": "string or null",
    "location": "string or null"
  },
  "summary": "string or null",
  "skills": {
    "languages": [],
    "frontend": [],
    "backend": [],
    "cloud": [],
    "ai": [],
    "tools": []
  },
  "experience": [
    {
      "company": "string",
      "title": "string",
      "start": "string",
      "end": "string or Present",
      "bullets": ["action bullet 1", "action bullet 2"]
    }
  ],
  "projects": [
    {
      "name": "string",
      "stack": ["tech1", "tech2"],
      "bullets": ["what it does"],
      "link": "string or null"
    }
  ],
  "education": [
    {
      "degree": "string",
      "institution": "string",
      "location": "string or null",
      "graduation_date": "string",
      "details": null
    }
  ],
  "certifications": [
    {
      "name": "string",
      "issuer": "string or null",
      "date": "string or null"
    }
  ]
}"""


def parse_resume_node(state: ResumeGraphState) -> dict:
    """LangGraph node: parse raw resume text into structured JSON.

    Sends the sanitised resume text to the LLM with a system prompt that
    defines the exact JSON schema.  Validates that at least ``basics.name``
    was extracted.

    Args:
        state: Pipeline state; reads ``raw_resume_text``.

    Returns:
        ``{"parsed_resume": dict}`` with keys ``basics``, ``summary``,
        ``skills``, ``experience``, ``projects``, ``education``,
        ``certifications``.
    """
    raw_text = sanitize_input(state["raw_resume_text"])

    data = invoke_llm_json([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"## Raw Resume Text\n\n{raw_text}"},
    ])

    # Validate essential structure
    basics = data.get("basics", {})
    if not basics.get("name"):
        logger.warning("Parser: could not extract name from resume.")

    skills = data.get("skills", {})
    experience = data.get("experience", [])
    projects = data.get("projects", [])

    logger.info(
        "Parser: name=%s, %d skill categories, %d experience, %d projects.",
        basics.get("name", "?"),
        len([k for k, v in skills.items() if v]),
        len(experience),
        len(projects),
    )

    return {"parsed_resume": data}
