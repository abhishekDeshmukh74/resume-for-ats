"""Agent 10 — Formatter / Export Agent.

Takes approved resume JSON and formats into:
  - Plain text
  - Markdown
  - Generates old→new replacements for PDF rewriting

ATS formatting rules:
  - Simple section headings
  - No complex tables
  - Standard date formatting
  - Predictable section order
"""

from __future__ import annotations

import json
import logging

from backend.models import TextReplacement
from backend.services.agents.llm import invoke_llm_json
from backend.services.agents.state import ResumeGraphState

logger = logging.getLogger(__name__)


def _format_plain_text(resume: dict) -> str:
    """Format resume dict into ATS-friendly plain text."""
    lines: list[str] = []

    # Header
    basics = resume.get("basics", {})
    if basics.get("name"):
        lines.append(basics["name"])
    contact = []
    if basics.get("email"):
        contact.append(basics["email"])
    if basics.get("phone"):
        contact.append(basics["phone"])
    if basics.get("location"):
        contact.append(basics["location"])
    if contact:
        lines.append(" | ".join(contact))
    links = []
    if basics.get("linkedin"):
        links.append(basics["linkedin"])
    if basics.get("github"):
        links.append(basics["github"])
    if links:
        lines.append(" | ".join(links))
    lines.append("")

    # Summary
    summary = resume.get("summary")
    if summary:
        lines.append("SUMMARY")
        lines.append(summary)
        lines.append("")

    # Skills
    skills = resume.get("skills", {})
    if skills:
        lines.append("SKILLS")
        if isinstance(skills, dict):
            for category, items in skills.items():
                if isinstance(items, list) and items:
                    lines.append(f"{category.title()}: {', '.join(items)}")
        elif isinstance(skills, list):
            lines.append(", ".join(skills))
        lines.append("")

    # Experience
    experience = resume.get("experience", [])
    if experience:
        lines.append("EXPERIENCE")
        for exp in experience:
            title_line = f"{exp.get('title', '')} — {exp.get('company', '')}"
            date_line = f"{exp.get('start', '')} – {exp.get('end', '')}"
            lines.append(f"{title_line}  ({date_line})")
            for bullet in exp.get("bullets", []):
                lines.append(f"  - {bullet}")
            lines.append("")

    # Projects
    projects = resume.get("projects", [])
    if projects:
        lines.append("PROJECTS")
        for proj in projects:
            name = proj.get("name", "")
            stack = proj.get("stack", [])
            if stack:
                lines.append(f"{name} [{', '.join(stack)}]")
            else:
                lines.append(name)
            for bullet in proj.get("bullets", []):
                lines.append(f"  - {bullet}")
            if proj.get("link"):
                lines.append(f"  {proj['link']}")
            lines.append("")

    # Education
    education = resume.get("education", [])
    if education:
        lines.append("EDUCATION")
        for edu in education:
            lines.append(f"{edu.get('degree', '')} — {edu.get('institution', '')}")
            if edu.get("graduation_date"):
                lines.append(f"  {edu['graduation_date']}")
        lines.append("")

    # Certifications
    certs = resume.get("certifications", [])
    if certs:
        lines.append("CERTIFICATIONS")
        for cert in certs:
            cert_line = cert.get("name", "")
            if cert.get("issuer"):
                cert_line += f" — {cert['issuer']}"
            if cert.get("date"):
                cert_line += f" ({cert['date']})"
            lines.append(f"  - {cert_line}")
        lines.append("")

    return "\n".join(lines)


def _format_markdown(resume: dict) -> str:
    """Format resume dict into Markdown."""
    lines: list[str] = []

    basics = resume.get("basics", {})
    if basics.get("name"):
        lines.append(f"# {basics['name']}")
    contact = []
    if basics.get("email"):
        contact.append(basics["email"])
    if basics.get("phone"):
        contact.append(basics["phone"])
    if basics.get("location"):
        contact.append(basics["location"])
    if contact:
        lines.append(" | ".join(contact))
    links = []
    if basics.get("linkedin"):
        links.append(f"[LinkedIn]({basics['linkedin']})")
    if basics.get("github"):
        links.append(f"[GitHub]({basics['github']})")
    if links:
        lines.append(" | ".join(links))
    lines.append("")

    summary = resume.get("summary")
    if summary:
        lines.append("## Summary")
        lines.append(summary)
        lines.append("")

    skills = resume.get("skills", {})
    if skills:
        lines.append("## Skills")
        if isinstance(skills, dict):
            for category, items in skills.items():
                if isinstance(items, list) and items:
                    lines.append(f"**{category.title()}:** {', '.join(items)}")
        lines.append("")

    experience = resume.get("experience", [])
    if experience:
        lines.append("## Experience")
        for exp in experience:
            lines.append(f"### {exp.get('title', '')} — {exp.get('company', '')}")
            lines.append(f"*{exp.get('start', '')} – {exp.get('end', '')}*")
            lines.append("")
            for bullet in exp.get("bullets", []):
                lines.append(f"- {bullet}")
            lines.append("")

    projects = resume.get("projects", [])
    if projects:
        lines.append("## Projects")
        for proj in projects:
            name = proj.get("name", "")
            stack = proj.get("stack", [])
            if stack:
                lines.append(f"### {name}")
                lines.append(f"*{', '.join(stack)}*")
            else:
                lines.append(f"### {name}")
            lines.append("")
            for bullet in proj.get("bullets", []):
                lines.append(f"- {bullet}")
            if proj.get("link"):
                lines.append(f"- [{proj['link']}]({proj['link']})")
            lines.append("")

    education = resume.get("education", [])
    if education:
        lines.append("## Education")
        for edu in education:
            lines.append(f"**{edu.get('degree', '')}** — {edu.get('institution', '')}")
            if edu.get("graduation_date"):
                lines.append(f"*{edu['graduation_date']}*")
            lines.append("")

    certs = resume.get("certifications", [])
    if certs:
        lines.append("## Certifications")
        for cert in certs:
            cert_line = f"- {cert.get('name', '')}"
            if cert.get("issuer"):
                cert_line += f" — {cert['issuer']}"
            if cert.get("date"):
                cert_line += f" ({cert['date']})"
            lines.append(cert_line)
        lines.append("")

    return "\n".join(lines)


_REPLACEMENT_SYSTEM = """You are a text replacement specialist.

You receive:
- The ORIGINAL raw resume text
- The OPTIMIZED resume content (summary, skills, experience, projects)

Your job: produce a JSON array of {"old": "...", "new": "..."} replacements
that transform the original text into the optimized version.

═══ RULES ═══

1. "old" MUST be a VERBATIM, character-for-character substring from the original resume text.
2. "new" is the optimized replacement text.
3. ONE replacement per logical section (one per bullet, one for summary, etc.).
4. Do NOT change section headers, job titles, company names, dates, or contact info.
5. Do NOT include replacements where old == new.
6. Focus on the sections that were actually changed: summary, skills lines, and experience bullets.
7. Keep replacements as targeted as possible — don't replace entire sections if only a bullet changed.

═══ OUTPUT FORMAT (pure JSON, no markdown) ═══

{
  "replacements": [
    {"old": "exact text from original resume", "new": "optimized replacement text"}
  ]
}"""


def _generate_replacements(
    raw_text: str, original_resume: dict, optimized_resume: dict,
) -> list[TextReplacement]:
    """Use LLM to generate old→new text replacements for PDF rewriting."""
    data = invoke_llm_json([
        {"role": "system", "content": _REPLACEMENT_SYSTEM},
        {"role": "user", "content": (
            f"## Original Raw Resume Text\n\n{raw_text}\n\n"
            f"## Original Parsed Resume\n\n{json.dumps(original_resume, indent=2)}\n\n"
            f"## Optimized Resume\n\n{json.dumps(optimized_resume, indent=2)}\n\n"
            "Generate targeted replacements for the changed content."
        )},
    ])

    raw = data.get("replacements", [])
    replacements: list[TextReplacement] = []
    for r in raw:
        old = r.get("old", "").strip()
        new = r.get("new", "").strip()
        if old and new and old != new:
            replacements.append(TextReplacement(old=old, new=new))

    return replacements


def export_node(state: ResumeGraphState) -> dict:
    """Node: format approved resume into text, markdown, and generate replacements."""
    draft = state.get("draft_resume", {})
    raw_text = state.get("raw_resume_text", "")
    parsed_resume = state.get("parsed_resume", {})

    if not draft:
        logger.warning("Formatter: no draft resume to export.")
        return {
            "final_resume_text": "",
            "final_resume_markdown": "",
            "replacements": [],
        }

    plain = _format_plain_text(draft)
    markdown = _format_markdown(draft)

    # Generate replacements for PDF rewriting
    replacements = _generate_replacements(raw_text, parsed_resume, draft)

    logger.info("Formatter: %d-char text, %d-char markdown, %d replacements.",
                len(plain), len(markdown), len(replacements))

    return {
        "final_resume_text": plain,
        "final_resume_markdown": markdown,
        "replacements": replacements,
    }
