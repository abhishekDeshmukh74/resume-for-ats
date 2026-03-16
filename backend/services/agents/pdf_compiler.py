"""pdf_compiler agent — applies AI replacements to the original resume file
and produces a final PDF.

For PDF uploads the existing PyMuPDF-based rewriter is used; for LaTeX
(.tex) uploads the source is patched and compiled with xelatex / pdflatex.
"""

from __future__ import annotations

import base64
import logging

from backend.models import ResumeData, TextReplacement
from backend.services.agents.state import ResumeGraphState
from backend.services.latex_rewriter import rewrite_tex
from backend.services.rewriter import rewrite_pdf

logger = logging.getLogger(__name__)


def compile_pdf(state: ResumeGraphState) -> dict:
    """Apply replacements to the original file and return base64-encoded PDF."""
    file_b64: str = state.get("resume_file_b64", "")
    file_type: str = state.get("resume_file_type", "pdf")
    raw_replacements = state.get("replacements", [])

    if not file_b64:
        logger.warning("compile_pdf: no original file bytes available, skipping PDF compilation.")
        return {"compiled_pdf_b64": ""}

    file_bytes = base64.b64decode(file_b64)

    # Convert raw dicts to TextReplacement objects
    replacements = []
    for r in raw_replacements:
        if isinstance(r, dict) and r.get("old") and r.get("new"):
            replacements.append(TextReplacement(old=r["old"], new=r["new"]))
        elif isinstance(r, TextReplacement):
            replacements.append(r)

    # Build a minimal ResumeData to carry replacements into the rewriter
    draft = state.get("draft_resume", {})
    name = draft.get("basics", {}).get("name", "")
    resume = ResumeData(name=name, replacements=replacements)

    if file_type == "tex":
        logger.info("compile_pdf: compiling LaTeX → PDF.")
        pdf_bytes = rewrite_tex(file_bytes, resume)
    else:
        logger.info("compile_pdf: rewriting PDF with PyMuPDF.")
        pdf_bytes = rewrite_pdf(file_bytes, resume)

    return {"compiled_pdf_b64": base64.b64encode(pdf_bytes).decode()}
