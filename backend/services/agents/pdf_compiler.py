"""pdf_compiler agent — applies AI replacements to the original resume file
and produces a final PDF.

For PDF uploads the existing PyMuPDF-based rewriter is used; for LaTeX
(.tex) uploads the source is patched and compiled with xelatex / pdflatex.
"""

from __future__ import annotations

import base64
import logging

from backend.models import ResumeData
from backend.services.agents.state import AgentState
from backend.services.latex_rewriter import rewrite_tex
from backend.services.rewriter import rewrite_pdf

logger = logging.getLogger(__name__)


def compile_pdf(state: AgentState) -> dict:
    """Apply replacements to the original file and return base64-encoded PDF."""
    file_b64: str = state.get("resume_file_b64", "")
    file_type: str = state.get("resume_file_type", "pdf")
    replacements = state.get("replacements", [])

    if not file_b64:
        logger.warning("compile_pdf: no original file bytes available, skipping PDF compilation.")
        return {"compiled_pdf_b64": ""}

    file_bytes = base64.b64decode(file_b64)

    # Minimal ResumeData used only to carry replacements into the rewriter
    resume = ResumeData(name=state.get("name", ""), replacements=replacements)

    if file_type == "tex":
        logger.info("compile_pdf: compiling LaTeX → PDF.")
        pdf_bytes = rewrite_tex(file_bytes, resume)
    else:
        logger.info("compile_pdf: rewriting PDF with PyMuPDF.")
        pdf_bytes = rewrite_pdf(file_bytes, resume)

    return {"compiled_pdf_b64": base64.b64encode(pdf_bytes).decode()}
