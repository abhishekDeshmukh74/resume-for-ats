"""PDF Compiler agent — applies AI-generated text replacements to the original
resume file and produces a final PDF.

This is the last agent in the pipeline, responsible for producing the
downloadable output.

Behaviour by file type:
    * **PDF uploads** — ``rewriter.py`` uses PyMuPDF to search for ``old``
      text in the PDF, redact it, and re-insert ``new`` text in-place,
      preserving fonts, colours, and layout.
    * **LaTeX uploads** (.tex) — ``latex_rewriter.py`` patches the ``.tex``
      source (with flexible pattern matching for line wrapping and LaTeX
      escapes), then compiles to PDF via xelatex / pdflatex.

Graph position:
    ``export`` → **compile_pdf** → ``END``

State reads:
    ``resume_file_b64``, ``resume_file_type``, ``replacements``,
    ``draft_resume``

State writes:
    ``compiled_pdf_b64`` — base64-encoded bytes of the rewritten PDF.
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
    """LangGraph node: apply text replacements to the original file and produce PDF.

    Workflow:
        1. Decode the original file from base64.
        2. Convert raw replacement dicts to ``TextReplacement`` Pydantic objects.
        3. Build a minimal ``ResumeData`` to carry replacements into the rewriter.
        4. Dispatch to ``rewrite_pdf()`` (PyMuPDF) or ``rewrite_tex()`` (LaTeX)
           based on ``resume_file_type``.
        5. Return the compiled PDF as base64.

    If no ``resume_file_b64`` is provided (e.g., text-only input), returns an
    empty string and logs a warning.

    Args:
        state: Pipeline state; reads ``resume_file_b64``, ``resume_file_type``,
               ``replacements``, ``draft_resume``.

    Returns:
        ``{"compiled_pdf_b64": str}`` — base64-encoded PDF bytes, or empty
        string if no input file was provided.
    """
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
