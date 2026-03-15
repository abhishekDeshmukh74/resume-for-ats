"""Shared state and schema for the LangGraph resume pipeline."""

from __future__ import annotations

from typing import Annotated, TypedDict

from backend.models import TextReplacement


def _merge_lists(a: list, b: list) -> list:
    """Reducer: extend list *a* with items from *b*."""
    return a + b


def _overwrite(a, b):
    """Reducer: last-write-wins."""
    return b


class AgentState(TypedDict, total=False):
    # ── Inputs (set once) ─────────────────────────────────────────────
    resume_text: str                     # Original plain text from the PDF
    jd_text: str                         # Target job description

    # ── Step 1 → Keyword Extractor ────────────────────────────────────
    jd_keywords: Annotated[list[str], _merge_lists]  # Deduplicated JD keywords
    keyword_categories: Annotated[dict, _overwrite]   # {category: [keywords]}

    # ── Step 2 → Resume Analyzer ──────────────────────────────────────
    resume_sections: Annotated[dict, _overwrite]      # Identified sections & their text
    gap_analysis: Annotated[str, _overwrite]           # Missing keywords / weak areas

    # ── Step 2b → Keyword gap tracking ───────────────────────────────
    missing_keywords: Annotated[list[str], _overwrite]     # Keywords NOT yet in resume
    required_keywords: Annotated[list[str], _overwrite]    # High-priority required skills
    preferred_keywords: Annotated[list[str], _overwrite]   # Nice-to-have skills

    # ── Step 3 → Rewriter ─────────────────────────────────────────────
    raw_replacements: Annotated[list[dict], _merge_lists]  # {"old": ..., "new": ...}

    # ── Step 4 → QA / Deduplication ───────────────────────────────────
    replacements: Annotated[list[TextReplacement], _merge_lists]  # Final clean replacements

    # ── Step 4b → Pre-rewrite ATS Score ────────────────────────────────
    ats_score_before: Annotated[int, _overwrite]

    # ── Step 5 → ATS Scorer ───────────────────────────────────────────
    ats_score: Annotated[int, _overwrite]
    algorithmic_score: Annotated[float, _overwrite]        # Deterministic word-boundary score
    matched_keywords: Annotated[list[str], _overwrite]     # Overwrite (not merge) for re-score
    still_missing_keywords: Annotated[list[str], _overwrite]  # After rewrite, still missing
    rewrite_pass: Annotated[int, _overwrite]               # 0 = first pass, 1 = refinement

    # ── Structured output (for ResumeData) ────────────────────────────
    name: Annotated[str, _overwrite]
    email: Annotated[str | None, _overwrite]
    phone: Annotated[str | None, _overwrite]
    linkedin: Annotated[str | None, _overwrite]
    github: Annotated[str | None, _overwrite]
    location: Annotated[str | None, _overwrite]
    summary: Annotated[str | None, _overwrite]
    skills: Annotated[list[str], _merge_lists]
    experience: Annotated[list[dict], _merge_lists]
    education: Annotated[list[dict], _merge_lists]
    certifications: Annotated[list[dict], _merge_lists]

    # ── Step 7 → PDF Compiler ─────────────────────────────────────────
    resume_file_b64: Annotated[str, _overwrite]    # Original file (b64) passed in for rewriting
    resume_file_type: Annotated[str, _overwrite]   # "pdf" or "tex"
    compiled_pdf_b64: Annotated[str, _overwrite]   # Final compiled/rewritten PDF (b64)
