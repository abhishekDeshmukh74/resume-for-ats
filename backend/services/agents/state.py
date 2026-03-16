"""Shared state for the LangGraph resume pipeline.

Structured state is the backbone of the multi-agent system.
Each agent reads from and writes to specific keys in this shared dict.
"""

from __future__ import annotations

from typing import Annotated, TypedDict


def _overwrite(a, b):
    return b


class ResumeGraphState(TypedDict, total=False):
    # ── Raw Inputs ────────────────────────────────────────────────────
    raw_resume_text: str
    raw_jd_text: str

    # ── Stage 1: Intake Parser → structured resume ────────────────────
    parsed_resume: Annotated[dict, _overwrite]

    # ── Stage 2: JD Analyzer → structured JD signals ──────────────────
    parsed_jd: Annotated[dict, _overwrite]

    # ── Stage 3: Gap Analysis ─────────────────────────────────────────
    gap_report: Annotated[dict, _overwrite]

    # ── Stage 3: Baseline ATS Score ───────────────────────────────────
    baseline_score: Annotated[dict, _overwrite]

    # ── Stage 4: Content Optimization ─────────────────────────────────
    optimized_summary: Annotated[str, _overwrite]
    optimized_skills: Annotated[dict, _overwrite]
    optimized_experience: Annotated[list, _overwrite]

    # ── Stage 5: Merged Draft ─────────────────────────────────────────
    draft_resume: Annotated[dict, _overwrite]

    # ── Stage 5: Truth Guard ──────────────────────────────────────────
    truth_report: Annotated[dict, _overwrite]

    # ── Stage 6: Critic / Reviewer ────────────────────────────────────
    critic_report: Annotated[dict, _overwrite]

    # ── Stage 7: Final ATS Score ──────────────────────────────────────
    final_score: Annotated[dict, _overwrite]

    # ── Revision Control ──────────────────────────────────────────────
    revision_count: Annotated[int, _overwrite]
    max_revisions: Annotated[int, _overwrite]

    # ── Stage 8: Export Formats ───────────────────────────────────────
    final_resume_text: Annotated[str, _overwrite]
    final_resume_markdown: Annotated[str, _overwrite]

    # ── PDF Compilation (carried from original) ───────────────────────
    resume_file_b64: Annotated[str, _overwrite]
    resume_file_type: Annotated[str, _overwrite]
    compiled_pdf_b64: Annotated[str, _overwrite]

    # ── Replacements (for PDF rewriting) ──────────────────────────────
    replacements: Annotated[list, _overwrite]
