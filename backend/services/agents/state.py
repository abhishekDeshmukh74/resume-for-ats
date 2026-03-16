"""Shared state for the LangGraph resume pipeline.

This module defines ``ResumeGraphState``, the TypedDict that every agent in
the 13-node LangGraph pipeline reads from and writes to.  It is the *single
source of truth* for inter-agent communication.

Design decisions
~~~~~~~~~~~~~~~~
* Every field uses ``Annotated[T, _overwrite]`` so that the *last write wins*.
  There are no list-accumulation reducers — each agent is responsible for
  returning the complete value it owns.
* Fields are grouped by pipeline stage so that it is easy to see which agent
  produces which data.
* All fields are ``total=False`` (optional) because the graph is invoked with
  a partial dict and agents populate keys incrementally.

Pipeline stage → state key mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Intake Parser      → ``parsed_resume``
2. JD Analyzer        → ``parsed_jd``
3. Gap Analyzer       → ``gap_report``
4. Baseline Scorer    → ``baseline_score``
5. Summary Optimizer  → ``optimized_summary``
6. Skills Optimizer   → ``optimized_skills``
7. Bullet Rewriter    → ``optimized_experience``
8. Merge Node         → ``draft_resume``
9. Truth Guard        → ``truth_report``
10. Critic            → ``critic_report``
11. Final Scorer      → ``final_score``
12. Formatter/Export  → ``final_resume_text``, ``final_resume_markdown``, ``replacements``
13. PDF Compiler      → ``compiled_pdf_b64``
"""

from __future__ import annotations

from typing import Annotated, TypedDict


def _overwrite(a, b):
    """LangGraph reducer: always keep the latest value (last-write-wins).

    Args:
        a: The previous value already in the state.
        b: The new value returned by the current agent node.

    Returns:
        ``b`` unconditionally.
    """
    return b


class ResumeGraphState(TypedDict, total=False):
    """Shared pipeline state passed between all LangGraph agent nodes.

    Every node function receives the full state dict and returns a *partial*
    dict with only the keys it wants to update.  LangGraph merges the
    partial dict back using the ``_overwrite`` reducer.

    Example::

        def my_node(state: ResumeGraphState) -> dict:
            # Read what you need
            resume = state["parsed_resume"]
            # Return only what you produce
            return {"gap_report": {...}}

    Key sections (in pipeline order):

    * **Raw Inputs** — provided by the caller of ``generate_resume()``.
    * **Parsed Structures** — produced by the Intake Parser and JD Analyzer.
    * **Diagnostics** — gap analysis and baseline ATS score.
    * **Optimization** — optimized summary, skills, and experience bullets.
    * **Draft** — merged resume combining all optimized sections.
    * **Safety** — truth guard and critic reports.
    * **Final Score** — ATS score after optimization.
    * **Export** — plain text, markdown, and PDF replacement mappings.
    * **Revision Control** — loop counter for rewrite cycles (max 2).
    """

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
