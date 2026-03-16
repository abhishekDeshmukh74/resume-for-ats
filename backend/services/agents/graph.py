"""LangGraph pipeline — 13-node multi-agent graph for ATS resume optimisation.

Flow:
  START
    → parse_resume        (Intake Parser)
    → analyze_jd          (JD Analyzer)
    → compute_gap         (Gap Analysis)
    → baseline_score      (ATS Scorer — before)
    → parallel(
        optimize_summary,
        optimize_skills,
        optimize_experience
      )
    → merge_resume        (Combine optimized sections)
    → truth_guard         (Verify truthfulness)
    → critic              (Quality gate)
    → conditional:
        if fail → rewrite_router → targeted optimizer
        if pass → final_score
    → export              (Formatter)
    → compile_pdf         (Apply replacements to original file)
  END

Rewrite router sends back to specific agents based on critic feedback.
Maximum 2 revision cycles to prevent infinite loops.

Each node reads from / writes to the shared ResumeGraphState.
Agent I/O is persisted to MongoDB for the /info UI.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import time

from langgraph.graph import StateGraph, END

from backend.models import ResumeData, TextReplacement
from backend.services.agents.state import ResumeGraphState

# Agent imports
from backend.services.agents.intake_parser import parse_resume_node
from backend.services.agents.jd_analyzer import analyze_jd_node
from backend.services.agents.gap_analyzer import compute_gap_node
from backend.services.agents.scorer import baseline_score_node, final_score_node
from backend.services.agents.summary_optimizer import optimize_summary_node
from backend.services.agents.skills_optimizer import optimize_skills_node
from backend.services.agents.bullet_rewriter import optimize_experience_node
from backend.services.agents.truth_guard import truth_guard_node
from backend.services.agents.critic import critic_node
from backend.services.agents.formatter import export_node
from backend.services.agents.pdf_compiler import compile_pdf

from backend.services.db import (
    create_pipeline_run,
    save_agent_step,
    complete_pipeline_run,
    fail_pipeline_run,
    save_compiled_pdf,
)

logger = logging.getLogger(__name__)

# ── Pipeline-run tracking ─────────────────────────────────────────────────

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_run_id", default=None,
)

_AGENT_INPUT_KEYS: dict[str, list[str]] = {
    "parse_resume": ["raw_resume_text"],
    "analyze_jd": ["raw_jd_text"],
    "compute_gap": ["parsed_resume", "parsed_jd"],
    "baseline_score": ["parsed_resume", "parsed_jd"],
    "optimize_summary": ["parsed_resume", "parsed_jd", "gap_report"],
    "optimize_skills": ["parsed_resume", "parsed_jd", "gap_report"],
    "optimize_experience": ["parsed_resume", "parsed_jd", "gap_report"],
    "merge_resume": ["parsed_resume", "optimized_summary", "optimized_skills", "optimized_experience"],
    "truth_guard": ["parsed_resume", "draft_resume"],
    "critic": ["draft_resume", "parsed_jd", "baseline_score", "truth_report"],
    "rewrite_router": ["critic_report", "draft_resume"],
    "final_score": ["draft_resume", "parsed_jd", "truth_report"],
    "export": ["draft_resume", "raw_resume_text", "parsed_resume"],
    "compile_pdf": ["replacements", "resume_file_b64", "resume_file_type"],
}


def _tracked(agent_name: str, agent_fn):
    """Wrap an agent node function to persist its I/O to MongoDB.

    Creates a closure that:
        1. Extracts the relevant input keys from state (defined in
           ``_AGENT_INPUT_KEYS``) for a compact summary.
        2. Calls the actual agent function and times it.
        3. Persists ``(run_id, agent_name, duration_ms, input_summary, result)``
           to MongoDB via ``save_agent_step()``.

    The run ID is retrieved from the ``_current_run_id`` context var, which is
    set in ``generate_resume()`` at the start of each pipeline invocation.

    Args:
        agent_name: Human-readable name used as the MongoDB document key.
        agent_fn:   The actual node function ``(state) -> dict``.

    Returns:
        A wrapper function with the same signature as ``agent_fn``.
    """
    input_keys = _AGENT_INPUT_KEYS.get(agent_name, [])

    def wrapper(state: ResumeGraphState) -> dict:
        run_id = _current_run_id.get()
        input_summary = {k: state.get(k) for k in input_keys}

        start = time.perf_counter()
        result = agent_fn(state)
        duration_ms = int((time.perf_counter() - start) * 1000)

        save_agent_step(run_id, agent_name, duration_ms, input_summary, result)
        return result

    return wrapper


# ── Merge node — combines optimized sections into a draft resume ──────────

def _merge_resume_node(state: ResumeGraphState) -> dict:
    """Merge optimised summary, skills, and experience into a complete draft.

    Takes the original ``parsed_resume`` as a base and overlays:
        * ``optimized_summary`` → replaces ``draft["summary"]``
        * ``optimized_skills``  → replaces ``draft["skills"]``
        * ``optimized_experience.experience`` → rebuilds ``draft["experience"]``
          by extracting the ``rewritten`` field from each bullet dict.
        * ``optimized_experience.projects`` → rebuilds ``draft["projects"]``
          similarly.

    The original ``basics``, ``education``, and ``certifications`` are carried
    through unchanged.

    This node is essential because the three optimizers run independently and
    produce partial outputs — this node combines them into a single coherent
    document for subsequent truth checking and scoring.

    Args:
        state: Pipeline state; reads ``parsed_resume``, ``optimized_summary``,
               ``optimized_skills``, ``optimized_experience``.

    Returns:
        ``{"draft_resume": dict}`` — the complete merged resume.
    """
    parsed = state.get("parsed_resume", {})
    draft = copy.deepcopy(parsed)

    # Replace summary
    optimized_summary = state.get("optimized_summary")
    if optimized_summary:
        draft["summary"] = optimized_summary

    # Replace skills
    optimized_skills = state.get("optimized_skills")
    if optimized_skills:
        draft["skills"] = optimized_skills

    # Replace experience and projects
    optimized_exp = state.get("optimized_experience", {})
    if optimized_exp:
        # Rebuild experience with rewritten bullets
        new_experience = []
        opt_exp_list = optimized_exp.get("experience", [])
        for opt_exp in opt_exp_list:
            exp_entry = {
                "company": opt_exp.get("company", ""),
                "title": opt_exp.get("title", ""),
                "start": opt_exp.get("start", ""),
                "end": opt_exp.get("end", ""),
                "bullets": [
                    b.get("rewritten", b.get("original", ""))
                    if isinstance(b, dict) else b
                    for b in opt_exp.get("bullets", [])
                ],
            }
            new_experience.append(exp_entry)
        if new_experience:
            draft["experience"] = new_experience

        # Rebuild projects with rewritten bullets
        new_projects = []
        opt_proj_list = optimized_exp.get("projects", [])
        for opt_proj in opt_proj_list:
            proj_entry = {
                "name": opt_proj.get("name", ""),
                "stack": opt_proj.get("stack", parsed.get("projects", [{}])[0].get("stack", [])
                                      if opt_proj.get("name") else []),
                "bullets": [
                    b.get("rewritten", b.get("original", ""))
                    if isinstance(b, dict) else b
                    for b in opt_proj.get("bullets", [])
                ],
                "link": opt_proj.get("link"),
            }
            new_projects.append(proj_entry)
        if new_projects:
            draft["projects"] = new_projects

    logger.info("Merge: assembled draft resume with optimized sections.")
    return {"draft_resume": draft}


# ── Rewrite router — targeted re-optimization based on critic feedback ────

def _rewrite_router_node(state: ResumeGraphState) -> dict:
    """Increment the revision counter and prepare for targeted re-optimisation.

    This node is reached when either the truth guard or critic has failed.
    It simply bumps ``revision_count``; the actual routing decision (which
    optimizer to call) is handled by the ``_after_rewrite_router`` conditional
    edge function.

    The revision count is checked by ``_after_critic`` and ``_after_truth_guard``
    to enforce the ``_MAX_REVISIONS`` (2) limit, preventing infinite loops.

    Args:
        state: Pipeline state; reads ``revision_count``.

    Returns:
        ``{"revision_count": int}`` — incremented by 1.
    """
    revision_count = state.get("revision_count", 0) + 1
    logger.info("Rewrite router: starting revision %d.", revision_count)
    return {"revision_count": revision_count}


# ── Conditional edges ─────────────────────────────────────────────────────

_MAX_REVISIONS = 2


def _after_critic(state: ResumeGraphState) -> str:
    """Conditional edge after the critic node.

    Decision logic:
        * If ``critic_report.passed == False`` AND ``revision_count`` is below
          ``_MAX_REVISIONS`` (2): route to ``rewrite_router`` for targeted
          re-optimisation.
        * Otherwise: proceed to ``final_score`` (the quality is acceptable or
          we’ve exhausted revision attempts).

    Returns:
        ``"rewrite_router"`` or ``"final_score"``.
    """
    critic = state.get("critic_report", {})
    revision_count = state.get("revision_count", 0)

    if not critic.get("passed", True) and revision_count < _MAX_REVISIONS:
        logger.info("Critic failed — routing to rewrite (revision %d/%d).",
                     revision_count + 1, _MAX_REVISIONS)
        return "rewrite_router"
    return "final_score"


def _after_truth_guard(state: ResumeGraphState) -> str:
    """Conditional edge after the truth guard node.

    Decision logic:
        * If ``truth_report.supported == False`` AND ``revision_count`` is
          below ``_MAX_REVISIONS``: route to ``rewrite_router`` to fix
          truthfulness violations.
        * Otherwise: proceed to ``critic`` for quality review.

    Returns:
        ``"rewrite_router"`` or ``"critic"``.
    """
    truth = state.get("truth_report", {})
    revision_count = state.get("revision_count", 0)

    if not truth.get("supported", True) and revision_count < _MAX_REVISIONS:
        logger.info("Truth guard failed — routing to rewrite.")
        return "rewrite_router"
    return "critic"


def _after_rewrite_router(state: ResumeGraphState) -> str:
    """Conditional edge after the rewrite router — pick which optimizer to run.

    Reads ``critic_report.revision_instructions`` to determine which
    section(s) need revision.  If multiple sections need work, routes to
    the highest-impact one first (experience > summary > skills).

    Falls back to ``optimize_experience`` if no specific instructions exist.

    Returns:
        ``"optimize_experience"``, ``"optimize_summary"``, or
        ``"optimize_skills"``.
    """
    critic = state.get("critic_report", {})
    instructions = critic.get("revision_instructions", {})

    # Check which sections need revision
    needs_summary = instructions.get("summary") is not None
    needs_skills = instructions.get("skills") is not None
    needs_experience = instructions.get("experience") is not None

    # Route to the most impactful agent
    # If multiple need revision, go to experience first (highest impact)
    if needs_experience:
        return "optimize_experience"
    if needs_summary:
        return "optimize_summary"
    if needs_skills:
        return "optimize_skills"

    # Default: re-run experience optimizer
    return "optimize_experience"


# ── Build the graph ───────────────────────────────────────────────────────

_builder = StateGraph(ResumeGraphState)

# Stage 1: Ingestion
_builder.add_node("parse_resume", _tracked("parse_resume", parse_resume_node))

# Stage 2: JD Analysis
_builder.add_node("analyze_jd", _tracked("analyze_jd", analyze_jd_node))

# Stage 3: Diagnostics
_builder.add_node("compute_gap", _tracked("compute_gap", compute_gap_node))
_builder.add_node("baseline_score", _tracked("baseline_score", baseline_score_node))

# Stage 4: Content Optimization (run after diagnostics)
_builder.add_node("optimize_summary", _tracked("optimize_summary", optimize_summary_node))
_builder.add_node("optimize_skills", _tracked("optimize_skills", optimize_skills_node))
_builder.add_node("optimize_experience", _tracked("optimize_experience", optimize_experience_node))

# Stage 5: Merge + Safety
_builder.add_node("merge_resume", _tracked("merge_resume", _merge_resume_node))
_builder.add_node("truth_guard", _tracked("truth_guard", truth_guard_node))

# Stage 6: Review
_builder.add_node("critic", _tracked("critic", critic_node))
_builder.add_node("rewrite_router", _tracked("rewrite_router", _rewrite_router_node))

# Stage 7: Rescore
_builder.add_node("final_score", _tracked("final_score", final_score_node))

# Stage 8: Export
_builder.add_node("export", _tracked("export", export_node))
_builder.add_node("compile_pdf", _tracked("compile_pdf", compile_pdf))

# ── Edges ─────────────────────────────────────────────────────────────────

# Linear backbone: parse → analyze_jd → compute_gap → baseline_score
_builder.set_entry_point("parse_resume")
_builder.add_edge("parse_resume", "analyze_jd")
_builder.add_edge("analyze_jd", "compute_gap")
_builder.add_edge("compute_gap", "baseline_score")

# After baseline score → run all 3 optimizers in sequence
# (LangGraph does not natively support parallel execution within a single
#  graph invocation, so we chain them. Each has independent inputs.)
_builder.add_edge("baseline_score", "optimize_summary")
_builder.add_edge("optimize_summary", "optimize_skills")
_builder.add_edge("optimize_skills", "optimize_experience")

# Merge → Truth Guard → conditional
_builder.add_edge("optimize_experience", "merge_resume")
_builder.add_edge("merge_resume", "truth_guard")

# Truth guard → conditional: pass → critic, fail → rewrite_router
_builder.add_conditional_edges("truth_guard", _after_truth_guard, {
    "critic": "critic",
    "rewrite_router": "rewrite_router",
})

# Critic → conditional: pass → final_score, fail → rewrite_router
_builder.add_conditional_edges("critic", _after_critic, {
    "final_score": "final_score",
    "rewrite_router": "rewrite_router",
})

# Rewrite router → targeted optimizer → merge → truth_guard (loop)
_builder.add_conditional_edges("rewrite_router", _after_rewrite_router, {
    "optimize_summary": "optimize_summary",
    "optimize_skills": "optimize_skills",
    "optimize_experience": "optimize_experience",
})

# Final score → export → compile_pdf → END
_builder.add_edge("final_score", "export")
_builder.add_edge("export", "compile_pdf")
_builder.add_edge("compile_pdf", END)

graph = _builder.compile()


# ── Public API ────────────────────────────────────────────────────────────

def generate_resume(
    resume_text: str,
    jd_text: str,
    resume_file_b64: str = "",
    resume_file_type: str = "pdf",
) -> tuple[ResumeData, str]:
    """Run the full 13-node multi-agent pipeline.

    This is the **public API** of the agents package, re-exported by
    ``__init__.py`` and called by the ``/api/generate-resume`` endpoint.

    Pipeline flow::

        parse_resume → analyze_jd → compute_gap → baseline_score
        → optimize_summary → optimize_skills → optimize_experience
        → merge_resume → truth_guard → [critic] → [final_score]
        → export → compile_pdf → END

    The truth guard and critic nodes can route back to targeted optimizers
    for up to 2 revision cycles.

    Side effects:
        * Creates a MongoDB pipeline run on entry.
        * Each agent’s I/O is logged to MongoDB for the ``/info`` UI.
        * Pipeline run is completed/failed on exit.

    Args:
        resume_text:     Plain text extracted from the uploaded resume.
        jd_text:         Raw job description text.
        resume_file_b64: Base64-encoded original resume file (PDF or .tex).
        resume_file_type: ``"pdf"`` or ``"tex"``.

    Returns:
        Tuple of ``(ResumeData, compiled_pdf_b64)`` where ``ResumeData``
        contains the structured resume fields and ATS scores, and
        ``compiled_pdf_b64`` is the base64-encoded rewritten PDF.

    Raises:
        RuntimeError: If the LLM fails after all retries.
        Exception: Any unhandled error during pipeline execution (logged
                   to MongoDB before re-raising).
    """
    logger.info("Starting LangGraph multi-agent pipeline (13 nodes).")

    run_id = create_pipeline_run(resume_text, jd_text)
    token = _current_run_id.set(run_id)

    initial_state: ResumeGraphState = {
        "raw_resume_text": resume_text,
        "raw_jd_text": jd_text,
        "parsed_resume": {},
        "parsed_jd": {},
        "gap_report": {},
        "baseline_score": {},
        "optimized_summary": "",
        "optimized_skills": {},
        "optimized_experience": [],
        "draft_resume": {},
        "truth_report": {},
        "critic_report": {},
        "final_score": {},
        "revision_count": 0,
        "max_revisions": _MAX_REVISIONS,
        "final_resume_text": "",
        "final_resume_markdown": "",
        "resume_file_b64": resume_file_b64,
        "resume_file_type": resume_file_type,
        "compiled_pdf_b64": "",
        "replacements": [],
    }

    try:
        final = graph.invoke(initial_state)

        # Extract structured data from the draft resume
        draft = final.get("draft_resume", {})
        basics = draft.get("basics", {})
        final_score_data = final.get("final_score", {})
        baseline_data = final.get("baseline_score", {})

        # Flatten skills for ResumeData
        skills_dict = draft.get("skills", {})
        flat_skills: list[str] = []
        if isinstance(skills_dict, dict):
            for items in skills_dict.values():
                if isinstance(items, list):
                    flat_skills.extend(items)
        elif isinstance(skills_dict, list):
            flat_skills = skills_dict

        # Build experience items for ResumeData
        experience_items = []
        for exp in draft.get("experience", []):
            experience_items.append({
                "job_title": exp.get("title", ""),
                "company": exp.get("company", ""),
                "location": None,
                "start_date": exp.get("start", ""),
                "end_date": exp.get("end", ""),
                "bullets": exp.get("bullets", []),
            })

        # Build education items
        education_items = []
        for edu in draft.get("education", []):
            education_items.append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", ""),
                "location": edu.get("location"),
                "graduation_date": edu.get("graduation_date", ""),
                "details": edu.get("details"),
            })

        # Build certification items
        cert_items = []
        for cert in draft.get("certifications", []):
            cert_items.append({
                "name": cert.get("name", ""),
                "issuer": cert.get("issuer"),
                "date": cert.get("date"),
            })

        # Get replacements
        replacements = final.get("replacements", [])
        if replacements and isinstance(replacements[0], dict):
            replacements = [
                TextReplacement(old=r["old"], new=r["new"])
                for r in replacements
                if r.get("old") and r.get("new")
            ]

        # Keyword coverage from final score
        keyword_data = final_score_data.get("keyword_coverage", {})
        matched = keyword_data.get("covered", [])

        resume = ResumeData(
            name=basics.get("name", ""),
            email=basics.get("email"),
            phone=basics.get("phone"),
            linkedin=basics.get("linkedin"),
            github=basics.get("github"),
            location=basics.get("location"),
            summary=draft.get("summary"),
            skills=flat_skills,
            experience=experience_items,
            education=education_items,
            certifications=cert_items,
            ats_score_before=int(baseline_data.get("overall_score", 0)),
            ats_score=int(final_score_data.get("overall_score", 0)),
            matched_keywords=matched,
            replacements=replacements,
        )

        compiled_pdf_b64: str = final.get("compiled_pdf_b64", "")
        n = len(resume.replacements)

        logger.info(
            "Pipeline complete: %d replacements, baseline=%s, final=%s, revisions=%d.",
            n, resume.ats_score_before, resume.ats_score,
            final.get("revision_count", 0),
        )

        complete_pipeline_run(run_id, {
            "ats_score_before": resume.ats_score_before,
            "ats_score": resume.ats_score,
            "matched_keywords": resume.matched_keywords,
            "replacements_count": n,
            "name": resume.name,
            "revisions": final.get("revision_count", 0),
            "truth_supported": final.get("truth_report", {}).get("supported", True),
            "critic_passed": final.get("critic_report", {}).get("passed", True),
        })

        save_compiled_pdf(run_id, compiled_pdf_b64)
        return resume, compiled_pdf_b64

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        raise
    finally:
        _current_run_id.reset(token)
