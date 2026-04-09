"""LangGraph pipeline — wires agents into a graph with parallel rewriting and refinement loop.

Flow:
  extract_keywords → analyse_resume → score_before
  → ┌ rewrite_skills  ┐
    │ rewrite_summary  │  (parallel)
    └ rewrite_experience┘
  → qa_deduplicate → score_extract
  → [if score < 90 & pass == 0: refine_rewrite → refine_qa → score_extract]
  → compile_pdf

Each node reads from / writes to the shared AgentState.
Agent I/O is persisted to MongoDB for the /info UI.
"""

from __future__ import annotations

import contextvars
import logging
import time

from langgraph.graph import StateGraph, END

from backend.models import ResumeData
from backend.services.agents.state import AgentState
from backend.services.agents.keyword_extractor import extract_keywords
from backend.services.agents.resume_analyser import analyse_resume
from backend.services.agents.skills_rewriter import rewrite_skills
from backend.services.agents.summary_rewriter import rewrite_summary
from backend.services.agents.experience_rewriter import rewrite_experience
from backend.services.agents.refinement_agent import refine_rewrite
from backend.services.agents.qa_agent import qa_and_deduplicate
from backend.services.agents.scorer import score_before_rewrite, score_and_extract
from backend.services.agents.pdf_compiler import compile_pdf
from backend.services.db import (
    create_pipeline_run,
    save_agent_step,
    complete_pipeline_run,
    fail_pipeline_run,
    save_compiled_pdf,
)

logger = logging.getLogger(__name__)

# ── Pipeline-run tracking (thread-safe via contextvars) ───────────────────

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_run_id", default=None,
)

# Which state keys each agent reads (used for input summaries)
_AGENT_INPUT_KEYS: dict[str, list[str]] = {
    "extract_keywords": ["jd_text"],
    "analyse_resume": ["resume_text", "jd_keywords"],
    "score_before": ["resume_text", "jd_keywords"],
    "rewrite_skills": ["resume_text", "resume_sections", "keyword_categories", "missing_keywords", "required_keywords"],
    "rewrite_summary": ["resume_text", "resume_sections", "keyword_categories", "missing_keywords", "required_keywords", "gap_analysis"],
    "rewrite_experience": ["resume_text", "resume_sections", "keyword_categories", "missing_keywords", "required_keywords", "preferred_keywords", "gap_analysis"],
    "qa_deduplicate": ["resume_text", "raw_replacements", "jd_keywords"],
    "score_extract": ["resume_text", "jd_text", "jd_keywords", "replacements"],
    "refine_rewrite": ["resume_text", "replacements", "still_missing_keywords", "required_keywords", "keyword_categories"],
    "refine_qa": ["resume_text", "raw_replacements", "jd_keywords"],
    "compile_pdf": ["replacements", "resume_file_b64", "resume_file_type"],
}


def _tracked(agent_name: str, agent_fn):
    """Wrap an agent function to persist its I/O to MongoDB."""
    input_keys = _AGENT_INPUT_KEYS.get(agent_name, [])

    def wrapper(state: AgentState) -> dict:
        run_id = _current_run_id.get()

        # Build input summary
        input_summary: dict = {}
        for key in input_keys:
            val = state.get(key)
            if isinstance(val, list):
                input_summary[key] = val
            else:
                input_summary[key] = val

        start = time.perf_counter()
        result = agent_fn(state)
        duration_ms = int((time.perf_counter() - start) * 1000)

        save_agent_step(run_id, agent_name, duration_ms, input_summary, result)
        return result

    return wrapper


# ── Build the graph ───────────────────────────────────────────────────────

_ATS_TARGET = 90  # Score threshold; below this triggers a refinement pass


def _should_refine(state: AgentState) -> str:
    """Conditional edge: decide whether to refine or go straight to compile."""
    score = state.get("ats_score", 0)
    pass_num = state.get("rewrite_pass", 0)
    missing = state.get("still_missing_keywords", [])

    if score < _ATS_TARGET and pass_num == 0 and len(missing) > 0:
        logger.info(
            "Score %d < %d with %d missing keywords — running refinement pass.",
            score, _ATS_TARGET, len(missing),
        )
        return "refine_rewrite"
    return "compile_pdf"


_builder = StateGraph(AgentState)

_builder.add_node("extract_keywords", _tracked("extract_keywords", extract_keywords))
_builder.add_node("analyse_resume", _tracked("analyse_resume", analyse_resume))
_builder.add_node("score_before", _tracked("score_before", score_before_rewrite))
_builder.add_node("rewrite_skills", _tracked("rewrite_skills", rewrite_skills))
_builder.add_node("rewrite_summary", _tracked("rewrite_summary", rewrite_summary))
_builder.add_node("rewrite_experience", _tracked("rewrite_experience", rewrite_experience))
_builder.add_node("qa_deduplicate", _tracked("qa_deduplicate", qa_and_deduplicate))
_builder.add_node("score_extract", _tracked("score_extract", score_and_extract))
_builder.add_node("refine_rewrite", _tracked("refine_rewrite", refine_rewrite))
_builder.add_node("refine_qa", _tracked("refine_qa", qa_and_deduplicate))
_builder.add_node("compile_pdf", _tracked("compile_pdf", compile_pdf))

_builder.set_entry_point("extract_keywords")
_builder.add_edge("extract_keywords", "analyse_resume")
_builder.add_edge("analyse_resume", "score_before")
# Fan-out: run all three section rewriters in parallel
_builder.add_edge("score_before", "rewrite_skills")
_builder.add_edge("score_before", "rewrite_summary")
_builder.add_edge("score_before", "rewrite_experience")
# Fan-in: all three must finish before QA
_builder.add_edge("rewrite_skills", "qa_deduplicate")
_builder.add_edge("rewrite_summary", "qa_deduplicate")
_builder.add_edge("rewrite_experience", "qa_deduplicate")
_builder.add_edge("qa_deduplicate", "score_extract")

# Conditional: refine if score < 90, otherwise go to compile
_builder.add_conditional_edges("score_extract", _should_refine, {
    "refine_rewrite": "refine_rewrite",
    "compile_pdf": "compile_pdf",
})
_builder.add_edge("refine_rewrite", "refine_qa")
_builder.add_edge("refine_qa", "compile_pdf")

_builder.add_edge("compile_pdf", END)

graph = _builder.compile()


# ── Preview-only graph (no PDF compilation) ───────────────────────────────

_preview_builder = StateGraph(AgentState)

_preview_builder.add_node("extract_keywords", _tracked("extract_keywords", extract_keywords))
_preview_builder.add_node("analyse_resume", _tracked("analyse_resume", analyse_resume))
_preview_builder.add_node("score_before", _tracked("score_before", score_before_rewrite))
_preview_builder.add_node("rewrite_skills", _tracked("rewrite_skills", rewrite_skills))
_preview_builder.add_node("rewrite_summary", _tracked("rewrite_summary", rewrite_summary))
_preview_builder.add_node("rewrite_experience", _tracked("rewrite_experience", rewrite_experience))
_preview_builder.add_node("qa_deduplicate", _tracked("qa_deduplicate", qa_and_deduplicate))
_preview_builder.add_node("score_extract", _tracked("score_extract", score_and_extract))
_preview_builder.add_node("refine_rewrite", _tracked("refine_rewrite", refine_rewrite))
_preview_builder.add_node("refine_qa", _tracked("refine_qa", qa_and_deduplicate))

_preview_builder.set_entry_point("extract_keywords")
_preview_builder.add_edge("extract_keywords", "analyse_resume")
_preview_builder.add_edge("analyse_resume", "score_before")
_preview_builder.add_edge("score_before", "rewrite_skills")
_preview_builder.add_edge("score_before", "rewrite_summary")
_preview_builder.add_edge("score_before", "rewrite_experience")
_preview_builder.add_edge("rewrite_skills", "qa_deduplicate")
_preview_builder.add_edge("rewrite_summary", "qa_deduplicate")
_preview_builder.add_edge("rewrite_experience", "qa_deduplicate")
_preview_builder.add_edge("qa_deduplicate", "score_extract")
_preview_builder.add_conditional_edges("score_extract", _should_refine, {
    "refine_rewrite": "refine_rewrite",
    "compile_pdf": END,  # Stop here — no PDF compilation
})
_preview_builder.add_edge("refine_rewrite", "refine_qa")
_preview_builder.add_edge("refine_qa", END)

_preview_graph = _preview_builder.compile()


# ── Public API ────────────────────────────────────────────────────────────

def generate_resume(
    resume_text: str,
    jd_text: str,
    resume_file_b64: str = "",
    resume_file_type: str = "pdf",
) -> tuple[ResumeData, str]:
    """Run the full multi-agent pipeline.

    Returns:
        (ResumeData, compiled_pdf_b64)  where *compiled_pdf_b64* is the
        base64-encoded rewritten PDF produced by the final compile_pdf node.
    """
    logger.info("Starting LangGraph pipeline (9 agents).")

    run_id = create_pipeline_run(resume_text, jd_text)
    token = _current_run_id.set(run_id)

    initial_state: AgentState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "jd_keywords": [],
        "keyword_categories": {},
        "resume_sections": {},
        "gap_analysis": "",
        "missing_keywords": [],
        "required_keywords": [],
        "preferred_keywords": [],
        "raw_replacements": [],
        "replacements": [],
        "ats_score_before": 0,
        "ats_score": 0,
        "algorithmic_score": 0.0,
        "matched_keywords": [],
        "still_missing_keywords": [],
        "rewrite_pass": 0,
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "location": None,
        "summary": None,
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "resume_file_b64": resume_file_b64,
        "resume_file_type": resume_file_type,
        "compiled_pdf_b64": "",
    }

    try:
        final = graph.invoke(initial_state)

        resume = ResumeData(
            name=final.get("name", ""),
            email=final.get("email"),
            phone=final.get("phone"),
            linkedin=final.get("linkedin"),
            github=final.get("github"),
            location=final.get("location"),
            summary=final.get("summary"),
            skills=final.get("skills", []),
            experience=final.get("experience", []),
            education=final.get("education", []),
            certifications=final.get("certifications", []),
            ats_score_before=final.get("ats_score_before", 0),
            ats_score=final.get("ats_score", 0),
            matched_keywords=final.get("matched_keywords", []),
            replacements=final.get("replacements", []),
        )

        n = len(resume.replacements)
        logger.info("Pipeline complete: %d replacements, ATS score=%s.",
                     n, resume.ats_score)

        compiled_pdf_b64: str = final.get("compiled_pdf_b64", "")

        complete_pipeline_run(run_id, {
            "ats_score_before": resume.ats_score_before,
            "ats_score": resume.ats_score,
            "matched_keywords": resume.matched_keywords,
            "replacements_count": n,
            "name": resume.name,
        })

        save_compiled_pdf(run_id, compiled_pdf_b64)

        return resume, compiled_pdf_b64

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        raise
    finally:
        _current_run_id.reset(token)


def _make_initial_state(resume_text: str, jd_text: str) -> AgentState:
    """Build the initial AgentState for any pipeline variant."""
    return {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "jd_keywords": [],
        "keyword_categories": {},
        "resume_sections": {},
        "gap_analysis": "",
        "missing_keywords": [],
        "required_keywords": [],
        "preferred_keywords": [],
        "raw_replacements": [],
        "replacements": [],
        "ats_score_before": 0,
        "ats_score": 0,
        "algorithmic_score": 0.0,
        "matched_keywords": [],
        "still_missing_keywords": [],
        "rewrite_pass": 0,
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "location": None,
        "summary": None,
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "resume_file_b64": "",
        "resume_file_type": "pdf",
        "compiled_pdf_b64": "",
    }


def preview_resume(resume_text: str, jd_text: str) -> dict:
    """Run the pipeline WITHOUT compiling a PDF.

    Returns a dict suitable for ``PreviewResponse``: replacements, scores,
    matched/missing keywords.
    """
    logger.info("Starting preview pipeline (no PDF compilation).")

    run_id = create_pipeline_run(resume_text, jd_text)
    token = _current_run_id.set(run_id)

    initial_state = _make_initial_state(resume_text, jd_text)

    try:
        final = _preview_graph.invoke(initial_state)

        replacements = final.get("replacements", [])
        result = {
            "replacements": replacements,
            "ats_score_before": final.get("ats_score_before", 0),
            "ats_score": final.get("ats_score", 0),
            "matched_keywords": final.get("matched_keywords", []),
            "still_missing_keywords": final.get("still_missing_keywords", []),
        }

        complete_pipeline_run(run_id, {
            "ats_score_before": result["ats_score_before"],
            "ats_score": result["ats_score"],
            "matched_keywords": result["matched_keywords"],
            "replacements_count": len(replacements),
            "mode": "preview",
        })

        logger.info("Preview complete: %d replacements, ATS %d→%d.",
                     len(replacements), result["ats_score_before"], result["ats_score"])
        return result

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        raise
    finally:
        _current_run_id.reset(token)


def stream_preview_resume(resume_text: str, jd_text: str):
    """Generator that yields (event_type, data) tuples as each preview-pipeline
    agent completes.  Call this from an SSE endpoint."""
    run_id = create_pipeline_run(resume_text, jd_text)
    token = _current_run_id.set(run_id)
    initial_state = _make_initial_state(resume_text, jd_text)
    accumulated: dict = dict(initial_state)

    try:
        for event in _preview_graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in event.items():
                accumulated.update(update)

                progress: dict = {"agent": node_name}
                if node_name == "score_before":
                    progress["ats_score_before"] = update.get("ats_score_before", 0)
                elif node_name == "score_extract":
                    progress["ats_score"] = update.get("ats_score", 0)
                    progress["matched_keywords"] = update.get("matched_keywords", [])
                    progress["still_missing_keywords"] = update.get("still_missing_keywords", [])
                elif node_name in ("rewrite_skills", "rewrite_summary", "rewrite_experience"):
                    progress["replacements_count"] = len(update.get("raw_replacements", []))
                elif node_name == "qa_deduplicate":
                    progress["replacements_count"] = len(update.get("replacements", []))

                yield ("agent_complete", progress)

        replacements = accumulated.get("replacements", [])
        result = {
            "replacements": replacements,
            "ats_score_before": accumulated.get("ats_score_before", 0),
            "ats_score": accumulated.get("ats_score", 0),
            "matched_keywords": accumulated.get("matched_keywords", []),
            "still_missing_keywords": accumulated.get("still_missing_keywords", []),
        }
        complete_pipeline_run(run_id, {
            "ats_score_before": result["ats_score_before"],
            "ats_score": result["ats_score"],
            "matched_keywords": result["matched_keywords"],
            "replacements_count": len(replacements),
            "mode": "preview",
        })
        logger.info("Preview stream complete: %d replacements, ATS %d→%d.",
                     len(replacements), result["ats_score_before"], result["ats_score"])
        yield ("complete", result)

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        yield ("error", {"detail": str(exc)})
    finally:
        _current_run_id.reset(token)


def confirm_resume(
    resume_text: str,
    replacements: list[dict],
    resume_file_b64: str,
    resume_file_type: str = "pdf",
) -> str:
    """Apply user-approved replacements and compile the final PDF.

    Returns the base64-encoded rewritten PDF.
    """
    from backend.models import TextReplacement as TR

    logger.info("Confirming %d replacements → compiling PDF.", len(replacements))

    typed = [TR(old=r["old"], new=r["new"]) for r in replacements]
    state: AgentState = _make_initial_state(resume_text, "")
    state["replacements"] = typed
    state["resume_file_b64"] = resume_file_b64
    state["resume_file_type"] = resume_file_type

    result = compile_pdf(state)
    compiled = result.get("compiled_pdf_b64", "")
    if not compiled:
        raise RuntimeError("PDF compilation produced no output.")
    return compiled
