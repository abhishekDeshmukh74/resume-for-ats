"""LangGraph pipeline — wires all agents into a sequential graph.

Flow:
  extract_keywords → analyse_resume → score_before → rewrite_sections → qa_deduplicate → score_extract

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
from backend.services.agents.rewriter_agent import rewrite_sections
from backend.services.agents.qa_agent import qa_and_deduplicate
from backend.services.agents.scorer import score_before_rewrite, score_and_extract
from backend.services.agents.pdf_compiler import compile_pdf
from backend.services.db import (
    create_pipeline_run,
    save_agent_step,
    complete_pipeline_run,
    fail_pipeline_run,
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
    "rewrite_sections": ["resume_text", "keyword_categories", "gap_analysis"],
    "qa_deduplicate": ["resume_text", "raw_replacements", "jd_keywords"],
    "score_extract": ["resume_text", "jd_text", "jd_keywords", "replacements"],
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

_builder = StateGraph(AgentState)

_builder.add_node("extract_keywords", _tracked("extract_keywords", extract_keywords))
_builder.add_node("analyse_resume", _tracked("analyse_resume", analyse_resume))
_builder.add_node("score_before", _tracked("score_before", score_before_rewrite))
_builder.add_node("rewrite_sections", _tracked("rewrite_sections", rewrite_sections))
_builder.add_node("qa_deduplicate", _tracked("qa_deduplicate", qa_and_deduplicate))
_builder.add_node("score_extract", _tracked("score_extract", score_and_extract))
_builder.add_node("compile_pdf", _tracked("compile_pdf", compile_pdf))

_builder.set_entry_point("extract_keywords")
_builder.add_edge("extract_keywords", "analyse_resume")
_builder.add_edge("analyse_resume", "score_before")
_builder.add_edge("score_before", "rewrite_sections")
_builder.add_edge("rewrite_sections", "qa_deduplicate")
_builder.add_edge("qa_deduplicate", "score_extract")
_builder.add_edge("score_extract", "compile_pdf")
_builder.add_edge("compile_pdf", END)

graph = _builder.compile()


# ── Public API (drop-in replacement for groq_service.generate_resume) ─────

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
    logger.info("Starting LangGraph pipeline (7 agents).")

    run_id = create_pipeline_run(resume_text, jd_text)
    token = _current_run_id.set(run_id)

    initial_state: AgentState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "jd_keywords": [],
        "keyword_categories": {},
        "resume_sections": {},
        "gap_analysis": "",
        "raw_replacements": [],
        "replacements": [],
        "ats_score_before": 0,
        "ats_score": 0,
        "matched_keywords": [],
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

        return resume, compiled_pdf_b64

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        raise
    finally:
        _current_run_id.reset(token)
