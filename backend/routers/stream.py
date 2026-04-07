"""SSE streaming endpoint — streams per-agent progress events during pipeline execution."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.agents.graph import (
    graph,
    _current_run_id,
    _make_initial_state,
)
from backend.services.db import create_pipeline_run, complete_pipeline_run, fail_pipeline_run, save_compiled_pdf

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamRequest(BaseModel):
    resume_text: str
    jd_text: str
    resume_file_b64: str = ""
    resume_file_type: str = "pdf"


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_pipeline(body: StreamRequest):
    """Generator that yields SSE events as each agent completes."""
    run_id = create_pipeline_run(body.resume_text, body.jd_text)
    token = _current_run_id.set(run_id)

    initial_state = _make_initial_state(body.resume_text, body.jd_text)
    initial_state["resume_file_b64"] = body.resume_file_b64
    initial_state["resume_file_type"] = body.resume_file_type

    yield _sse_event("started", {"run_id": run_id})

    try:
        # Collect all updates to reconstruct final state
        accumulated: dict = dict(initial_state)

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in event.items():
                # Merge update into accumulated state
                accumulated.update(update)

                # Send a lightweight progress event (no huge payloads)
                progress: dict = {"agent": node_name}

                if node_name == "score_before":
                    progress["ats_score_before"] = update.get("ats_score_before", 0)
                elif node_name == "score_extract":
                    progress["ats_score"] = update.get("ats_score", 0)
                    progress["matched_keywords"] = update.get("matched_keywords", [])
                    progress["still_missing_keywords"] = update.get("still_missing_keywords", [])
                elif node_name in ("rewrite_skills", "rewrite_summary", "rewrite_experience"):
                    raw = update.get("raw_replacements", [])
                    progress["replacements_count"] = len(raw)
                elif node_name == "qa_deduplicate":
                    repls = update.get("replacements", [])
                    progress["replacements_count"] = len(repls)
                elif node_name == "compile_pdf":
                    progress["has_pdf"] = bool(update.get("compiled_pdf_b64"))

                yield _sse_event("agent_complete", progress)

        final = accumulated

        from backend.models import ResumeData
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

        compiled_pdf_b64 = final.get("compiled_pdf_b64", "")

        complete_pipeline_run(run_id, {
            "ats_score_before": resume.ats_score_before,
            "ats_score": resume.ats_score,
            "matched_keywords": resume.matched_keywords,
            "replacements_count": len(resume.replacements),
            "name": resume.name,
        })
        save_compiled_pdf(run_id, compiled_pdf_b64)

        yield _sse_event("complete", {
            "resume": resume.model_dump(),
            "rewritten_file_b64": compiled_pdf_b64,
        })

    except Exception as exc:
        fail_pipeline_run(run_id, str(exc))
        yield _sse_event("error", {"detail": str(exc)})
    finally:
        _current_run_id.reset(token)


@router.post("/generate-resume-stream")
async def generate_resume_stream(body: StreamRequest):
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty.")
    if not body.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    return StreamingResponse(
        _stream_pipeline(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
