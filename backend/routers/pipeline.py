"""API routes for inspecting pipeline runs (agent I/O)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from backend.services.db import get_pipeline_runs, get_pipeline_run, get_compiled_pdf, _get_db

router = APIRouter()


@router.get("/pipeline-runs/status")
def pipeline_runs_status():
    """Return whether the pipeline run DB is connected."""
    db = _get_db()
    return {"db_connected": db is not None}


@router.get("/pipeline-runs")
def list_pipeline_runs(limit: int = 20, skip: int = 0):
    runs = get_pipeline_runs(limit=min(limit, 100), skip=skip)
    return runs


@router.get("/pipeline-runs/{run_id}")
def get_run_detail(run_id: str):
    run = get_pipeline_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found.")
    return run


@router.get("/pipeline-runs/{run_id}/pdf")
def download_compiled_pdf(run_id: str):
    pdf_bytes = get_compiled_pdf(run_id)
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail="No compiled PDF found for this run.")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="resume_{run_id}.pdf"'},
    )
