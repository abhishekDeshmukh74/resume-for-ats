import base64

from fastapi import APIRouter, HTTPException
from backend.models import (
    GenerateRequest, GenerateResponse,
    CoverLetterRequest, CoverLetterResponse,
    PreviewRequest, PreviewResponse,
    ConfirmRequest, ConfirmResponse,
)
from backend.services.agents import generate_resume
from backend.services.agents.graph import preview_resume, confirm_resume
from backend.services.agents.cover_letter import generate_cover_letter

router = APIRouter()


@router.post("/generate-resume", response_model=GenerateResponse)
async def generate_resume_endpoint(body: GenerateRequest):
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty.")
    if not body.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    try:
        resume_data, rewritten_b64 = generate_resume(
            body.resume_text,
            body.jd_text,
            body.resume_file_b64,
            body.resume_file_type,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"AI generation failed: {exc}"
        ) from exc

    if not rewritten_b64:
        raise HTTPException(status_code=500, detail="PDF compilation produced no output.")

    return GenerateResponse(resume=resume_data, rewritten_file_b64=rewritten_b64)


@router.post("/generate-cover-letter", response_model=CoverLetterResponse)
async def cover_letter_endpoint(body: CoverLetterRequest):
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty.")
    if not body.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    try:
        result = generate_cover_letter(
            body.resume_text, body.jd_text, body.company_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Cover letter generation failed: {exc}"
        ) from exc

    return CoverLetterResponse(**result)


@router.post("/preview", response_model=PreviewResponse)
async def preview_endpoint(body: PreviewRequest):
    """Run the pipeline without PDF compilation — return proposed replacements."""
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty.")
    if not body.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    try:
        result = preview_resume(body.resume_text, body.jd_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Preview generation failed: {exc}"
        ) from exc

    return PreviewResponse(**result)


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm_endpoint(body: ConfirmRequest):
    """Apply user-approved replacements and compile the final PDF."""
    if not body.resume_file_b64.strip():
        raise HTTPException(status_code=400, detail="resume_file_b64 is required.")
    if not body.replacements:
        raise HTTPException(status_code=400, detail="At least one replacement is required.")

    try:
        result = confirm_resume(
            body.resume_text,
            [r.model_dump() for r in body.replacements],
            body.resume_file_b64,
            body.resume_file_type,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"PDF compilation failed: {exc}"
        ) from exc

    return ConfirmResponse(rewritten_file_b64=result)
