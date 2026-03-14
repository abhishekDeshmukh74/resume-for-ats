import base64

from fastapi import APIRouter, HTTPException
from backend.models import GenerateRequest, GenerateResponse
from backend.services.agents import generate_resume

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
