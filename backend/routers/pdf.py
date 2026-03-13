from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.models import ResumeData
from backend.services.pdf_generator import generate_pdf
import io
import re

router = APIRouter()


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", name).strip()
    safe = re.sub(r"\s+", "_", safe)
    return f"{safe}_resume.pdf" if safe else "resume.pdf"


@router.post("/download-pdf")
async def download_pdf(resume: ResumeData):
    try:
        pdf_bytes = generate_pdf(resume)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    filename = _safe_filename(resume.name)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
