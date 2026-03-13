from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models import TextResponse
from backend.services.parser import parse_pdf, parse_docx

router = APIRouter()

_ALLOWED_TYPES = {
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
}


@router.post("/parse-resume", response_model=TextResponse)
async def parse_resume(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    # Also allow detection by filename extension as fallback
    if content_type not in _ALLOWED_TYPES:
        filename = file.filename or ""
        if filename.lower().endswith(".pdf"):
            content_type = "application/pdf"
        elif filename.lower().endswith(".docx"):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Please upload a PDF or DOCX file.",
            )

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    try:
        parser = _ALLOWED_TYPES[content_type]
        text = parser(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}") from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    return TextResponse(text=text)
