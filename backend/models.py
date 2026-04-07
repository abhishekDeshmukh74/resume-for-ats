from pydantic import BaseModel, field_validator
from typing import Optional


class ExperienceItem(BaseModel):
    job_title: str
    company: str
    location: Optional[str] = None
    start_date: str
    end_date: str  # "Present" or date string
    bullets: list[str]


class EducationItem(BaseModel):
    degree: str
    institution: str
    location: Optional[str] = None
    graduation_date: str
    details: Optional[list[str]] = None

    @field_validator("details", mode="before")
    @classmethod
    def coerce_details_to_list(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else None
        return v


class CertificationItem(BaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None


class TextReplacement(BaseModel):
    """A single old→new text substitution identified by the AI."""
    old: str   # exact substring from the original resume text
    new: str   # rewritten substring to replace it with


class ResumeData(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = []
    experience: list[ExperienceItem] = []
    education: list[EducationItem] = []
    certifications: list[CertificationItem] = []
    ats_score_before: Optional[int] = None   # 0-100 (before rewrite)
    ats_score: Optional[int] = None            # 0-100 (after rewrite)
    matched_keywords: list[str] = []
    replacements: list[TextReplacement] = []  # explicit old→new pairs for PDF rewriting


class GenerateRequest(BaseModel):
    resume_text: str
    jd_text: str
    resume_file_b64: str   # base64 original PDF bytes
    resume_file_type: str = "pdf"


class GenerateResponse(BaseModel):
    resume: ResumeData
    rewritten_file_b64: str  # base64-encoded rewritten PDF


class ScrapeRequest(BaseModel):
    url: str


class TextResponse(BaseModel):
    text: str


class ParsedResumeResponse(BaseModel):
    text: str        # plain text (used by AI pipeline)
    html: str        # styled HTML preserving fonts, colours, layout
    file_b64: str    # base64-encoded original PDF bytes
    file_type: str   # "pdf"


# ── Cover letter & outreach ──────────────────────────────────────────────

class CoverLetterRequest(BaseModel):
    resume_text: str
    jd_text: str
    company_name: Optional[str] = None


class CoverLetterResponse(BaseModel):
    cover_letter: str
    suggested_job_title: str
    linkedin_message: str


# ── Preview / Confirm two-phase flow ─────────────────────────────────────

class PreviewRequest(BaseModel):
    resume_text: str
    jd_text: str


class PreviewResponse(BaseModel):
    replacements: list[TextReplacement]
    ats_score_before: int
    ats_score: int
    matched_keywords: list[str]
    still_missing_keywords: list[str]


class ConfirmRequest(BaseModel):
    resume_text: str
    replacements: list[TextReplacement]
    resume_file_b64: str
    resume_file_type: str = "pdf"


class ConfirmResponse(BaseModel):
    rewritten_file_b64: str
