import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.routers import resume, jd, generate, pdf  # noqa: E402

app = FastAPI(title="pass-ats API", version="1.0.0")

origins = os.getenv("ALLOWED_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume.router, prefix="/api")
app.include_router(jd.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(pdf.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
