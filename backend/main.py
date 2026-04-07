import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

def _configure_logging() -> None:
    from pythonjsonlogger.json import JsonFormatter

    handler = logging.StreamHandler(sys.stdout)
    fmt = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

_configure_logging()

from backend.routers import resume, jd, generate, pipeline, stream  # noqa: E402

app = FastAPI(title="Resume for ATS API", version="1.0.0")

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
app.include_router(pipeline.router, prefix="/api")
app.include_router(stream.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
