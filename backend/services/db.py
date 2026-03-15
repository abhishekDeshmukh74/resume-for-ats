"""MongoDB persistence for pipeline run tracking.

Stores each agent's input/output so runs can be inspected via the /info UI.
All operations are best-effort — MongoDB failures never break the pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from bson import Binary, ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db = None


def _get_db():
    global _client, _db
    if _db is not None:
        return _db
    url = os.getenv("MONGODB_URL")
    if not url:
        logger.warning("MONGODB_URL not set — pipeline tracking disabled.")
        return None
    try:
        _client = MongoClient(url, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client.get_default_database()
        logger.info("Connected to MongoDB: %s", _db.name)
        return _db
    except PyMongoError as e:
        logger.warning("Could not connect to MongoDB: %s", e)
        return None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_MAX_STR = 2000
_MAX_LIST = 50


def _serialize(value: Any) -> Any:
    """Recursively convert a value to a MongoDB-safe type."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_STR] + "…" if len(value) > _MAX_STR else value
    if hasattr(value, "model_dump"):          # Pydantic model
        return _serialize(value.model_dump())
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_serialize(v) for v in value[:_MAX_LIST]]
        if len(value) > _MAX_LIST:
            items.append(f"… and {len(value) - _MAX_LIST} more")
        return items
    return str(value)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def create_pipeline_run(resume_text: str, jd_text: str) -> str | None:
    db = _get_db()
    if db is None:
        return None
    try:
        result = db.pipeline_runs.insert_one({
            "created_at": datetime.now(timezone.utc),
            "status": "running",
            "resume_summary": resume_text[:500],
            "jd_summary": jd_text[:500],
            "agents": [],
            "final_result": None,
        })
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.warning("Failed to create pipeline run: %s", e)
        return None


def save_agent_step(
    run_id: str | None,
    agent_name: str,
    duration_ms: int,
    input_summary: dict,
    output_data: dict,
) -> None:
    if run_id is None:
        return
    db = _get_db()
    if db is None:
        return
    step = {
        "name": agent_name,
        "duration_ms": duration_ms,
        "input_summary": _serialize(input_summary),
        "output": _serialize(output_data),
    }
    try:
        db.pipeline_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$push": {"agents": step}},
        )
    except PyMongoError as e:
        logger.warning("Failed to save agent step %s: %s", agent_name, e)


def complete_pipeline_run(run_id: str | None, final_result: dict) -> None:
    if run_id is None:
        return
    db = _get_db()
    if db is None:
        return
    try:
        db.pipeline_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
                "final_result": _serialize(final_result),
            }},
        )
    except PyMongoError as e:
        logger.warning("Failed to complete pipeline run: %s", e)


def fail_pipeline_run(run_id: str | None, error: str) -> None:
    if run_id is None:
        return
    db = _get_db()
    if db is None:
        return
    try:
        db.pipeline_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc),
                "error": error[:1000],
            }},
        )
    except PyMongoError as e:
        logger.warning("Failed to mark pipeline run as failed: %s", e)


def save_compiled_pdf(run_id: str | None, pdf_b64: str) -> None:
    """Store the compiled PDF (as binary) on the pipeline run document."""
    if not run_id or not pdf_b64:
        return
    db = _get_db()
    if db is None:
        return
    import base64

    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        db.pipeline_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "compiled_pdf": Binary(pdf_bytes),
                "has_compiled_pdf": True,
            }},
        )
    except PyMongoError as e:
        logger.warning("Failed to save compiled PDF: %s", e)


def get_compiled_pdf(run_id: str) -> bytes | None:
    """Retrieve the compiled PDF bytes for a pipeline run."""
    db = _get_db()
    if db is None:
        return None
    try:
        doc = db.pipeline_runs.find_one(
            {"_id": ObjectId(run_id)},
            {"compiled_pdf": 1},
        )
        if doc and doc.get("compiled_pdf"):
            return bytes(doc["compiled_pdf"])
        return None
    except PyMongoError as e:
        logger.warning("Failed to fetch compiled PDF for %s: %s", run_id, e)
        return None


# ---------------------------------------------------------------------------
# Read helpers (used by the /api/pipeline-runs route)
# ---------------------------------------------------------------------------

def get_pipeline_runs(limit: int = 20, skip: int = 0) -> list[dict]:
    db = _get_db()
    if db is None:
        return []
    try:
        cursor = (
            db.pipeline_runs
            .find({}, {
                "resume_summary": 1,
                "jd_summary": 1,
                "status": 1,
                "created_at": 1,
                "completed_at": 1,
                "final_result": 1,
                "has_compiled_pdf": 1,
                "agents.name": 1,
                "agents.duration_ms": 1,
            })
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        runs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            runs.append(doc)
        return runs
    except PyMongoError as e:
        logger.warning("Failed to fetch pipeline runs: %s", e)
        return []


def get_pipeline_run(run_id: str) -> dict | None:
    db = _get_db()
    if db is None:
        return None
    try:
        doc = db.pipeline_runs.find_one(
            {"_id": ObjectId(run_id)},
            {"compiled_pdf": 0},
        )
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except PyMongoError as e:
        logger.warning("Failed to fetch pipeline run %s: %s", run_id, e)
        return None
