import logging
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import json

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

from app.core.logging import setup_logging, get_request_id
from app.core.errors import AppError, ErrorResponse
from app.rag.ingest import ingest_pdf, ingest_text
from app.services.match_service import match
from app.services.rank_service import rank_jobs_against_cv
from app.services.rewrite_service import rewrite_bullets
from app.schemas.requests import MatchRequest, RewriteRequest
from app.schemas.responses import MatchResponse, RewriteResponse

from app.rag.ingest import ingest_text

print("LOADED app/main.py ✅")
setup_logging()
logger = logging.getLogger("app")

app = FastAPI(title="RAG Career Copilot")

JSON_SAMPLES_DIR = Path("sample_data/json")
# -----------------------------
# Middleware: request_id + access logs
# -----------------------------
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = get_request_id(request)
    request.state.request_id = request_id

    logger.info(
        "request_started",
        extra={"request_id": request_id, "path": request.url.path, "method": request.method},
    )

    try:
        response = await call_next(request)
    except Exception as e:
        # Let exception handlers format the response
        raise e

    response.headers["x-request-id"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
        },
    )
    return response


# -----------------------------
# Exception handlers
# -----------------------------
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    logger.warning(
        "app_error",
        extra={"request_id": request.state.request_id, "path": request.url.path, "method": request.method},
    )
    payload = ErrorResponse(error=exc.error, detail=exc.detail).model_dump()
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(error="validation_error", detail=exc.errors()).model_dump()
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        extra={"request_id": getattr(request.state, "request_id", "unknown"), "path": request.url.path, "method": request.method},
    )
    payload = ErrorResponse(error="internal_server_error").model_dump()
    return JSONResponse(status_code=500, content=payload)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("job"),
    company: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
):
    # Save to a unique temp file to avoid name collisions
    contents = await file.read()
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(contents)

    doc_id, chunks = ingest_pdf(
        path=tmp_path,
        filename=file.filename,
        doc_type=doc_type,
        company=company,
        role=role,
    )

    return {"status": "ok", "doc_id": doc_id, "chunks": chunks}

from pydantic import BaseModel

class TextIngestRequest(BaseModel):
    text: str
    doc_type: str
    company: str | None = None
    role: str | None = None
    source: str

@app.post("/ingest/text")
async def upload_text(request: TextIngestRequest):
    doc_id, chunks = ingest_text(
        text=request.text,
        source=request.source,
        doc_type=request.doc_type,
        company=request.company,
        role=request.role,
    )

    return {"status": "ok", "doc_id": doc_id, "chunks": chunks}

@app.post("/match", response_model=MatchResponse)
async def match_endpoint(request: MatchRequest):
    return match(request.job_id, request.cv_id)

class RankRequest(BaseModel):
    cv_id: str

@app.post("/rank-jobs")
async def rank_jobs(request: RankJobsRequest):
    rankings = rank_jobs_against_cv(request.cv_id)
    return {"rankings": rankings}
    
@app.post("/rewrite-bullets", response_model=RewriteResponse)
async def rewrite_endpoint(request: RewriteRequest):
    return rewrite_bullets(request.job_id, request.cv_id, request.bullets)

# --- DEV: ingest all JSON samples from sample_data/json ---
from pathlib import Path
import json
from typing import Any, Dict, List
from fastapi import HTTPException

from app.rag.ingest import ingest_text

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_SAMPLES_DIR = BASE_DIR / "sample_data" / "json"


@app.post("/ingest/samples/json")
async def ingest_samples_json() -> Dict[str, Any]:
    if not JSON_SAMPLES_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {JSON_SAMPLES_DIR}",
        )

    files = sorted(JSON_SAMPLES_DIR.glob("*.json"))
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"No .json files found in: {JSON_SAMPLES_DIR}",
        )

    results: List[Dict[str, Any]] = []
    ingested = 0
    failed = 0

    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8", errors="replace"))

            # allow file to contain either a single object or a list of objects
            items = payload if isinstance(payload, list) else [payload]

            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError("JSON item is not an object")

                text = item.get("text")
                doc_type = item.get("doc_type")
                source = item.get("source") or fp.name
                company = item.get("company")
                role = item.get("role")

                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Missing/invalid 'text'")
                if not isinstance(doc_type, str) or not doc_type.strip():
                    raise ValueError("Missing/invalid 'doc_type'")

                doc_id, chunks = ingest_text(
                    text=text,
                    source=str(source),
                    doc_type=doc_type,
                    company=company,
                    role=role,
                )

                results.append(
                    {
                        "file": fp.name,
                        "index": idx,
                        "status": "ok",
                        "doc_type": doc_type,
                        "company": company or "",
                        "role": role or "",
                        "source": source,
                        "doc_id": doc_id,
                        "chunks": chunks,
                    }
                )
                ingested += 1

        except Exception as e:
            results.append({"file": fp.name, "status": "error", "error": str(e)})
            failed += 1

    return {"status": "ok", "ingested": ingested, "failed": failed, "results": results}
