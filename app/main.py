import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.logging import setup_logging, get_request_id
from app.core.errors import AppError, ErrorResponse
from app.rag.ingest import ingest_pdf, ingest_text
from app.services.match_service import match
from app.services.rewrite_service import rewrite_bullets
from app.schemas.requests import MatchRequest, RewriteRequest
from app.schemas.responses import MatchResponse, RewriteResponse

setup_logging()
logger = logging.getLogger("app")

app = FastAPI(title="RAG Career Copilot")


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


@app.post("/rewrite-bullets", response_model=RewriteResponse)
async def rewrite_endpoint(request: RewriteRequest):
    return rewrite_bullets(request.job_id, request.cv_id, request.bullets)
