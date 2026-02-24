from fastapi import FastAPI, UploadFile, File
from app.rag.ingest import ingest_pdf
from app.services.match_service import match
from app.services.rewrite_service import rewrite_bullets
from app.schemas.requests import MatchRequest, RewriteRequest
from app.schemas.responses import MatchResponse, RewriteResponse

app = FastAPI(title="RAG Career Copilot")


@app.post("/ingest/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    path = f"/tmp/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)

    doc_id, chunks = ingest_pdf(path, file.filename)
    return {"doc_id": doc_id, "chunks": chunks}


@app.post("/match", response_model=MatchResponse)
async def match_endpoint(request: MatchRequest):
    return match(request.job_id, request.cv_id)


@app.post("/rewrite-bullets", response_model=RewriteResponse)
async def rewrite_endpoint(request: RewriteRequest):
    return rewrite_bullets(
        request.job_id,
        request.cv_id,
        request.bullets
    )
