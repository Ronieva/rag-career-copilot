from fastapi import FastAPI, UploadFile, File
from app.rag.ingest import ingest_pdf
from app.rag.vectorstore import vectorstore
from app.rag.retriever import build_retriever
from app.services.match_service import match

app = FastAPI(title="RAG Career Copilot")

@app.post("/ingest/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    doc_id, chunks = ingest_pdf(path, file.filename)
    return {"doc_id": doc_id, "chunks": chunks}


@app.post("/match")
async def match_endpoint(job_id: str, cv_id: str):
    return match(job_id, cv_id)
