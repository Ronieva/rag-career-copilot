import os
import uuid
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vector_db")

app = FastAPI(title="RAG Career Copilot")

# Embeddings + Vector store (Chroma)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings,
    collection_name="career_docs",
)

# Chat model (LLM)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    # Optional metadata filters
    doc_type: Optional[str] = None   # "job" | "cv" | "notes"
    company: Optional[str] = None
    role: Optional[str] = None


class Citation(BaseModel):
    doc_id: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


class MatchRequest(BaseModel):
    job_doc_id: str
    cv_doc_id: str


class MatchResponse(BaseModel):
    match_score: float
    required_skills: List[str]
    nice_to_have_skills: List[str]
    cv_skills: List[str]
    missing_required_skills: List[str]
    missing_nice_skills: List[str]


class RewriteRequest(BaseModel):
    job_doc_id: str
    cv_doc_id: str
    bullets: List[str]


class RewriteResponse(BaseModel):
    rewritten_bullets: List[str]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _split_docs(docs: List[Document], base_meta: Dict[str, Any]) -> List[Document]:
    """
    Split documents into chunks with overlap and attach consistent metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    out: List[Document] = []
    for i, ch in enumerate(chunks):
        meta = dict(ch.metadata or {})
        meta.update(base_meta)
        meta["chunk_id"] = meta.get("chunk_id") or f"chunk_{i}"
        out.append(Document(page_content=ch.page_content, metadata=meta))
    return out


def _build_filter(req: QueryRequest) -> Dict[str, Any]:
    """
    Build a Chroma metadata filter from the query request.
    """
    f: Dict[str, Any] = {}
    if req.doc_type:
        f["doc_type"] = req.doc_type
    if req.company:
        f["company"] = req.company
    if req.role:
        f["role"] = req.role
    return f


def _format_citations(docs: List[Document]) -> List[Citation]:
    """
    Convert retrieved chunks to a compact citation list.
    """
    cites: List[Citation] = []
    for d in docs:
        meta = d.metadata or {}
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        cites.append(
            Citation(
                doc_id=str(meta.get("doc_id", "")),
                source=str(meta.get("source", "")),
                page=meta.get("page"),
                chunk_id=meta.get("chunk_id"),
                snippet=snippet,
            )
        )
    return cites


def _make_context(docs: List[Document]) -> str:
    """
    Create a context string with per-chunk headers that include source/page/doc_id.
    """
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        header = (
            f"[source={meta.get('source','')} "
            f"page={meta.get('page','?')} "
            f"doc_id={meta.get('doc_id','')} "
            f"chunk={meta.get('chunk_id','')}]"
        )
        blocks.append(header + "\n" + d.page_content)
    return "\n\n---\n\n".join(blocks)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON strictly. If it fails, raise a helpful error.
    """
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON. Raw output: {text[:500]}") from e


def extract_json_from_llm(system_prompt: str, user_content: str) -> Dict[str, Any]:
    """
    Call the LLM and enforce a strict JSON response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = llm.invoke(messages).content
    return _safe_json_loads(response)


def _get_doc_text_by_id(doc_id: str, hint_query: str, k: int = 25) -> str:
    """
    Fetch chunks for a given doc_id from the vector store and concatenate them.
    We use similarity_search with a stable hint query so we get representative chunks.
    """
    docs = vectorstore.similarity_search(
        query=hint_query,
        k=k,
        filter={"doc_id": doc_id},
    )
    if not docs:
        return ""
    return "\n".join([d.page_content for d in docs])


# ------------------------------------------------------------
# Prompts (English)
# ------------------------------------------------------------
RAG_SYSTEM = """You are a specialized assistant that analyzes job descriptions and CVs using ONLY the provided context.
Rules:
- Do not invent facts. If the context is insufficient, say: "I don't know based on the available context."
- Be concise, practical, and structured.
- Prefer bullet points where appropriate.
"""

RAG_USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

Instructions:
- Answer clearly and directly.
- If information is missing, explicitly say what is missing and what document would help.
"""


JOB_SKILL_EXTRACTION_PROMPT = """
You are an expert technical recruiter.

Extract structured technical skills from the JOB DESCRIPTION.

Return ONLY valid JSON in this exact format:

{
  "required_skills": ["..."],
  "nice_to_have_skills": ["..."]
}

Rules:
- Do NOT invent skills.
- Only extract skills explicitly mentioned.
- Keep skills concise (e.g., "Python", "SQL", "Airflow", "Docker").
- Do not include soft skills unless they are explicitly technical (e.g., "Git").
"""


CV_SKILL_EXTRACTION_PROMPT = """
You are analyzing a CANDIDATE CV.

Extract all explicitly mentioned technical skills.

Return ONLY valid JSON:

{
  "skills": ["..."]
}

Rules:
- Do NOT infer or assume.
- Only extract skills explicitly stated.
- Keep skills concise (e.g., "Python", "SQL", "FastAPI", "LangChain").
"""


REWRITE_PROMPT = """
You are an ATS optimization assistant.

Rewrite the CV bullet points to better align with the job description.

Return ONLY valid JSON in this format:

{
  "rewritten_bullets": ["...", "..."]
}

Rules:
- Do NOT invent new technologies or experiences.
- Only use skills/technologies already present in the CV text OR already present in the original bullets.
- Emphasize overlap with job requirements using strong action verbs.
- Keep each bullet concise and measurable when possible.
"""


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Basic health endpoint.
    """
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("job"),
    company: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
):
    """
    Ingest a PDF into the vector store.
    Metadata is stored per chunk: doc_id, doc_type, company, role, source, page, chunk_id.
    """
    raw = await file.read()
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(raw)

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()  # Each page becomes a Document with page metadata

    doc_id = str(uuid.uuid4())
    base_meta = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "company": company or "",
        "role": role or "",
        "source": file.filename,
    }

    chunks = _split_docs(pages, base_meta)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}


@app.post("/ingest/text")
async def ingest_text(payload: Dict[str, Any]):
    """
    Ingest raw text into the vector store.

    Expected payload:
    {
      "text": "...",
      "doc_type": "cv",
      "company": "X",
      "role": "Data Engineer",
      "source": "cv.txt"
    }
    """
    text = payload.get("text", "")
    if not text.strip():
        return {"status": "error", "message": "Empty text"}

    doc_id = str(uuid.uuid4())
    base_meta = {
        "doc_id": doc_id,
        "doc_type": payload.get("doc_type", "notes"),
        "company": payload.get("company", ""),
        "role": payload.get("role", ""),
        "source": payload.get("source", "text"),
        "page": None,
    }

    docs = [Document(page_content=text, metadata={"page": None})]
    chunks = _split_docs(docs, base_meta)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    RAG query endpoint with citations.
    Uses MMR retrieval and optional metadata filters.
    """
    f = _build_filter(req)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": req.top_k,
            "fetch_k": max(20, req.top_k * 3),
            "filter": f or None,
        },
    )

    docs = retriever.get_relevant_documents(req.question)

    # Guardrail: if nothing retrieved, respond safely
    if not docs:
        return QueryResponse(
            answer="I don't know based on the available context.",
            citations=[],
        )

    context = _make_context(docs)

    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": RAG_USER_TEMPLATE.format(question=req.question, context=context)},
    ]

    out = llm.invoke(messages).content
    return QueryResponse(answer=out, citations=_format_citations(docs))


@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    """
    Compute a skill-based match score between a job description and a CV.
    Skills are extracted via LLM with strict JSON outputs, then scored deterministically.
    """

    job_text = _get_doc_text_by_id(req.job_doc_id, hint_query="job description")
    cv_text = _get_doc_text_by_id(req.cv_doc_id, hint_query="candidate cv")

    if not job_text.strip() or not cv_text.strip():
        # Return a consistent structure; keep it simple for clients
        return MatchResponse(
            match_score=0.0,
            required_skills=[],
            nice_to_have_skills=[],
            cv_skills=[],
            missing_required_skills=[],
            missing_nice_skills=[],
        )

    job_data = extract_json_from_llm(JOB_SKILL_EXTRACTION_PROMPT, job_text)
    cv_data = extract_json_from_llm(CV_SKILL_EXTRACTION_PROMPT, cv_text)

    required = set(job_data.get("required_skills", []) or [])
    nice = set(job_data.get("nice_to_have_skills", []) or [])
    cv_skills = set(cv_data.get("skills", []) or [])

    missing_required = sorted(list(required - cv_skills))
    missing_nice = sorted(list(nice - cv_skills))

    # Deterministic score: % of required skills covered
    if len(required) > 0:
        score = (len(required & cv_skills) / len(required)) * 100.0
    else:
        score = 0.0

    return MatchResponse(
        match_score=round(score, 2),
        required_skills=sorted(list(required)),
        nice_to_have_skills=sorted(list(nice)),
        cv_skills=sorted(list(cv_skills)),
        missing_required_skills=missing_required,
        missing_nice_skills=missing_nice,
    )


@app.post("/rewrite-bullets", response_model=RewriteResponse)
async def rewrite_bullets(req: RewriteRequest):
    """
    Rewrite CV bullet points to better match a job description.
    Guardrails: do not invent new technologies; only use what is in CV or original bullets.
    """

    job_text = _get_doc_text_by_id(req.job_doc_id, hint_query="job description")
    cv_text = _get_doc_text_by_id(req.cv_doc_id, hint_query="candidate cv")

    if not job_text.strip() or not cv_text.strip():
        return RewriteResponse(rewritten_bullets=req.bullets)

    context = f"""
JOB DESCRIPTION:
{job_text}

CV TEXT:
{cv_text}

CURRENT BULLETS (rewrite these):
{json.dumps(req.bullets, ensure_ascii=False)}
"""

    result = extract_json_from_llm(REWRITE_PROMPT, context)
    rewritten = result.get("rewritten_bullets", []) or req.bullets

    # Minimal validation: ensure list of strings
    if not isinstance(rewritten, list) or not all(isinstance(x, str) for x in rewritten):
        rewritten = req.bullets

    return RewriteResponse(rewritten_bullets=rewritten)
