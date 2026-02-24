from typing import Dict, Any, List
from app.core.llm import llm
from app.core.json_utils import parse_llm_json
from app.core.errors import NotFoundError
from app.rag.prompts import REWRITE_PROMPT
from app.rag.vectorstore import vectorstore


def rewrite_bullets(job_id: str, cv_id: str, bullets: List[str]) -> Dict[str, Any]:
    """
    Rewrite CV bullet points to better align with a job description.
    Guardrail: Do not invent new technologies.
    """
    job_docs = vectorstore.similarity_search("job description", k=20, filter={"doc_id": job_id})
    cv_docs = vectorstore.similarity_search("candidate cv", k=20, filter={"doc_id": cv_id})

    if not job_docs:
        raise NotFoundError(detail={"doc_id": job_id, "type": "job"})
    if not cv_docs:
        raise NotFoundError(detail={"doc_id": cv_id, "type": "cv"})

    job_text = "\n".join(d.page_content for d in job_docs)
    cv_text = "\n".join(d.page_content for d in cv_docs)

    context = f"""
JOB DESCRIPTION:
{job_text}

CV CONTENT:
{cv_text}

CURRENT BULLETS (rewrite these):
{bullets}
""".strip()

    response = llm.invoke([
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": context}
    ]).content

    data = parse_llm_json(response)
    rewritten = data.get("rewritten_bullets", bullets)

    if not isinstance(rewritten, list) or not all(isinstance(x, str) for x in rewritten):
        rewritten = bullets

    return {"rewritten_bullets": rewritten}
