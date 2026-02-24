import json
from app.core.llm import llm
from app.rag.prompts import REWRITE_PROMPT
from app.rag.vectorstore import vectorstore


def extract_json(prompt: str, text: str):
    response = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]).content

    return json.loads(response)


def rewrite_bullets(job_id: str, cv_id: str, bullets: list[str]) -> dict:
    """
    Rewrite CV bullet points to better align with a job description.
    Guardrail: Do not invent new technologies.
    """

    job_docs = vectorstore.similarity_search(
        query="job description",
        k=20,
        filter={"doc_id": job_id}
    )

    cv_docs = vectorstore.similarity_search(
        query="cv",
        k=20,
        filter={"doc_id": cv_id}
    )

    job_text = "\n".join([d.page_content for d in job_docs])
    cv_text = "\n".join([d.page_content for d in cv_docs])

    context = f"""
JOB DESCRIPTION:
{job_text}

CV CONTENT:
{cv_text}

CURRENT BULLETS:
{bullets}
"""

    result = extract_json(REWRITE_PROMPT, context)

    return {
        "rewritten_bullets": result.get("rewritten_bullets", bullets)
    }
