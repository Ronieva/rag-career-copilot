from typing import Dict, Any
from app.core.llm import llm
from app.core.json_utils import parse_llm_json
from app.core.errors import NotFoundError
from app.rag.prompts import JOB_SKILL_EXTRACTION, CV_SKILL_EXTRACTION
from app.rag.vectorstore import vectorstore


def _extract_json(system_prompt: str, user_text: str) -> Dict[str, Any]:
    """Call the LLM and enforce JSON parsing with robust handling."""
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]).content
    return parse_llm_json(response)


def match(job_id: str, cv_id: str) -> Dict[str, Any]:
    """Compute a deterministic match score based on required skills overlap."""
    job_docs = vectorstore.similarity_search("job description", k=20, filter={"doc_id": job_id})
    cv_docs = vectorstore.similarity_search("candidate cv", k=20, filter={"doc_id": cv_id})

    if not job_docs:
        raise NotFoundError(detail={"doc_id": job_id, "type": "job"})
    if not cv_docs:
        raise NotFoundError(detail={"doc_id": cv_id, "type": "cv"})

    job_text = "\n".join(d.page_content for d in job_docs)
    cv_text = "\n".join(d.page_content for d in cv_docs)

    job_data = _extract_json(JOB_SKILL_EXTRACTION, job_text)
    cv_data = _extract_json(CV_SKILL_EXTRACTION, cv_text)

    required = set(job_data.get("required_skills", []) or [])
    cv_skills = set(cv_data.get("skills", []) or [])

    score = (len(required & cv_skills) / len(required) * 100.0) if required else 0.0

    return {
        "match_score": round(score, 2),
        "required_skills": sorted(list(required)),
        "cv_skills": sorted(list(cv_skills)),
        "missing_required_skills": sorted(list(required - cv_skills)),
    }
