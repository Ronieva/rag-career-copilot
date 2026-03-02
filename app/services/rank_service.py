from __future__ import annotations

from typing import Dict, List, Any

from app.rag.vectorstore import vectorstore
from app.services.match_service import match


def rank_jobs_against_cv(cv_id: str) -> List[Dict[str, Any]]:
    """
    Rank all ingested job descriptions against a given CV.

    Returns a list sorted by match_score desc with:
    - company
    - role
    - job_id (kept for traceability)
    - match_score
    - missing_required_skills (optional but useful)
    """

    # Pull all stored docs metadata from Chroma/LangChain
    store_dump = vectorstore.get()
    metadatas = store_dump.get("metadatas") or []

    # Collect unique jobs by doc_id + keep company/role
    jobs: Dict[str, Dict[str, str]] = {}
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        if meta.get("doc_type") != "job":
            continue

        job_id = meta.get("doc_id")
        if not job_id:
            continue

        # Keep first seen company/role for that job_id
        if job_id not in jobs:
            jobs[job_id] = {
                "company": meta.get("company", "") or "",
                "role": meta.get("role", "") or "",
            }

    rankings: List[Dict[str, Any]] = []

    for job_id, info in jobs.items():
        result = match_job_and_cv(job_id=job_id, cv_id=cv_id)

        rankings.append(
            {
                "company": info["company"],
                "role": info["role"],
                "job_id": job_id,
                "match_score": result.get("match_score", 0),
                "missing_required_skills": result.get("missing_required_skills", []),
            }
        )

    rankings.sort(key=lambda x: x["match_score"], reverse=True)
    return rankings
