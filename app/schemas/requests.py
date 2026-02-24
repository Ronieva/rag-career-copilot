from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    doc_type: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None


class MatchRequest(BaseModel):
    job_id: str
    cv_id: str


class RewriteRequest(BaseModel):
    job_id: str
    cv_id: str
    bullets: List[str]
