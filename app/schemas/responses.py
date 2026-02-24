from pydantic import BaseModel
from typing import List, Optional


class Citation(BaseModel):
    doc_id: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


class MatchResponse(BaseModel):
    match_score: float
    required_skills: List[str]
    cv_skills: List[str]
    missing_required_skills: List[str]


class RewriteResponse(BaseModel):
    rewritten_bullets: List[str]
