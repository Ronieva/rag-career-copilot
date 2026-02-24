from pydantic import BaseModel
from typing import Optional, Any, Dict


class ErrorResponse(BaseModel):
    """Standard API error response."""
    error: str
    detail: Optional[Any] = None


class AppError(Exception):
    """Base application error."""
    def __init__(self, error: str, detail: Any = None, status_code: int = 400):
        super().__init__(error)
        self.error = error
        self.detail = detail
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, detail: Any = None):
        super().__init__("not_found", detail=detail, status_code=404)


class BadRequestError(AppError):
    def __init__(self, detail: Any = None):
        super().__init__("bad_request", detail=detail, status_code=400)


class LLMJsonError(AppError):
    def __init__(self, raw_output: str):
        super().__init__("llm_invalid_json", detail={"raw_output": raw_output[:800]}, status_code=502)
