import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for easier parsing in production."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach common structured fields if present
        for k in ("request_id", "path", "method", "status_code"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to output structured JSON logs."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Avoid duplicate handlers in reload/dev
    root.handlers = [handler]


def get_request_id(request: Request) -> str:
    """Get or create a request id from headers."""
    rid = request.headers.get("x-request-id")
    return rid or str(uuid.uuid4())
