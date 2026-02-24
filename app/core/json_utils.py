import json
import re
from typing import Any, Dict

from app.core.errors import LLMJsonError


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM output robustly.

    Strategy:
    1) Try strict json.loads on full text
    2) If it fails, attempt to extract the first {...} block and parse it
    3) If it still fails, raise LLMJsonError
    """
    text = (text or "").strip()

    # 1) Strict parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # Some models might return a list; wrap is not desired here
        raise ValueError("JSON is not an object")
    except Exception:
        pass

    # 2) Extract JSON object block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) Fail
    raise LLMJsonError(raw_output=text)
