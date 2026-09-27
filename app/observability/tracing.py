"""LangSmith tracing setup.

LangChain/LangGraph trace automatically when the LANGSMITH_* environment
variables are present. Settings from `.env` are loaded into Python (not the
process environment), so `configure_langsmith()` exports them once at startup.

What a trace contains:
- one root run per conversation turn ("conversation_turn"), tagged with the
  user and grouped by session_id, so LangSmith's Threads view shows whole
  conversations;
- a child run per graph node (agent) with its inputs/outputs, including our
  decision log entries;
- every LLM call with prompt, response, token usage and latency;
- the hybrid retrieval step as a retriever run listing the returned chunks.
"""

import logging
import os
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def configure_langsmith() -> bool:
    """Export LangSmith settings to the environment. Returns True if tracing is enabled."""
    s = get_settings()
    key = s.langsmith_api_key.get_secret_value().strip() if s.langsmith_api_key else ""
    if not s.langsmith_tracing:
        return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"  # respect a shell-level setting
    if not key:
        logger.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is empty; tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    if s.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
    logger.info("LangSmith tracing enabled (project=%s)", s.langsmith_project)
    return True
