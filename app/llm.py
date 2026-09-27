"""LLM and embedding model factories, plus provider-agnostic call helpers.

Every agent obtains its model through `get_llm()` and every vector operation
through `get_embeddings()`, so switching provider is a `.env` change only.
Provider packages are imported lazily: only the one you configure must be installed.

Agents call models through `invoke_structured()` / `invoke_text()` rather
than the model directly, so provider differences are handled in one place.
"""

import json
import logging
import re
from functools import lru_cache
from typing import TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model's reply could not be parsed into the requested schema."""


def _require_key(key, name: str):
    if key is None or not key.get_secret_value().strip():
        raise ValueError(f"{name} is not set; add it to your .env file.")
    return key


@lru_cache(maxsize=2)
def get_llm(fast: bool = False) -> BaseChatModel:
    """Return the configured chat model.

    Args:
        fast: If True, return the smaller/cheaper model used by routing and
            classification agents (query understanding, router, memory manager).
    """
    s = get_settings()
    model = s.resolved_llm_fast_model if fast else s.resolved_llm_model
    common = {"temperature": s.llm_temperature, "max_tokens": s.llm_max_tokens}

    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=_require_key(s.anthropic_api_key, "ANTHROPIC_API_KEY"), **common
        )
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, api_key=_require_key(s.openai_api_key, "OPENAI_API_KEY"), **common
        )
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model, api_key=_require_key(s.groq_api_key, "GROQ_API_KEY"), **common
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {s.llm_provider}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the configured embedding model (shared by ingestion and retrieval)."""
    s = get_settings()
    model = s.resolved_embedding_model

    if s.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=_require_key(s.openai_api_key, "OPENAI_API_KEY"))
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {s.embedding_provider}")


@lru_cache(maxsize=1)
def get_reranker():
    """Return the local cross-encoder that rescores (query, chunk) pairs for reranking."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


# --------------------------------------------------------------------------- #
# Call helpers
# --------------------------------------------------------------------------- #

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def invoke_text(messages: list[BaseMessage], fast: bool = False) -> str:
    """Plain-text completion."""
    return str(get_llm(fast=fast).invoke(messages).text).strip()


def invoke_structured(schema: type[ModelT], messages: list[BaseMessage], fast: bool = True) -> ModelT:
    """Get a reply validated against a Pydantic schema, on any provider.

    1. Native structured output (tool calling on Anthropic, OpenAI and Groq).
    2. If the provider rejects the call or returns something unparseable —
       most common with small open models — ask for raw JSON and validate it.

    Raises StructuredOutputError if both attempts fail; callers fall back to
    safe defaults so a flaky model never breaks the conversation.
    """
    llm = get_llm(fast=fast)
    try:
        result = llm.with_structured_output(schema, include_raw=True).invoke(messages)
        if isinstance(result.get("parsed"), schema):
            return result["parsed"]
        logger.warning("%s: structured output unparseable (%s); retrying as JSON",
                       schema.__name__, result.get("parsing_error"))
    except Exception as exc:  # providers raise different types for rejected tool calls
        logger.warning("%s: structured output failed (%s); retrying as JSON", schema.__name__, exc)

    instruction = HumanMessage(
        "Reply with only a JSON object - no prose, no code fences - that matches this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    text = str(llm.invoke([*messages, instruction]).text)
    match = _JSON_OBJECT_RE.search(text)
    try:
        return schema.model_validate_json(match.group(0) if match else text)
    except Exception as exc:
        raise StructuredOutputError(f"{schema.__name__}: could not parse model reply: {text[:200]!r}") from exc
