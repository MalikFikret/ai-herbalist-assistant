"""Cached LLM clients, retriever, and runtime config resolvers.

All node modules pull their LLMs and retriever from this module so the live
chat path benefits from ``@lru_cache``-backed reuse, and so a single
``reset_runtime_caches()`` call can flush every cache atomically (used by the
admin "Re-index" action).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from herbalist_assistant import config
from herbalist_assistant.graph.state import AgentState
from herbalist_assistant.llm.groq import create_groq_llm, get_groq_api_key

_logger = logging.getLogger(__name__)

_GROQ_MODELS = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile"}
DEFAULT_WEB_SEARCH_PROVIDER = "Tavily"


def _resolve_model_name(state: AgentState) -> str:
    candidate = str(state.get("model_name", "") or "").strip()
    return candidate or config.GROQ_MODEL


def _resolve_web_search_provider(state: AgentState) -> str:
    provider = str(state.get("web_search_provider", "") or "").strip()
    return provider or DEFAULT_WEB_SEARCH_PROVIDER


def _get_required_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"{var_name} is not set. Please add it to your .env file.")
    return value


def _create_chat_model(*, model_name: str, temperature: float):
    if model_name in _GROQ_MODELS:
        return create_groq_llm(
            api_key=get_groq_api_key(),
            model_name=model_name,
            temperature=temperature,
        )
    if model_name == "gemini-1.5-flash":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=_get_required_env("GEMINI_API_KEY"),
        )
    if model_name == "deepseek-chat":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=_get_required_env("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    _logger.warning("Unknown model '%s'; falling back to %s", model_name, config.GROQ_MODEL)
    return create_groq_llm(
        api_key=get_groq_api_key(),
        model_name=config.GROQ_MODEL,
        temperature=temperature,
    )


@lru_cache(maxsize=8)
def _router_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.0)


@lru_cache(maxsize=8)
def _expansion_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.4)


@lru_cache(maxsize=8)
def _grader_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=0.0)


@lru_cache(maxsize=8)
def _generator_llm(model_name: str):
    return _create_chat_model(model_name=model_name, temperature=config.LLM_TEMPERATURE)


@lru_cache(maxsize=1)
def _retriever():
    from herbalist_assistant.rag.embeddings import create_embeddings
    from herbalist_assistant.rag.vectorstore import load_or_build_vectorstore, make_retriever

    embeddings = create_embeddings(config.EMBEDDING_MODEL)
    vectorstore = load_or_build_vectorstore(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_DIR,
        embeddings=embeddings,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return make_retriever(vectorstore, k=config.RETRIEVER_K)


def reset_runtime_caches() -> None:
    """Invalidate all module-level LLM and retriever caches.

    Called by the admin "Re-index" action so that a fresh vectorstore is
    actually used by the live chat path (not just by the Streamlit caches).
    """
    _router_llm.cache_clear()
    _expansion_llm.cache_clear()
    _grader_llm.cache_clear()
    _generator_llm.cache_clear()
    _retriever.cache_clear()
