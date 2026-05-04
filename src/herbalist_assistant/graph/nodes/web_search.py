"""Web search fallback node.

When CRAG grading drops every retrieved document, this node runs a web
search (Tavily or DuckDuckGo) and converts the results into ``Document``
objects so the medical-answer node can treat them like any other context.
"""

from __future__ import annotations

import json
import logging

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _dedupe_documents, _format_chat_history
from herbalist_assistant.graph.runtime import (
    _expansion_llm,
    _get_required_env,
    _resolve_model_name,
    _resolve_web_search_provider,
)
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)

_WEB_SEARCH_RESULT_LIMIT = 4
_PROVIDER_TAVILY = "Tavily"
_PROVIDER_DUCK = "DuckDuckGo"

RETRY_WEB_QUERY_SYSTEM = """You rewrite web search queries for a herbal assistant.

The previous attempt did not yield grounded enough results. Produce up to 3 NEW
queries that are meaningfully different from the original phrasing, mixing
broader and more specific variants to improve recall and precision.

Avoid returning the exact original query. Return only JSON:
{"queries": ["q1", "q2", "q3"]}"""


def _extract_retry_queries(raw: str, original_question: str) -> list[str]:
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        queries = data.get("queries", [])
    except Exception:
        return [original_question]

    if not isinstance(queries, list):
        return [original_question]

    seen: set[str] = {original_question.strip().lower()}
    result: list[str] = []
    for item in queries:
        q = str(item).strip()
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        result.append(q)
        if len(result) == 3:
            break
    return result or [original_question]


def _build_web_search_queries(state: AgentState, question: str) -> list[str]:
    retries = int(state.get("generation_retries", 0) or 0)
    if retries <= 0:
        return [question]

    history_block = _format_chat_history(state.get("chat_history"))
    human_parts = [f"Original user question:\n{question}", f"Current retry count:\n{retries}"]
    if history_block:
        human_parts.insert(1, f"Recent conversation:\n{history_block}")
    prompt = "\n\n".join(human_parts)

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=RETRY_WEB_QUERY_SYSTEM), HumanMessage(content=prompt)]
        )
        return _extract_retry_queries(getattr(response, "content", str(response)), question)
    except Exception:
        _logger.exception("Retry web query rewriting failed; using original question")
        return [question]


def _normalize_web_search_results(raw_results, *, provider_name: str) -> list[Document]:
    source_label = f"Web Search ({provider_name})"
    docs: list[Document] = []

    if isinstance(raw_results, str):
        text = raw_results.strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": source_label}))
        return docs

    if isinstance(raw_results, dict):
        # TavilySearch returns a dict like:
        # {"query": "...", "results": [...], "answer": "...", ...}
        error_text = str(raw_results.get("error", "")).strip()
        if error_text:
            _logger.warning("%s returned error payload: %s", source_label, error_text)
            return docs

        answer_text = str(raw_results.get("answer", "")).strip()
        if answer_text:
            docs.append(
                Document(
                    page_content=f"Answer: {answer_text}",
                    metadata={"source": source_label},
                )
            )

        if isinstance(raw_results.get("results"), list):
            raw_results = raw_results["results"]
        else:
            raw_results = [raw_results]

    if not isinstance(raw_results, list):
        return docs

    for item in raw_results:
        if isinstance(item, str):
            text = item.strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": source_label}))
            continue

        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        snippet = str(
            item.get("content")
            or item.get("snippet")
            or item.get("body")
            or ""
        ).strip()
        url = str(item.get("url", "")).strip()
        if not (title or snippet or url):
            continue

        body_parts: list[str] = []
        if title:
            body_parts.append(f"Title: {title}")
        if snippet:
            body_parts.append(f"Snippet: {snippet}")
        if url:
            body_parts.append(f"URL: {url}")

        docs.append(
            Document(
                page_content="\n".join(body_parts),
                metadata={
                    "source": source_label,
                    "title": title,
                    "url": url,
                    "score": item.get("score"),
                },
            )
        )

    return docs


def _search_with_duck(queries: list[str]) -> list[Document]:
    search_tool = DuckDuckGoSearchRun()
    docs: list[Document] = []
    for query in queries:
        raw_results = search_tool.invoke(query)
        docs.extend(_normalize_web_search_results(raw_results, provider_name=_PROVIDER_DUCK))
    return docs


def _search_with_tavily(queries: list[str]) -> list[Document]:
    from langchain_tavily import TavilySearch

    # TavilySearch reads the key from TAVILY_API_KEY env var.
    _get_required_env("TAVILY_API_KEY")
    search_tool = TavilySearch(max_results=_WEB_SEARCH_RESULT_LIMIT)
    docs: list[Document] = []
    for query in queries:
        if len(query) > 400:
            query = query[:400]
        raw_results = search_tool.invoke(query)
        docs.extend(_normalize_web_search_results(raw_results, provider_name=_PROVIDER_TAVILY))
    return docs


def web_search_node(state: AgentState) -> AgentState:
    """Fallback web search when all local docs are filtered out."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"documents": []}

    queries = _build_web_search_queries(state, question)
    resolved = _resolve_web_search_provider(state)
    docs: list[Document] = []
    try:
        if resolved == _PROVIDER_TAVILY:
            try:
                docs = _search_with_tavily(queries)
            except Exception:
                _logger.warning(
                    "Tavily web search failed; falling back to DuckDuckGo",
                    exc_info=True,
                )
                docs = _search_with_duck(queries)
        else:
            docs = _search_with_duck(queries)
    except Exception:
        _logger.exception("Web search failed after provider=%s", resolved)
        return {"documents": []}

    web_docs = _dedupe_documents(docs)
    if not web_docs:
        return {"documents": []}
    return {"documents": web_docs}
