"""Web search fallback node.

When CRAG grading drops every retrieved document, this node runs a web
search (Tavily or DuckDuckGo) and converts the results into ``Document``
objects so the medical-answer node can treat them like any other context.
"""

from __future__ import annotations

import logging

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

from herbalist_assistant.graph.runtime import (
    DEFAULT_WEB_SEARCH_PROVIDER,
    _get_required_env,
    _resolve_web_search_provider,
)
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)

_WEB_SEARCH_RESULT_LIMIT = 4


def _normalize_web_search_results(raw_results, *, provider_name: str) -> list[Document]:
    source_label = f"Web Search ({provider_name})"
    docs: list[Document] = []

    if isinstance(raw_results, str):
        text = raw_results.strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": source_label}))
        return docs

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
                },
            )
        )

    return docs


def web_search_node(state: AgentState) -> AgentState:
    """Fallback web search when all local docs are filtered out."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"documents": []}

    provider_name = _resolve_web_search_provider(state)
    try:
        if provider_name == "Tavily":
            from langchain_tavily import TavilySearch

            # TavilySearch reads the key from TAVILY_API_KEY env var.
            _get_required_env("TAVILY_API_KEY")  # raises early if missing
            search_tool = TavilySearch(max_results=_WEB_SEARCH_RESULT_LIMIT)
            raw_results = search_tool.invoke(question)
        else:
            provider_name = DEFAULT_WEB_SEARCH_PROVIDER
            search_tool = DuckDuckGoSearchRun()
            raw_results = search_tool.invoke(question)
    except Exception:
        _logger.exception("Web search failed with provider=%s", provider_name)
        return {"documents": []}

    web_docs = _normalize_web_search_results(raw_results, provider_name=provider_name)
    if not web_docs:
        return {"documents": []}
    return {"documents": web_docs}
