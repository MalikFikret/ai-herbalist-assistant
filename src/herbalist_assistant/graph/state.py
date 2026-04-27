"""Shared state schema for the agentic-RAG LangGraph workflow.

Every node reads/writes through ``AgentState``. The literals describe the two
binary decisions that show up in router and grader outputs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document

RouteLiteral = Literal["VECTOR_SEARCH", "DIRECT_ANSWER"]
ScoreLiteral = Literal["yes", "no"]


class AgentState(TypedDict, total=False):
    """State shared across all nodes in the advanced LangGraph workflow."""

    question: str
    expanded_queries: list[str]
    documents: list[Document]
    is_medical: bool
    direct_answer: str
    final_answer: str
    # Recent chat turns as [{"role": "user"|"assistant", "content": str}, ...].
    # The CURRENT user question is NOT in this list; it lives in `question`.
    chat_history: list[dict]
    # Runtime model id chosen by the sidebar settings.
    model_name: str
    # Runtime web-search backend chosen by the sidebar settings.
    web_search_provider: str
