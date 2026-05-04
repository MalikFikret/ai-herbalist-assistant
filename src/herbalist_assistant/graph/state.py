"""Shared state schema for the agentic-RAG LangGraph workflow.

Every node reads/writes through ``AgentState``. The literals describe the two
binary decisions that show up in router and grader outputs.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document

RouteLiteral = Literal["VECTOR_SEARCH", "DIRECT_ANSWER"]
ScoreLiteral = Literal["yes", "no"]


class AgentState(TypedDict, total=False):
    """State shared across all nodes in the advanced LangGraph workflow."""

    question: str
    # User health profile metadata (e.g., age, allergies, conditions).
    user_profile: dict[str, Any]
    expanded_queries: list[str]
    documents: list[Document]
    # Initial top-10 retrieved docs before any grading/filtering.
    candidate_docs: list[Document]
    # Docs selected after grading (score > 70/100).
    selected_docs: list[Document]
    is_medical: bool
    # Hallucination check result from the evaluator.
    hallucination_score: bool | str
    # Whether the final answer matches the user's intent.
    answer_relevance_score: bool | str
    # Free-text evaluator feedback from answer_relevance_node.
    answer_relevance_feedback: str
    # Number of post-generation retry attempts (starts at 0).
    generation_retries: int
    direct_answer: str
    final_answer: str
    # Recent chat turns as [{"role": "user"|"assistant", "content": str}, ...].
    # The CURRENT user question is NOT in this list; it lives in `question`.
    chat_history: list[dict]
    # Runtime model id chosen by the sidebar settings.
    model_name: str
    # Runtime web-search backend chosen by the sidebar settings.
    web_search_provider: str
