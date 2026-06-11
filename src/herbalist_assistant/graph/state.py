"""Shared state schema for the agentic-RAG LangGraph workflow.

Every node reads/writes through ``AgentState``. The literals describe the two
binary decisions that show up in router and grader outputs.

Phase-3 additions
─────────────────
* ``web_search_queries``  — list of queries actually sent to the web-search
  provider.  Written by web_search_node; read by the hallucination grader
  and retry rewriter so they know what was already searched and can avoid
  repeating the same queries on the next attempt.

* ``answer_source_type``  — one of ``"local_rag"``, ``"web_search"``, or
  ``"direct"`` (set by each terminal node).  Useful for logging, admin
  analytics, and future A/B testing of retrieval strategies.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document

RouteLiteral = Literal["VECTOR_SEARCH", "DIRECT_ANSWER"]
ScoreLiteral = Literal["yes", "no"]

# Phase-3: typed alias for the answer_source_type field
AnswerSourceType = Literal["local_rag", "web_search", "direct", "unknown"]


class AgentState(TypedDict, total=False):
    """State shared across all nodes in the advanced LangGraph workflow."""

    # ── input fields ──────────────────────────────────────────────────────────
    question: str
    # User health profile metadata (e.g., age, allergies, conditions).
    user_profile: dict[str, Any]
    # Recent chat turns as [{"role": "user"|"assistant", "content": str}, ...].
    # The CURRENT user question is NOT in this list; it lives in `question`.
    chat_history: list[dict]
    # Runtime model id chosen by the sidebar settings.
    model_name: str
    # Runtime web-search backend chosen by the sidebar settings.
    web_search_provider: str
    # Optional UI language preference (for retrieval language priority fallback).
    ui_language: str

    # ── retrieval fields ──────────────────────────────────────────────────────
    expanded_queries: list[str]
    # Initial top-k retrieved docs before any grading/filtering.
    candidate_docs: list[Document]
    # Docs that passed the CRAG relevance grader (score > threshold).
    selected_docs: list[Document]
    # All documents available to the generator (either selected_docs or web results).
    documents: list[Document]

    # ── web-search fields (Phase-3) ───────────────────────────────────────────
    # Exact queries sent to the web-search provider by web_search_node.
    # Allows the hallucination grader and retry rewriter to inspect / avoid
    # repeating the same queries on subsequent attempts.
    web_search_queries: list[str]

    # ── routing / grading fields ──────────────────────────────────────────────
    is_medical: bool
    # Hallucination check result from the evaluator ("yes" = grounded, "no" = not).
    hallucination_score: bool | str
    # Whether the final answer addresses the user's intent ("yes" / "no").
    answer_relevance_score: bool | str
    # Free-text evaluator feedback from answer_relevance_node.
    answer_relevance_feedback: str
    # Number of post-generation retry attempts (starts at 0).
    generation_retries: int

    # ── answer fields ─────────────────────────────────────────────────────────
    direct_answer: str
    final_answer: str

    # ── provenance field (Phase-3) ────────────────────────────────────────────
    # Records which path produced the final answer.
    # Values: "local_rag" | "web_search" | "direct" | "unknown"
    # Written by generate_medical_answer_node, direct_answer_node, and
    # web_search_node (the latter sets "web_search" before the generator runs).
    answer_source_type: AnswerSourceType