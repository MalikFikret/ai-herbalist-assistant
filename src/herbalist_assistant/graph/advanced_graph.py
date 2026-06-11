"""Compiled LangGraph workflow for the agentic-RAG pipeline.

FIX (v2): Pydantic schemas cleaned up + graph wiring clarified.
NOTE: direct_answer_node intentionally routes through answer_relevance_node
      (not directly to END) — this acts as a safety gate to catch insults,
      inappropriate content, or cases where the first LLM misunderstood the
      message. The answer_relevance grader is the final sanity check for ALL
      responses, including direct ones.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from herbalist_assistant.graph.nodes import (
    _route_after_grading,
    _route_after_hallucination,
    _route_after_web_search,
    _route_from_router,
    answer_relevance_node,
    direct_answer_node,
    expand_and_retrieve_node,
    generate_medical_answer_node,
    grade_documents_node,
    hallucination_grader_node,
    route_question,
    web_search_node,
)
from herbalist_assistant.graph.runtime import reset_runtime_caches
from herbalist_assistant.graph.state import AgentState

__all__ = ["AgentState", "app", "reset_runtime_caches"]


# ── Pydantic grading schemas ──────────────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Evaluates how relevant a retrieved document is to the user's question."""
    score: int = Field(description="Relevance score from 0 to 100")
    reasoning: str = Field(description="A brief explanation for the given score")


class HallucinationGrade(BaseModel):
    """Checks if the generated answer is grounded in the retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Is the answer fully supported by the sources? 'yes' means it is supported."
    )
    reasoning: str = Field(description="Explanation of any discrepancies, if present")


class AnswerRelevanceGrade(BaseModel):
    """Evaluates how well the final answer addresses the user's original question."""
    reasoning: str = Field(
        description=(
            "Step-by-step explanation of why the answer is or isn't relevant "
            "to the user's question. Think before scoring."
        )
    )
    binary_score: Literal["yes", "no"] = Field(
        description="Does the answer actually address the user's intent? 'yes' means it does."
    )
    feedback: str = Field(
        default="",
        description=(
            "If 'no', provide a short actionable reason. "
            "If 'yes', this can be empty or 'Good'."
        ),
    )


# ── Graph wiring ──────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("route_question",               route_question)
workflow.add_node("direct_answer_node",           direct_answer_node)
workflow.add_node("expand_and_retrieve_node",     expand_and_retrieve_node)
workflow.add_node("grade_documents_node",         grade_documents_node)
workflow.add_node("web_search_node",              web_search_node)
workflow.add_node("generate_medical_answer_node", generate_medical_answer_node)
workflow.add_node("hallucination_grader_node",    hallucination_grader_node)
workflow.add_node("answer_relevance_node",        answer_relevance_node)

# Entry point
workflow.set_entry_point("route_question")

# Route: medical → RAG path | direct → answer immediately
workflow.add_conditional_edges(
    "route_question",
    _route_from_router,
    {
        "medical": "expand_and_retrieve_node",
        "direct": "direct_answer_node",
    },
)

# direct_answer_node → answer_relevance_node (intentional safety gate)
# Even greetings/small-talk pass through the relevance grader to catch
# insults, inappropriate content, or LLM misunderstandings before the
# response reaches the user.
workflow.add_edge("direct_answer_node", "answer_relevance_node")

# RAG path
workflow.add_edge("expand_and_retrieve_node", "grade_documents_node")

workflow.add_conditional_edges(
    "grade_documents_node",
    _route_after_grading,
    {
        "has_docs": "generate_medical_answer_node",
        "no_docs":  "web_search_node",
    },
)

workflow.add_conditional_edges(
    "web_search_node",
    _route_after_web_search,
    {
        "has_web_docs": "generate_medical_answer_node",
        "no_web_docs":  END,
    },
)
workflow.add_edge("generate_medical_answer_node", "hallucination_grader_node")

workflow.add_conditional_edges(
    "hallucination_grader_node",
    _route_after_hallucination,
    {
        "answer_relevance": "answer_relevance_node",
        "retry_web":        "web_search_node",
    },
)

workflow.add_edge("answer_relevance_node", END)

app = workflow.compile()