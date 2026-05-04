"""Compiled LangGraph workflow for the agentic-RAG pipeline.

This module wires the per-node modules in ``herbalist_assistant.graph.nodes``
into a ``StateGraph`` and exposes the compiled ``app`` symbol consumed by:

* the Streamlit UI (``herbalist_assistant.ui.streamlit_app``),
* the LangGraph deployment manifest (``langgraph.json``),
* the diagram script (``scripts/visualize_graph.py``).

``reset_runtime_caches`` is re-exported here so that existing callers that
import it from ``advanced_graph`` keep working after the split.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from herbalist_assistant.graph.nodes import (
    _route_after_grading,
    _route_after_hallucination,
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


# 1. Document Relevance Grader Model
class DocumentGrade(BaseModel):
    """Evaluates how relevant a retrieved document is to the user's question."""

    score: int = Field(description="Relevance score from 0 to 100")
    reasoning: str = Field(description="A brief explanation for the given score")


# 2. Hallucination Grader Model
class HallucinationGrade(BaseModel):
    """Checks if the generated answer is grounded in the retrieved documents or if it contains hallucinations."""

    binary_score: Literal["yes", "no"] = Field(
        description="Is the answer fully supported by the sources? 'yes' means it is supported."
    )
    reasoning: str = Field(description="Explanation of any discrepancies, if present")


# 3. Answer Relevance Grader Model
class AnswerRelevanceGrade(BaseModel):
    """Evaluates how well the final answer addresses the user's original question."""

    reasoning: str = Field(
        description="Step-by-step explanation of why the answer is or isn't relevant to the user's question. Think before scoring."
    )
    binary_score: Literal["yes", "no"] = Field(
        description="Does the answer actually address the user's intent? 'yes' means it does."
    )
    feedback: str = Field(
        default="",
        description="If 'no', provide a short actionable reason. If 'yes', this can be empty or 'Good'.",
    )


workflow = StateGraph(AgentState)

workflow.add_node("route_question", route_question)
workflow.add_node("direct_answer_node", direct_answer_node)
workflow.add_node("expand_and_retrieve_node", expand_and_retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_medical_answer_node", generate_medical_answer_node)
workflow.add_node("hallucination_grader_node", hallucination_grader_node)
workflow.add_node("answer_relevance_node", answer_relevance_node)

workflow.set_entry_point("route_question")
workflow.add_conditional_edges(
    "route_question",
    _route_from_router,
    {"medical": "expand_and_retrieve_node", "direct": "direct_answer_node"},
)
workflow.add_edge("direct_answer_node", "answer_relevance_node")
workflow.add_edge("expand_and_retrieve_node", "grade_documents_node")

# Explicit conditional branch after CRAG grading.
workflow.add_conditional_edges(
    "grade_documents_node",
    _route_after_grading,
    {
        "has_docs": "generate_medical_answer_node",
        "no_docs": "web_search_node",
    },
)
workflow.add_edge("web_search_node", "generate_medical_answer_node")
workflow.add_edge("generate_medical_answer_node", "hallucination_grader_node")
workflow.add_conditional_edges(
    "hallucination_grader_node",
    _route_after_hallucination,
    {
        "answer_relevance": "answer_relevance_node",
        "retry_web": "web_search_node",
    },
)
workflow.add_edge("answer_relevance_node", END)

app = workflow.compile()
