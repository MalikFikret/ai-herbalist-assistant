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

from langgraph.graph import END, StateGraph

from herbalist_assistant.graph.nodes import (
    _route_after_grading,
    _route_from_router,
    direct_answer_node,
    expand_and_retrieve_node,
    generate_medical_answer_node,
    grade_documents_node,
    route_question,
    web_search_node,
)
from herbalist_assistant.graph.runtime import reset_runtime_caches
from herbalist_assistant.graph.state import AgentState

__all__ = ["AgentState", "app", "reset_runtime_caches"]


workflow = StateGraph(AgentState)

workflow.add_node("route_question", route_question)
workflow.add_node("direct_answer_node", direct_answer_node)
workflow.add_node("expand_and_retrieve_node", expand_and_retrieve_node)
workflow.add_node("grade_documents_node", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_medical_answer_node", generate_medical_answer_node)

workflow.set_entry_point("route_question")
workflow.add_conditional_edges(
    "route_question",
    _route_from_router,
    {"medical": "expand_and_retrieve_node", "direct": "direct_answer_node"},
)
workflow.add_edge("direct_answer_node", END)
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
workflow.add_edge("generate_medical_answer_node", END)

app = workflow.compile()
