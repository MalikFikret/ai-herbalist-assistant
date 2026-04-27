"""CRAG grading node + the "has_docs / no_docs" branch helper.

Each retrieved document is graded yes/no by an LLM; only the top-rated
documents survive to the medical-answer node, otherwise the workflow falls
back to web search.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import _extract_score
from herbalist_assistant.graph.runtime import _grader_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState, ScoreLiteral

_logger = logging.getLogger(__name__)


GRADER_SYSTEM = """You are a strict relevance grader for herbal-health RAG.

Decide if a document EXPLICITLY contains information that directly answers the user query.
Answer "yes" ONLY if the document is highly relevant and directly helps form the answer.
Answer "no" if the document is only vaguely related, general, or does not contain the specific answer.

Return only JSON:
{"score": "yes"} or {"score": "no"}"""

# Cap documents sent to the CRAG grading loop to limit serial LLM calls.
_MAX_GRADE_DOCS = 8

# Cap the number of graded-yes documents handed to the answer node, to keep
# the final prompt within model context budgets.
_MAX_KEPT_DOCS = 3


def _grade_document(question: str, document_text: str, *, model_name: str) -> ScoreLiteral:
    if not question.strip() or not document_text.strip():
        return "no"

    grader_human = f"User query:\n{question}\n\nDocument text:\n{document_text}"
    try:
        response = _grader_llm(model_name).invoke(
            [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=grader_human)]
        )
        return _extract_score(getattr(response, "content", str(response)))
    except Exception:
        _logger.exception("Grader LLM failed; rejecting document (defaulting to 'no')")
        # Fail-closed: drop the document on a transient error so web search can trigger.
        return "no"


def grade_documents_node(state: AgentState) -> AgentState:
    """Keep only documents that pass CRAG grading ("yes")."""
    question = str(state.get("question", "")).strip()
    documents = state.get("documents", []) or []
    if not question or not documents:
        return {"documents": []}

    if len(documents) > _MAX_GRADE_DOCS:
        _logger.info(
            "Capping grading candidates from %d to %d",
            len(documents),
            _MAX_GRADE_DOCS,
        )
        documents = documents[:_MAX_GRADE_DOCS]

    model_name = _resolve_model_name(state)
    filtered: list[Document] = []
    for doc in documents:
        if _grade_document(question, doc.page_content, model_name=model_name) == "yes":
            filtered.append(doc)
    return {"documents": filtered[:_MAX_KEPT_DOCS]}


def _route_after_grading(state: AgentState) -> Literal["has_docs", "no_docs"]:
    return "has_docs" if bool(state.get("documents")) else "no_docs"
