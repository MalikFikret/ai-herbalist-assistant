"""CRAG grading node + the "has_docs / no_docs" branch helper.

Each candidate document is scored 0-100 by an LLM; only high-scoring
documents survive to the medical-answer node, otherwise the workflow falls
back to web search.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.runtime import _grader_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


GRADER_SYSTEM = """You are a strict relevance grader for herbal-health RAG.

Score how relevant a document is to the user's question on a scale from 0 to 100.
Use higher scores only when the document clearly and directly supports answering
the specific question. Penalize vague, generic, or tangential content.
Provide brief reasoning for the score."""

# Candidate docs are already capped upstream; keep the grading cap aligned.
_MAX_GRADE_DOCS = 10

# Cap the number of selected documents handed to the answer node, to keep
# the final prompt within model context budgets.
_MAX_KEPT_DOCS = 3
_PASSING_SCORE_THRESHOLD = 70

HALLUCINATION_GRADER_SYSTEM = """You are a strict grounding evaluator.

Check whether the answer is fully supported by the provided source documents.
Return "yes" ONLY when every substantive claim in the answer is grounded in
the sources. Return "no" if there are unsupported claims, speculation, or
fabricated details."""

ANSWER_RELEVANCE_SYSTEM = """You evaluate whether an AI assistant's answer addresses the user's intent.

CRITICAL INSTRUCTIONS FOR GRADING:
1. Be highly lenient. If the answer provides helpful, relevant information to the user's query, you MUST return "yes".
2. Do NOT penalize the answer for including extra details, conversational elements (like greetings or creator info), or standard medical disclaimers. 
3. Return "no" ONLY if there is a catastrophic failure, such as:
   - The answer is completely off-topic or evasive.
   - It directly contradicts the user's core request.
   - It ignores a critical user safety constraint (e.g., an explicit allergy).

First, write out your reasoning, then provide the final binary score."""


def _grade_document(question: str, document_text: str, *, model_name: str) -> int | None:
    if not question.strip() or not document_text.strip():
        return None

    grader_human = f"User query:\n{question}\n\nDocument text:\n{document_text}"
    try:
        # Import lazily to avoid module import cycles with advanced_graph wiring.
        from herbalist_assistant.graph.advanced_graph import DocumentGrade

        response = _grader_llm(model_name).with_structured_output(DocumentGrade).invoke(
            [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=grader_human)],
        )
        score = int(getattr(response, "score", 0))
        # Keep score bounded in case a model returns out-of-range values.
        return max(0, min(100, score))
    except Exception:
        _logger.exception("Grader LLM failed; rejecting document")
        # Fail-closed: drop the document on a transient error so web search can trigger.
        return None


def _extract_latest_answer(state: AgentState) -> str:
    """Get the latest assistant answer from state across route types."""
    final_answer = str(state.get("final_answer", "")).strip()
    if final_answer:
        return final_answer

    direct_answer = str(state.get("direct_answer", "")).strip()
    if direct_answer:
        return direct_answer

    history = state.get("chat_history", []) or []
    if isinstance(history, list):
        for msg in reversed(history):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role == "assistant" and content:
                return content
    return ""


def grade_documents_node(state: AgentState) -> AgentState:
    """Score candidate docs, keep >70, sort by score, and return top 3."""
    question = str(state.get("question", "")).strip()
    documents = state.get("candidate_docs", []) or []
    if not question or not documents:
        return {"selected_docs": []}

    if len(documents) > _MAX_GRADE_DOCS:
        _logger.info(
            "Capping grading candidates from %d to %d",
            len(documents),
            _MAX_GRADE_DOCS,
        )
        documents = documents[:_MAX_GRADE_DOCS]

    model_name = _resolve_model_name(state)
    scored_docs: list[tuple[int, Document]] = []
    for doc in documents:
        score = _grade_document(question, doc.page_content, model_name=model_name)
        if score is None or score <= _PASSING_SCORE_THRESHOLD:
            continue
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda pair: pair[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:_MAX_KEPT_DOCS]]
    return {"selected_docs": top_docs}


def hallucination_grader_node(state: AgentState) -> AgentState:
    """Grade whether the generated answer is fully grounded in selected docs."""
    answer = _extract_latest_answer(state)
    selected_docs = state.get("selected_docs", []) or []
    if not answer or not selected_docs:
        retries = int(state.get("generation_retries", 0) or 0)
        if retries < 3:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        return {"hallucination_score": "no"}

    context_parts: list[str] = []
    for idx, doc in enumerate(selected_docs, start=1):
        source = str(doc.metadata.get("source", "unknown")) if doc.metadata else "unknown"
        context_parts.append(f"[Doc {idx} | {source}]\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    model_name = _resolve_model_name(state)
    prompt = f"Answer:\n{answer}\n\nSupporting documents:\n{context}"
    try:
        # Import lazily to avoid module import cycles with advanced_graph wiring.
        from herbalist_assistant.graph.advanced_graph import HallucinationGrade

        grade = _grader_llm(model_name).with_structured_output(HallucinationGrade).invoke(
            [SystemMessage(content=HALLUCINATION_GRADER_SYSTEM), HumanMessage(content=prompt)],
        )
        binary_score = str(getattr(grade, "binary_score", "no")).strip().lower()
        if binary_score == "yes":
            return {"hallucination_score": "yes"}

        retries = int(state.get("generation_retries", 0) or 0)
        if retries < 3:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        return {"hallucination_score": "no"}
    except Exception:
        _logger.exception("Hallucination grading failed; defaulting to 'no'")
        retries = int(state.get("generation_retries", 0) or 0)
        if retries < 3:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        return {"hallucination_score": "no"}


def answer_relevance_node(state: AgentState) -> AgentState:
    """Grade whether the generated answer matches the user's original intent."""
    question = str(state.get("question", "")).strip()
    answer = _extract_latest_answer(state)
    if not question or not answer:
        return {
            "answer_relevance_score": "no",
            "answer_relevance_feedback": "Missing question or answer for relevance evaluation.",
        }

    model_name = _resolve_model_name(state)
    prompt = f"User question:\n{question}\n\nGenerated answer:\n{answer}"
    try:
        # Import lazily to avoid module import cycles with advanced_graph wiring.
        from herbalist_assistant.graph.advanced_graph import AnswerRelevanceGrade

        grade = _grader_llm(model_name).with_structured_output(AnswerRelevanceGrade).invoke(
            [SystemMessage(content=ANSWER_RELEVANCE_SYSTEM), HumanMessage(content=prompt)],
        )
        binary_score = str(getattr(grade, "binary_score", "no")).strip().lower()
        feedback = str(getattr(grade, "feedback", "")).strip()
        if not feedback:
            feedback = "The answer did not fully satisfy the user's request."

        # Turn the relevance evaluator into a user-visible quality gate for the
        # medical route. For direct/small-talk route we keep the original answer.
        if binary_score == "no" and bool(state.get("is_medical")):
            rejection_message = (
                "I could not confidently provide a sufficiently relevant herbal answer "
                "for your request this time.\n\n"
                f"Reason: {feedback}\n\n"
                "Please try rephrasing your question with more detail about your "
                "symptom, conditions, allergies, and what kind of remedy you want."
            )
            return {
                "answer_relevance_score": "no",
                "answer_relevance_feedback": feedback,
                "final_answer": rejection_message,
            }

        return {
            "answer_relevance_score": "yes" if binary_score == "yes" else "no",
            "answer_relevance_feedback": feedback,
        }
    except Exception:
        _logger.exception("Answer relevance grading failed; defaulting to 'no'")
        return {
            "answer_relevance_score": "no",
            "answer_relevance_feedback": "Relevance evaluator failed unexpectedly.",
        }


def _route_after_grading(state: AgentState) -> Literal["has_docs", "no_docs"]:
    return "has_docs" if bool(state.get("selected_docs")) else "no_docs"


def _route_after_hallucination(state: AgentState) -> Literal["answer_relevance", "retry_web"]:
    """Route to relevance check when grounded or retries exhausted; else retry web."""
    hallucination_score = str(state.get("hallucination_score", "no")).strip().lower()
    if hallucination_score == "yes":
        return "answer_relevance"

    retries = int(state.get("generation_retries", 0) or 0)
    if retries >= 3:
        return "answer_relevance"
    return "retry_web"
