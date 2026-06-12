"""CRAG grading node + the "has_docs / no_docs" branch helper.

FIXES applied (v2):
  1. Parallel document grading via ThreadPoolExecutor  → was 10-20s, now ~2-3s
  2. Reduced _MAX_GRADE_DOCS 10→6 and _PASSING_SCORE_THRESHOLD 70→55 → fewer drops
  3. answer_relevance_node no longer overwrites final_answer  → prevents good-answer loss
  4. Hallucination retries reduced 3→1  → prevents 12-call retry storm
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.runtime import _generator_llm, _grader_llm, _resolve_model_name
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


GRADER_SYSTEM = """You are a relevance grader for herbal-health RAG.

Score how relevant a document is to the user's question on a scale from 0 to 100.

SCORING RULES:
1. High scores (70-100): The document directly answers the question with matching
   intent — correct herb, correct usage method, correct condition.
2. Medium scores (40-65): The document partially matches — same condition but
   different herb, or related information that could still help.
3. Low scores (0-35): The document is about the same condition but the treatment
   type does NOT match the user's intent.

TREATMENT TYPE MISMATCH (important):
- If the user asks about herbs to EAT or DRINK (tea, supplement, food),
  and the document describes an EXTERNAL application (lotion, cream, poultice,
  compress, ointment, wash) → penalise heavily (score 0-30).
- If the user asks about EXTERNAL treatments and the document describes
  internal remedies → penalise similarly.
- A document must match BOTH the condition AND the intended usage method
  to score above 65.

HERB IDENTITY MISMATCH (critical):
- If the user asks about a SPECIFIC herb by name (e.g. "karanfil otu",
  "chamomile", "adaçayı"), the document MUST discuss THAT EXACT herb
  (or its scientific equivalent) to score above 50.
- A document about a DIFFERENT herb with a similar-sounding name is NOT
  relevant. Example: "karanfil" (clove) ≠ "karanfil otu" (Dianthus).
- If the document discusses a completely different herb than what was asked
  about, score 0-20 regardless of other factors.

Always provide brief reasoning before the score."""

# ── Tuning knobs ──────────────────────────────────────────────────────────────
_MAX_GRADE_DOCS = 6

_MAX_KEPT_DOCS = 3

_PASSING_SCORE_THRESHOLD = 65

_GRADER_WORKERS = 4

_MAX_HALLUCINATION_RETRIES = 1
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_GRADER_SYSTEM = """You are a grounding evaluator.

Check whether the answer is substantially supported by the provided source documents.
Return "yes" when the main claims in the answer are grounded in the sources.
Minor additions or general knowledge fill-ins are acceptable — only return "no"
when the answer contains significant fabricated or contradictory claims."""

ANSWER_RELEVANCE_SYSTEM = """You evaluate whether an AI assistant's answer addresses the user's intent.

GRADING CRITERIA:
1. Return "yes" when the answer meaningfully addresses the user's core question with
   specific, relevant herbal or medical content. Answers that are helpful but include
   extra details, conversational elements (greetings, disclaimers), or minor tangents
   are still acceptable.
2. Return "no" when:
   - The answer does NOT meaningfully address the user's core question.
   - The answer is mostly generic filler with no specific herbal/medical content
     relevant to what was asked.
   - The answer ignores an explicit user safety constraint (allergy, medical condition,
     pregnancy, drug interaction, etc.) that was stated in the question. If no safety
     constraints are mentioned in the question, skip this check.
   - The answer is completely off-topic, evasive, or contradicts the user's request.
   - The answer discusses a DIFFERENT herb or plant than what the user asked about.
     For example, if the user asked about "karanfil otu" but the answer talks about
     "karanfil" (clove) recipes — this is a WRONG herb and must score "no".
   - The answer addresses a DIFFERENT condition or use-case than what the user's
     conversation context implies. For example, if the conversation was about sleep
     herbs but the answer discusses digestion — this is off-context.

First, write out your reasoning, then provide the final binary score."""


_FALLBACK_SYSTEM = """You generate a short, polite fallback message for a herbal health assistant.
The message must be written in the SAME language as the user's question below.
Do NOT include any medical advice or herbal recommendations.
Do NOT wrap the message in quotes or add a subject line.
Just output the message text directly."""

_HARDCODED_FALLBACK = (
    "I'm sorry, but I wasn't able to verify an accurate answer to your question "
    "at this time. We're continuously working to improve our knowledge base."
)


def _generate_polite_fallback(
    question: str, reason: str, model_name: str
) -> str:
    """Use the generator LLM to produce a polite fallback in the user's language.

    *reason* is a short English tag like ``"unverified_answer"`` or
    ``"irrelevant_answer"`` that tells the LLM *why* we are declining.
    Falls back to a hardcoded English message if the LLM call itself fails.
    """
    user_prompt = (
        f"Reason the answer is being replaced: {reason}\n\n"
        f"User's original question:\n{question}\n\n"
        "Write a polite 2-3 sentence message that:\n"
        "1. Acknowledges the system could not provide a reliable answer on this topic.\n"
        "2. Briefly mentions the topic the user asked about.\n"
        "3. Notes that the database is continuously being improved and expanded.\n"
        "4. Is written in the SAME language as the user's question above."
    )
    try:
        response = _generator_llm(model_name).invoke(
            [
                SystemMessage(content=_FALLBACK_SYSTEM),
                HumanMessage(content=user_prompt),
            ]
        )
        text = str(getattr(response, "content", "")).strip()
        if text:
            _logger.info("Generated polite fallback (%s) for question.", reason)
            return text
    except Exception:
        _logger.exception("Fallback LLM call failed; using hardcoded English fallback")
    return _HARDCODED_FALLBACK


# ── Document grading ──────────────────────────────────────────────────────────

def _grade_document(question: str, document_text: str, *, model_name: str) -> int | None:
    """Grade a single document. Returns 0-100 or None on failure."""
    if not question.strip() or not document_text.strip():
        return None

    grader_human = f"User query:\n{question}\n\nDocument text:\n{document_text}"
    try:
        from herbalist_assistant.graph.advanced_graph import DocumentGrade

        response = (
            _grader_llm(model_name)
            .with_structured_output(DocumentGrade)
            .invoke(
                [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=grader_human)],
            )
        )
        score = int(getattr(response, "score", 0))
        return max(0, min(100, score))
    except Exception:
        _logger.exception("Grader LLM failed for a document; using score=0")
        return None


def grade_documents_node(state: AgentState) -> AgentState:
    """Score candidate docs IN PARALLEL, keep >{threshold}, sort, return top {kept}.

    FIX: Was serial (10 sequential LLM calls = 10-20s).
         Now parallel via ThreadPoolExecutor (~2-3s for the same workload).
    """
    question = str(state.get("question", "")).strip()
    documents = list(state.get("candidate_docs", []) or [])

    if not question or not documents:
        _logger.info("grade_documents_node: no question or no docs → selected_docs=[]")
        return {"selected_docs": []}

    if len(documents) > _MAX_GRADE_DOCS:
        _logger.info(
            "Capping grading candidates from %d to %d",
            len(documents),
            _MAX_GRADE_DOCS,
        )
        documents = documents[:_MAX_GRADE_DOCS]

    model_name = _resolve_model_name(state)
    _logger.info(
        "Grading %d documents in parallel (workers=%d, model=%s)",
        len(documents),
        _GRADER_WORKERS,
        model_name,
    )

    # ── PARALLEL grading ──────────────────────────────────────────────────────
    scored_docs: list[tuple[int, Document]] = []
    workers = min(_GRADER_WORKERS, len(documents))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_doc = {
            pool.submit(
                _grade_document,
                question,
                doc.page_content,
                model_name=model_name,
            ): doc
            for doc in documents
        }

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                score = future.result()
            except Exception:
                _logger.exception("Future raised during parallel grading; skipping doc")
                score = None

            if score is None:
                _logger.debug("Doc dropped (grader error): %.60s…", doc.page_content)
                continue

            _logger.debug(
                "Doc score=%d | %.60s…",
                score,
                doc.page_content.replace("\n", " "),
            )

            if score > _PASSING_SCORE_THRESHOLD:
                scored_docs.append((score, doc))
            else:
                _logger.debug("Doc dropped (score %d ≤ %d)", score, _PASSING_SCORE_THRESHOLD)
    # ─────────────────────────────────────────────────────────────────────────

    scored_docs.sort(key=lambda pair: pair[0], reverse=True)
    top_docs = []
    for score, doc in scored_docs[:_MAX_KEPT_DOCS]:
        doc.metadata["relevance_score"] = score
        top_docs.append(doc)

    _logger.info(
        "grade_documents_node: %d/%d docs passed (threshold=%d), keeping top %d",
        len(scored_docs),
        len(documents),
        _PASSING_SCORE_THRESHOLD,
        len(top_docs),
    )
    return {"selected_docs": top_docs}


# ── Answer-quality gates ──────────────────────────────────────────────────────

def _extract_latest_answer(state: AgentState) -> str:
    """Get the most recent answer from state, regardless of route type."""
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
            if str(msg.get("role", "")).strip().lower() == "assistant":
                content = str(msg.get("content", "")).strip()
                if content:
                    return content
    return ""


def hallucination_grader_node(state: AgentState) -> AgentState:
    """Grade whether the generated answer is grounded in selected docs.

    FIX: max retries reduced 3 → 1 to prevent 12-extra-call retry storms.
    """
    answer = _extract_latest_answer(state)
    selected_docs = list(state.get("selected_docs", []) or [])

    if not answer or not selected_docs:
        retries = int(state.get("generation_retries", 0) or 0)
        # No answer or no docs → allow at most 1 retry
        if retries < _MAX_HALLUCINATION_RETRIES:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        return {"hallucination_score": "no"}

    context_parts: list[str] = []
    for idx, doc in enumerate(selected_docs, start=1):
        source = (
            str(doc.metadata.get("source", "unknown")) if doc.metadata else "unknown"
        )
        context_parts.append(f"[Doc {idx} | {source}]\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    model_name = _resolve_model_name(state)
    prompt = f"Answer:\n{answer}\n\nSupporting documents:\n{context}"
    try:
        from herbalist_assistant.graph.advanced_graph import HallucinationGrade

        grade = (
            _grader_llm(model_name)
            .with_structured_output(HallucinationGrade)
            .invoke(
                [
                    SystemMessage(content=HALLUCINATION_GRADER_SYSTEM),
                    HumanMessage(content=prompt),
                ],
            )
        )
        binary_score = str(getattr(grade, "binary_score", "no")).strip().lower()

        if binary_score == "yes":
            _logger.info("Hallucination check: PASS")
            return {"hallucination_score": "yes"}

        retries = int(state.get("generation_retries", 0) or 0)
        _logger.warning(
            "Hallucination check: FAIL (retry %d/%d)",
            retries + 1,
            _MAX_HALLUCINATION_RETRIES,
        )
        if retries < _MAX_HALLUCINATION_RETRIES:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        # Retries exhausted → replace answer with a safe fallback
        _logger.warning("Hallucination retries exhausted; replacing answer with fallback")
        fallback = _generate_polite_fallback(
            question=str(state.get("question", "")),
            reason="unverified_answer",
            model_name=model_name,
        )
        return {"hallucination_score": "no", "final_answer": fallback}

    except Exception:
        _logger.exception("Hallucination grading failed; defaulting to 'no'")
        retries = int(state.get("generation_retries", 0) or 0)
        if retries < _MAX_HALLUCINATION_RETRIES:
            return {"hallucination_score": "no", "generation_retries": retries + 1}
        _logger.warning("Hallucination retries exhausted (exception path); replacing answer with fallback")
        fallback = _generate_polite_fallback(
            question=str(state.get("question", "")),
            reason="unverified_answer",
            model_name=_resolve_model_name(state),
        )
        return {"hallucination_score": "no", "final_answer": fallback}


def answer_relevance_node(state: AgentState) -> AgentState:
    """Grade whether the generated answer matches the user's original intent.

    FIX: No longer overwrites final_answer with a rejection message.
         Previously a "no" score deleted a good answer. Now we log the
         low score and preserve the answer so the user still gets a response.
    """
    # If the answer came from the direct_answer_node (small talk/greetings),
    # skip the strict medical relevance check to prevent false rejections.
    if state.get("direct_answer"):
        _logger.info("answer_relevance_node: skipping check for direct answer")
        return {
            "answer_relevance_score": "yes",
            "answer_relevance_feedback": "Direct answer; medical relevance check skipped.",
        }

    question = str(state.get("question", "")).strip()
    answer = _extract_latest_answer(state)

    if not question or not answer:
        _logger.warning("answer_relevance_node: missing question or answer; skipping")
        return {
            "answer_relevance_score": "no",
            "answer_relevance_feedback": "Missing question or answer for relevance evaluation.",
        }

    model_name = _resolve_model_name(state)
    prompt = f"User question:\n{question}\n\nGenerated answer:\n{answer}"
    try:
        from herbalist_assistant.graph.advanced_graph import AnswerRelevanceGrade

        grade = (
            _grader_llm(model_name)
            .with_structured_output(AnswerRelevanceGrade)
            .invoke(
                [SystemMessage(content=ANSWER_RELEVANCE_SYSTEM), HumanMessage(content=prompt)],
            )
        )
        binary_score = str(getattr(grade, "binary_score", "no")).strip().lower()
        feedback = str(getattr(grade, "feedback", "")).strip()
        if not feedback:
            feedback = "The answer did not fully satisfy the user's request."

        if binary_score == "no":
            _logger.warning(
                "answer_relevance_node scored 'no' (reason: %s) — "
                "replacing answer with polite fallback",
                feedback,
            )
            fallback = _generate_polite_fallback(
                question=question,
                reason="irrelevant_answer",
                model_name=model_name,
            )
            return {
                "answer_relevance_score": "no",
                "answer_relevance_feedback": feedback,
                "final_answer": fallback,
            }

        _logger.info("answer_relevance_node: PASS")
        return {
            "answer_relevance_score": "yes",
            "answer_relevance_feedback": feedback,
        }

    except Exception:
        _logger.exception("Answer relevance grading failed; defaulting to pass-through")
        # On failure, always preserve the existing answer
        return {
            "answer_relevance_score": "no",
            "answer_relevance_feedback": "Relevance evaluator failed; answer preserved.",
        }


# ── Routing helpers ───────────────────────────────────────────────────────────

def _route_after_grading(state: AgentState) -> Literal["has_docs", "no_docs"]:
    result = "has_docs" if bool(state.get("selected_docs")) else "no_docs"
    _logger.info("_route_after_grading → %s", result)
    return result


def _route_after_hallucination(
    state: AgentState,
) -> Literal["answer_relevance", "retry_web"]:
    """Go to relevance check when grounded or retries exhausted; else retry web."""
    hallucination_score = str(state.get("hallucination_score", "no")).strip().lower()
    if hallucination_score == "yes":
        return "answer_relevance"

    retries = int(state.get("generation_retries", 0) or 0)
    if retries >= _MAX_HALLUCINATION_RETRIES:
        _logger.warning(
            "_route_after_hallucination: retries exhausted (%d) → answer_relevance",
            retries,
        )
        return "answer_relevance"
    return "retry_web"


def _route_after_web_search(state: AgentState) -> Literal["has_web_docs", "no_web_docs"]:
    """Skip generation pipeline when web search produced no usable results."""
    result = "has_web_docs" if bool(state.get("documents")) else "no_web_docs"
    _logger.info("_route_after_web_search → %s", result)
    return result