"""Query expansion + vector retrieval node.

Expands the user's question into three diverse search queries and then runs
each against the Chroma retriever, merging and de-duping the resulting
documents before they hit the CRAG grader.
"""

from __future__ import annotations

import json
import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import (
    _dedupe_documents,
    _format_chat_history,
)
from herbalist_assistant.graph.runtime import (
    _expansion_llm,
    _resolve_model_name,
    _retriever,
)
from herbalist_assistant.graph.state import AgentState

_logger = logging.getLogger(__name__)


EXPANSION_SYSTEM = """You generate retrieval rewrites for a herbal health assistant.

If the user's current question is a follow-up that contains pronouns or vague
references (it / this / that / the tea / the herb), USE the recent conversation
turns to resolve the referent before rewriting. For example, if the user asked
about chamomile earlier and now asks "how do I prepare it?", rewrite the three
queries around chamomile preparation, not a generic preparation query.

When a user profile is provided, incorporate medically relevant profile details
(e.g., age, allergies, conditions, medications) into the search intent only
when they are relevant to the current question.

Produce up to 3 concise, DISTINCT search queries that preserve the same user
intent while varying terminology. Use botanical, scientific, and traditional
phrasing where appropriate. If only 1-2 high-quality rewrites are appropriate,
return fewer queries.

Return only JSON:
{"expanded_queries": ["query1", "query2", "query3"]}"""


def _extract_up_to_three_queries(raw: str, fallback_question: str) -> list[str]:
    """Parse up to 3 rewritten queries from model output, with safe fallback."""
    try:
        data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        queries = data.get("expanded_queries", [])
    except Exception:
        return [fallback_question]

    if not isinstance(queries, list):
        return [fallback_question]

    cleaned: list[str] = []
    seen: set[str] = set()
    for q in queries:
        query = str(q).strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) == 3:
            break

    return cleaned or [fallback_question]


def expand_and_retrieve_node(state: AgentState) -> AgentState:
    """Expand into up-to-3 search variants and collect top candidate docs."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"expanded_queries": [], "candidate_docs": []}

    history_block = _format_chat_history(state.get("chat_history"))
    user_profile = state.get("user_profile", {})
    if not isinstance(user_profile, dict):
        user_profile = {}

    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    if user_profile:
        human_parts.append(
            "User profile (use only if relevant to this question):\n"
            f"{json.dumps(user_profile, ensure_ascii=True)}"
        )
    human_parts.append(f"Current user question:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=EXPANSION_SYSTEM), HumanMessage(content=human_message)]
        )
        expanded = _extract_up_to_three_queries(
            getattr(response, "content", str(response)),
            question,
        )
    except Exception:
        _logger.exception("Query expansion failed; falling back to original question only")
        expanded = [question]

    docs: list[Document] = []
    retriever = _retriever()
    for query in expanded:
        try:
            batch = retriever.invoke(query)
        except Exception:
            _logger.exception("Retriever.invoke failed for query=%r", query)
            continue
        if batch:
            docs.extend(batch)

    candidate_docs = _dedupe_documents(docs)[:10]
    return {"expanded_queries": expanded, "candidate_docs": candidate_docs}
