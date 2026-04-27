"""Query expansion + vector retrieval node.

Expands the user's question into three diverse search queries and then runs
each against the Chroma retriever, merging and de-duping the resulting
documents before they hit the CRAG grader.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from herbalist_assistant.graph.extractors import (
    _dedupe_documents,
    _extract_expanded_queries,
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

Produce exactly 3 concise, DISTINCT search queries that preserve the same user
intent while varying terminology. Use botanical, scientific, and traditional
phrasing where appropriate.

Return only JSON:
{"expanded_queries": ["query1", "query2", "query3"]}"""


def expand_and_retrieve_node(state: AgentState) -> AgentState:
    """Expand into 3 search variants, retrieve from Chroma, and dedupe documents."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"expanded_queries": [], "documents": []}

    history_block = _format_chat_history(state.get("chat_history"))
    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    human_parts.append(f"Current user question:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=EXPANSION_SYSTEM), HumanMessage(content=human_message)]
        )
        q1, q2, q3 = _extract_expanded_queries(getattr(response, "content", str(response)))
        expanded = [q1, q2, q3]
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

    return {"expanded_queries": expanded, "documents": _dedupe_documents(docs)}
