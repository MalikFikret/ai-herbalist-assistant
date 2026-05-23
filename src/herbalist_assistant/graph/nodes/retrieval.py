"""Query expansion + vector retrieval node.

Expands the user's question into language-prioritized search queries and runs
them against the Chroma retriever, merging and de-duping the resulting
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

LANGUAGE POLICY (VERY IMPORTANT):
- Detect the language of the CURRENT user question.
- The search system must prioritize that same language first.
- Generate primary queries in the same language as the current question.
- You may add fallback cross-language queries only when they could improve
  recall (for example, English scientific terms), but never replace the
  primary-language queries.

Produce concise, DISTINCT search queries that preserve the same user intent
while varying terminology. Use botanical, scientific, and traditional phrasing
where appropriate. If only 1-2 high-quality rewrites are appropriate, return
fewer queries.

Return only JSON:
{
  "primary_queries": ["same-language query 1", "same-language query 2", "..."],
  "fallback_queries": ["optional cross-language query 1", "..."]
}

Backward-compatible format is also accepted:
{"expanded_queries": ["query1", "query2", "query3"]}"""


def _unique_queries(items: list[object], *, limit: int, seen: set[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        query = str(item).strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) >= limit:
            break
    return cleaned


def _extract_prioritized_queries(raw: str, fallback_question: str) -> list[str]:
    """Parse same-language-first queries from model output, with safe fallback."""
    try:
        data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        return [fallback_question]

    if not isinstance(data, dict):
        return [fallback_question]

    seen: set[str] = set()
    prioritized: list[str] = []

    primary_raw = data.get("primary_queries", [])
    if isinstance(primary_raw, list):
        prioritized.extend(_unique_queries(primary_raw, limit=3, seen=seen))

    fallback_raw = data.get("fallback_queries", [])
    if isinstance(fallback_raw, list) and len(prioritized) < 5:
        prioritized.extend(_unique_queries(fallback_raw, limit=5 - len(prioritized), seen=seen))

    # Backward compatibility with older prompt shape.
    legacy_raw = data.get("expanded_queries", [])
    if isinstance(legacy_raw, list) and not prioritized:
        prioritized.extend(_unique_queries(legacy_raw, limit=3, seen=seen))

    return prioritized or [fallback_question]


def expand_and_retrieve_node(state: AgentState) -> AgentState:
    """Expand into same-language-first search variants and collect candidate docs."""
    question = str(state.get("question", "")).strip()
    if not question:
        return {"expanded_queries": [], "candidate_docs": []}

    history_block = _format_chat_history(state.get("chat_history"))
    user_profile = state.get("user_profile", {})
    if not isinstance(user_profile, dict):
        user_profile = {}
    ui_language = str(state.get("ui_language", "")).strip()

    human_parts: list[str] = []
    if history_block:
        human_parts.append(f"Recent conversation:\n{history_block}")
    if user_profile:
        human_parts.append(
            "User profile (use only if relevant to this question):\n"
            f"{json.dumps(user_profile, ensure_ascii=True)}"
        )
    if ui_language:
        human_parts.append(
            "UI language preference (fallback hint only; prioritize question language):\n"
            f"{ui_language}"
        )
    human_parts.append(f"Current user question:\n{question}")
    human_message = "\n\n".join(human_parts)

    try:
        response = _expansion_llm(_resolve_model_name(state)).invoke(
            [SystemMessage(content=EXPANSION_SYSTEM), HumanMessage(content=human_message)]
        )
        expanded = _extract_prioritized_queries(
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
