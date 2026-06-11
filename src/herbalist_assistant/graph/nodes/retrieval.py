"""Query expansion + vector retrieval node.

FIX (v2):
  - Primary query limit increased 3 → 4 (richer expansion pool).
  - candidate_docs cap increased 10 → 12 (grader still caps at 6,
    but the input pool is now wider after increasing RETRIEVER_K to 5).
  - Added explicit logging of expanded queries for easier debugging.
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

# ── Tuning knobs ──────────────────────────────────────────────────────────────
# FIX: was 3 primary + 2 fallback; now 4 primary + 3 fallback = richer pool
_PRIMARY_QUERY_LIMIT  = 4
_FALLBACK_QUERY_LIMIT = 3
# FIX: cap before passing to grader (grader itself caps at _MAX_GRADE_DOCS=6)
_CANDIDATE_DOC_CAP = 12   # was 10
# ─────────────────────────────────────────────────────────────────────────────


EXPANSION_SYSTEM = """You generate retrieval rewrites for a herbal health assistant.

If the user's current question is a follow-up that contains pronouns or vague
references (it / this / that / the tea / the herb / onu / bunu / şunu / o çayı),
USE the recent conversation turns to resolve the referent before rewriting.
For example, if the user asked about chamomile earlier and now asks
"how do I prepare it?", rewrite the queries around chamomile preparation,
not a generic preparation query.

When a user profile is provided, incorporate medically relevant profile details
(e.g., age, allergies, conditions) into the search intent only when relevant.

LANGUAGE POLICY (VERY IMPORTANT):
- Detect the language of the CURRENT user question.
- Generate primary queries in the same language as the current question.
- You may add fallback cross-language queries (e.g. English scientific names
  for Turkish questions) only when they could improve recall — never replace
  the primary-language queries with them.

Produce concise, DISTINCT search queries that preserve the same user intent
while varying terminology. Use botanical, scientific, and traditional phrasing
where appropriate.

Return only JSON:
{
  "primary_queries":  ["same-language query 1", "same-language query 2", "same-language query 3", "same-language query 4"],
  "fallback_queries": ["optional cross-language query 1", "optional cross-language query 2"]
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
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
    except Exception:
        _logger.warning("Query expansion: JSON parse failed; using original question")
        return [fallback_question]

    if not isinstance(data, dict):
        return [fallback_question]

    seen: set[str] = set()
    prioritized: list[str] = []

    primary_raw = data.get("primary_queries", [])
    if isinstance(primary_raw, list):
        prioritized.extend(
            _unique_queries(primary_raw, limit=_PRIMARY_QUERY_LIMIT, seen=seen)
        )

    fallback_raw = data.get("fallback_queries", [])
    remaining = _PRIMARY_QUERY_LIMIT + _FALLBACK_QUERY_LIMIT - len(prioritized)
    if isinstance(fallback_raw, list) and remaining > 0:
        prioritized.extend(
            _unique_queries(fallback_raw, limit=remaining, seen=seen)
        )

    # Backward compatibility with older prompt shape
    legacy_raw = data.get("expanded_queries", [])
    if isinstance(legacy_raw, list) and not prioritized:
        prioritized.extend(
            _unique_queries(legacy_raw, limit=_PRIMARY_QUERY_LIMIT, seen=seen)
        )

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

    _logger.info(
        "expand_and_retrieve_node: %d expanded queries: %s",
        len(expanded),
        expanded,
    )

    docs: list[Document] = []
    retriever = _retriever()
    for query in expanded:
        try:
            batch = retriever.invoke(query)
            _logger.debug("Query %r → %d docs", query, len(batch) if batch else 0)
        except Exception:
            _logger.exception("Retriever.invoke failed for query=%r", query)
            continue
        if batch:
            docs.extend(batch)

    candidate_docs = _dedupe_documents(docs)[:_CANDIDATE_DOC_CAP]
    _logger.info(
        "expand_and_retrieve_node: %d unique candidate docs (cap=%d)",
        len(candidate_docs),
        _CANDIDATE_DOC_CAP,
    )
    return {"expanded_queries": expanded, "candidate_docs": candidate_docs}