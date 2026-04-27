"""Pure helpers that parse LLM output and normalise text/documents.

These functions are intentionally side-effect free: no LLM calls, no I/O, no
LangGraph state mutation. That keeps them trivially unit-testable
(see ``tests/test_graph_extractors.py``) and reusable from any node.
"""

from __future__ import annotations

import hashlib
import json
import re

from langchain_core.documents import Document

from herbalist_assistant.graph.state import RouteLiteral, ScoreLiteral

# How many past messages to include when we format chat history for the LLMs.
# 6 = roughly the last 3 user/assistant turns; enough for follow-ups without
# blowing up the prompt.
_MAX_HISTORY_MESSAGES = 6


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_route(raw: str) -> RouteLiteral:
    try:
        data = json.loads(_strip_fences(raw))
        route = str(data.get("route", "")).strip().upper()
        if route in ("VECTOR_SEARCH", "DIRECT_ANSWER"):
            return route  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    for line in raw.splitlines():
        token = line.strip().upper()
        if token in ("VECTOR_SEARCH", "DIRECT_ANSWER"):
            return token  # type: ignore[return-value]

    raise ValueError(f"Unparseable router output: {raw!r}")


def _extract_expanded_queries(raw: str) -> tuple[str, str, str]:
    data = json.loads(_strip_fences(raw))
    queries = data.get("expanded_queries")
    if not isinstance(queries, list) or len(queries) != 3:
        raise ValueError("expanded_queries must be a list of exactly 3 strings")

    cleaned = tuple(str(q).strip() for q in queries)
    if any(not q for q in cleaned):
        raise ValueError("expanded queries must be non-empty")
    if len({q.lower() for q in cleaned}) != 3:
        raise ValueError("expanded queries must be distinct")
    return cleaned  # type: ignore[return-value]


def _extract_score(raw: str) -> ScoreLiteral:
    try:
        data = json.loads(_strip_fences(raw))
        score = str(data.get("score", "")).strip().lower()
        if score in ("yes", "no"):
            return score  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    m = re.search(r'"score"\s*:\s*"(yes|no)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower()  # type: ignore[return-value]

    for line in raw.splitlines():
        token = line.strip().lower()
        if token in ("yes", "no"):
            return token  # type: ignore[return-value]

    raise ValueError(f"Unparseable grader output: {raw!r}")


def _document_key(doc: Document) -> str:
    explicit_id = getattr(doc, "id", None)
    if explicit_id:
        return f"id:{explicit_id}"
    meta_id = doc.metadata.get("id") if doc.metadata else None
    if meta_id is not None:
        return f"meta_id:{meta_id}"
    source = doc.metadata.get("source", "") if doc.metadata else ""
    page = doc.metadata.get("page", "")
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"src:{source!s}|page:{page!s}|sha256:{digest}"


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        key = _document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _format_chat_history(history: list[dict] | None) -> str:
    """Format recent turns for inclusion in LLM prompts.

    Returns an empty string when there is nothing to show. Truncates each
    message to keep prompt sizes bounded.
    """
    if not history:
        return ""

    turns = history[-_MAX_HISTORY_MESSAGES:]
    lines: list[str] = []
    for msg in turns:
        role = (msg.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        label = "User" if role == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Prevent a single huge assistant message from dominating the prompt.
        if len(content) > 800:
            content = content[:800] + "..."
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


def _sanitize_medical_answer(answer: str) -> str:
    """Remove meta/disclaimer lines that degrade user-facing answer quality."""
    banned_fragments = [
        "consult a healthcare professional",
        "consult your doctor",
        "talk to your doctor",
        "seek medical advice",
        "mentioned in the context",
        "according to the context",
        "based on the context",
        "provided context",
        "provided text",
        "doktora danış",
        "bir doktora danış",
    ]

    kept_lines: list[str] = []
    for line in (answer or "").splitlines():
        lowered = line.lower()
        if any(fragment in lowered for fragment in banned_fragments):
            continue
        kept_lines.append(line.rstrip())

    cleaned = "\n".join(kept_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)
