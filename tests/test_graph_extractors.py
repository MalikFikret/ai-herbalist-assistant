"""Unit tests for the JSON extractors in advanced_graph.py.

These are pure functions that parse LLM output; they never need a live
LLM and run offline in milliseconds.
"""

from __future__ import annotations

import pytest

from herbalist_assistant.graph.advanced_graph import (
    _dedupe_documents,
    _extract_expanded_queries,
    _extract_route,
    _extract_score,
    _format_chat_history,
    _sanitize_medical_answer,
    _strip_fences,
)


class _Doc:
    """Minimal stand-in for a langchain Document."""

    def __init__(self, page_content: str, metadata: dict | None = None, id_: str | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        if id_ is not None:
            self.id = id_


# ---------- _strip_fences ---------------------------------------------------


def test_strip_fences_plain_json():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


def test_strip_fences_with_markdown_block():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences("```\n{\"a\":1}\n```") == '{"a":1}'


# ---------- _extract_route --------------------------------------------------


def test_extract_route_vector_search():
    assert _extract_route('{"route":"VECTOR_SEARCH"}') == "VECTOR_SEARCH"


def test_extract_route_direct_answer_with_fences():
    assert _extract_route('```json\n{"route":"DIRECT_ANSWER"}\n```') == "DIRECT_ANSWER"


def test_extract_route_falls_back_to_plain_token():
    assert _extract_route("some chatter\nVECTOR_SEARCH\n") == "VECTOR_SEARCH"


def test_extract_route_rejects_unknown():
    with pytest.raises(ValueError):
        _extract_route("totally unparseable")


# ---------- _extract_expanded_queries --------------------------------------


def test_extract_expanded_queries_happy_path():
    raw = '{"expanded_queries":["chamomile tea","matricaria recutita","papatya infusion"]}'
    assert _extract_expanded_queries(raw) == (
        "chamomile tea",
        "matricaria recutita",
        "papatya infusion",
    )


def test_extract_expanded_queries_rejects_wrong_length():
    raw = '{"expanded_queries":["only","two"]}'
    with pytest.raises(ValueError):
        _extract_expanded_queries(raw)


def test_extract_expanded_queries_rejects_duplicates():
    raw = '{"expanded_queries":["a","a","b"]}'
    with pytest.raises(ValueError):
        _extract_expanded_queries(raw)


def test_extract_expanded_queries_rejects_blank():
    raw = '{"expanded_queries":["a","","c"]}'
    with pytest.raises(ValueError):
        _extract_expanded_queries(raw)


# ---------- _extract_score --------------------------------------------------


def test_extract_score_yes():
    assert _extract_score('{"score":"yes"}') == "yes"


def test_extract_score_no_with_fences():
    assert _extract_score('```\n{"score":"no"}\n```') == "no"


def test_extract_score_regex_fallback():
    assert _extract_score('rambling text then "score":"yes" more text') == "yes"


def test_extract_score_token_line_fallback():
    assert _extract_score("sure\nno") == "no"


def test_extract_score_rejects_garbage():
    with pytest.raises(ValueError):
        _extract_score("???")


# ---------- _sanitize_medical_answer ---------------------------------------


def test_sanitize_removes_banned_fragments():
    raw = (
        "Chamomile tea is calming.\n"
        "Please consult your doctor before use.\n"
        "Based on the context, chamomile is safe.\n"
        "Steep 1 tsp for 5 minutes.\n"
    )
    cleaned = _sanitize_medical_answer(raw)
    assert "Chamomile tea is calming." in cleaned
    assert "Steep 1 tsp for 5 minutes." in cleaned
    assert "consult your doctor" not in cleaned.lower()
    assert "based on the context" not in cleaned.lower()


def test_sanitize_collapses_triple_blank_lines():
    raw = "A\n\n\n\nB"
    assert _sanitize_medical_answer(raw) == "A\n\nB"


# ---------- _dedupe_documents ----------------------------------------------


def test_dedupe_documents_by_explicit_id():
    a = _Doc("text-a", id_="doc-1")
    b = _Doc("text-b-diff", id_="doc-1")  # same id -> dedupe
    c = _Doc("text-c", id_="doc-2")
    result = _dedupe_documents([a, b, c])
    assert [d.page_content for d in result] == ["text-a", "text-c"]


def test_dedupe_documents_by_content_hash():
    a = _Doc("same", metadata={"source": "x.pdf", "page": 1})
    b = _Doc("same", metadata={"source": "x.pdf", "page": 1})
    c = _Doc("different", metadata={"source": "x.pdf", "page": 2})
    result = _dedupe_documents([a, b, c])
    assert [d.page_content for d in result] == ["same", "different"]


# ---------- _format_chat_history -------------------------------------------


def test_format_chat_history_empty():
    assert _format_chat_history([]) == ""
    assert _format_chat_history(None) == ""


def test_format_chat_history_filters_roles():
    history = [
        {"role": "user", "content": "Benefits of chamomile?"},
        {"role": "system", "content": "ignore me"},
        {"role": "assistant", "content": "Calming, helps sleep."},
        {"role": "user", "content": "how do I prepare it?"},
    ]
    formatted = _format_chat_history(history)
    assert "User: Benefits of chamomile?" in formatted
    assert "Assistant: Calming, helps sleep." in formatted
    assert "User: how do I prepare it?" in formatted
    assert "ignore me" not in formatted


def test_format_chat_history_truncates_long_messages():
    long = "x" * 2000
    formatted = _format_chat_history([{"role": "assistant", "content": long}])
    assert formatted.endswith("...")
    assert len(formatted) < 900  # label + 800 body chars + ellipsis


def test_format_chat_history_respects_max_messages():
    history = [
        {"role": "user", "content": f"msg-{i}"} for i in range(20)
    ]
    formatted = _format_chat_history(history).splitlines()
    # _MAX_HISTORY_MESSAGES == 6 in the module.
    assert len(formatted) <= 6
    assert formatted[-1].endswith("msg-19")
