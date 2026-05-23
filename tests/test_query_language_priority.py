"""Tests for same-language-first query parsing in retrieval/web-search nodes."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_NODES_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "herbalist_assistant"
    / "graph"
    / "nodes"
)


def _load_node_module(filename: str, module_name: str):
    path = _NODES_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_web_search_dependency_stubs() -> None:
    """Provide minimal stubs so web_search.py can be imported in unit tests."""
    if "langchain_community.tools" not in sys.modules:
        tools_mod = types.ModuleType("langchain_community.tools")

        class _DuckDuckGoSearchRun:
            def invoke(self, _query):
                return ""

        tools_mod.DuckDuckGoSearchRun = _DuckDuckGoSearchRun
        sys.modules["langchain_community.tools"] = tools_mod


def test_extract_prioritized_queries_prefers_primary_then_fallback():
    retrieval = _load_node_module("retrieval.py", "test_retrieval_module")
    raw = """
    {
      "primary_queries": ["rezene çayı mide bulantısı", "mide bulantısına rezene"],
      "fallback_queries": ["fennel tea nausea", "fennel infusion nausea relief"]
    }
    """

    out = retrieval._extract_prioritized_queries(raw, "rezene faydalari")
    assert out == [
        "rezene çayı mide bulantısı",
        "mide bulantısına rezene",
        "fennel tea nausea",
        "fennel infusion nausea relief",
    ]


def test_extract_prioritized_queries_supports_legacy_shape():
    retrieval = _load_node_module("retrieval.py", "test_retrieval_module_legacy")
    raw = '{"expanded_queries": ["melisa çayı uyku", "lemon balm sleep"]}'
    out = retrieval._extract_prioritized_queries(raw, "melisa")
    assert out == ["melisa çayı uyku", "lemon balm sleep"]


def test_extract_retry_queries_prefers_primary_then_fallback():
    _install_web_search_dependency_stubs()
    web_search = _load_node_module("web_search.py", "test_web_search_module")
    raw = """
    {
      "primary_queries": ["ıhlamur çayı öksürük", "öksürük için ıhlamur"],
      "fallback_queries": ["linden tea cough"]
    }
    """
    out = web_search._extract_retry_queries(raw, "ıhlamur")
    assert out == ["ıhlamur çayı öksürük", "öksürük için ıhlamur", "linden tea cough"]

