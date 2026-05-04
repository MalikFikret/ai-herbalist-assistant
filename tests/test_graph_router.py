"""Unit tests for deterministic router safeguards."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROUTER_PATH = Path(__file__).resolve().parents[1] / "src" / "herbalist_assistant" / "graph" / "nodes" / "router.py"
_SPEC = importlib.util.spec_from_file_location("test_router_module", _ROUTER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load router module for tests.")
router = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(router)


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _LLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return _Resp(self._content)


def test_route_question_forces_direct_for_identity_greeting(monkeypatch):
    def _raise_if_called(_model_name: str):
        raise AssertionError("Router LLM should not be called for explicit identity greeting.")

    monkeypatch.setattr(router, "_router_llm", _raise_if_called)
    out = router.route_question({"question": "hi, who made you?"})
    assert out["is_medical"] is False


def test_route_question_uses_llm_for_medical_query(monkeypatch):
    monkeypatch.setattr(router, "_resolve_model_name", lambda _state: "dummy")
    monkeypatch.setattr(router, "_router_llm", lambda _model_name: _LLM('{"route":"VECTOR_SEARCH"}'))
    out = router.route_question({"question": "best herb tea for headache"})
    assert out["is_medical"] is True


def test_route_question_lexical_fallback_defaults_direct_without_medical_signal(monkeypatch):
    monkeypatch.setattr(router, "_resolve_model_name", lambda _state: "dummy")

    class _BrokenLLM:
        def invoke(self, _messages):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(router, "_router_llm", lambda _model_name: _BrokenLLM())
    out = router.route_question({"question": "who made you?"})
    assert out["is_medical"] is False

