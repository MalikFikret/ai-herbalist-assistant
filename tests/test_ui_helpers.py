"""Unit tests for pure-Python helpers in the UI layer.

We import a handful of pure functions from ``streamlit_app`` and
``state``. These functions do not touch the Streamlit runtime, so they
can be tested offline using the Streamlit stub installed by
``tests/conftest.py``.
"""

from __future__ import annotations

import importlib
import sys


def _ui_module():
    from herbalist_assistant.ui import streamlit_app as ui

    return ui


def test_hash_password_deterministic_with_same_salt():
    ui = _ui_module()
    assert ui._hash_password("hunter2", "abc") == ui._hash_password("hunter2", "abc")
    assert ui._hash_password("hunter2", "abc") != ui._hash_password("hunter2", "xyz")


def test_verify_password_round_trip():
    ui = _ui_module()
    salt = "cafebabe"
    hashed = ui._hash_password("correct horse", salt)
    assert ui._verify_password("correct horse", hashed, salt) is True
    assert ui._verify_password("wrong horse", hashed, salt) is False


def test_extract_blocked_herbs_known_alias():
    ui = _ui_module()
    result = ui._extract_blocked_herbs("I am allergic to chamomile and pollen")
    assert "papatya/chamomile" in result


def test_extract_blocked_herbs_empty_input():
    ui = _ui_module()
    assert ui._extract_blocked_herbs("") == []
    assert ui._extract_blocked_herbs(None) == []  # type: ignore[arg-type]


def test_extract_sources_from_docs_dedupes_and_adjusts_page():
    ui = _ui_module()

    class FakeDoc:
        def __init__(self, source, page):
            self.metadata = {"source": source, "page": page}

    docs = [
        FakeDoc("/abs/path/herbs.pdf", 0),
        FakeDoc("/abs/path/herbs.pdf", 0),
        FakeDoc("/abs/path/herbs.pdf", 6),
        FakeDoc("/abs/path/remedies.pdf", None),
    ]
    sources = ui._extract_sources_from_docs(docs)
    assert sources == [
        {"kind": "pdf", "file": "herbs.pdf", "page": 1},
        {"kind": "pdf", "file": "herbs.pdf", "page": 7},
        {"kind": "pdf", "file": "remedies.pdf", "page": None},
    ]


def test_normalize_sources_accepts_legacy_strings():
    ui = _ui_module()
    result = ui._normalize_sources(["herbs.pdf", {"kind": "pdf", "file": "a.pdf", "page": 2}])
    assert result == [
        {"kind": "pdf", "file": "herbs.pdf", "page": None},
        {"kind": "pdf", "file": "a.pdf", "page": 2},
    ]


def test_normalize_sources_adds_default_kind():
    ui = _ui_module()
    result = ui._normalize_sources([{"file": "x.pdf", "page": 3}])
    assert result == [{"kind": "pdf", "file": "x.pdf", "page": 3}]


def test_source_entry_label_pdf_and_url():
    ui = _ui_module()
    assert ui._source_entry_label({"kind": "pdf", "file": "herbs.pdf", "page": 4}) == "herbs.pdf (p. 4)"
    assert ui._source_entry_label({"kind": "pdf", "file": "herbs.pdf"}) == "herbs.pdf"
    assert ui._source_entry_label({"kind": "url", "url": "https://example.com", "title": "Docs"}) == "Docs"
    assert ui._source_entry_label({"kind": "url", "url": "https://example.com"}) == "https://example.com"


def test_verify_admin_password_env_plain(monkeypatch):
    monkeypatch.setenv("HA_ADMIN_PASSWORD", "s3cret")
    # Force-reimport so module-level env reads pick up our override.
    sys.modules.pop("herbalist_assistant.ui.streamlit_app", None)
    ui = importlib.import_module("herbalist_assistant.ui.streamlit_app")

    assert ui._verify_admin_password("s3cret") is True
    assert ui._verify_admin_password("wrong") is False


def test_verify_admin_password_env_hash(monkeypatch):
    import hashlib

    salt = "deadbeef"
    pw = "super-strong"
    hashed = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
    monkeypatch.setenv("HA_ADMIN_PASSWORD_HASH", hashed)
    monkeypatch.setenv("HA_ADMIN_PASSWORD_SALT", salt)
    monkeypatch.delenv("HA_ADMIN_PASSWORD", raising=False)

    sys.modules.pop("herbalist_assistant.ui.streamlit_app", None)
    ui = importlib.import_module("herbalist_assistant.ui.streamlit_app")

    assert ui._verify_admin_password(pw) is True
    assert ui._verify_admin_password("nope") is False
