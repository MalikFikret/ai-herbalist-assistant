"""Tests for the Streamlit session-state helpers in ``ui/state.py``.

The tests run against a fresh on-disk SQLite database inside a temp
directory (one per test), so no global state leaks across tests and
the real repository code -- not mocks -- gets exercised end-to-end.
"""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Point the DB at a per-test SQLite file and bounce the engine cache."""
    db_file = tmp_path / "herbalist.db"
    monkeypatch.setenv("HA_DB_PATH", str(db_file))

    for mod in [
        "herbalist_assistant.ui.state",
        "herbalist_assistant.db",
        "herbalist_assistant.db.engine",
        "herbalist_assistant.db.models",
        "herbalist_assistant.db.migration",
        "herbalist_assistant.db.repository",
    ]:
        sys.modules.pop(mod, None)

    from herbalist_assistant.db import ensure_database_ready
    from herbalist_assistant.db.engine import reset_engine_cache

    reset_engine_cache()
    ensure_database_ready()
    yield db_file
    reset_engine_cache()


@pytest.fixture()
def clean_streamlit_session():
    """Start each test with an empty Streamlit session dict."""
    import streamlit as st

    st.session_state.clear()
    yield
    st.session_state.clear()


def _import_state():
    from herbalist_assistant.ui import state

    return state


def test_init_session_state_creates_user_and_first_chat(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("alice")

    import streamlit as st

    assert st.session_state.messages_owner == "alice"
    assert st.session_state.messages == []
    assert isinstance(st.session_state.active_chat_id, str)
    assert len(st.session_state.user_chats) == 1
    assert st.session_state.user_chats[0]["title"] == "New Chat"


def test_append_message_flow_and_title_promotion(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("alice")

    idx = state.append_message(
        role="user",
        content="What about chamomile for sleep?",
        username="alice",
    )
    assert idx == 0

    import streamlit as st

    assert st.session_state.messages[0]["role"] == "user"
    assert st.session_state.messages[0]["content"].startswith("What about chamomile")
    # First user utterance should promote to the chat title.
    assert st.session_state.user_chats[0]["title"].startswith("What about chamomile")

    idx2 = state.append_message(
        role="assistant",
        content="Chamomile tea is calming and may help sleep.",
        username="alice",
        sources=[{"kind": "pdf", "file": "herbs.pdf", "page": 12}],
    )
    assert idx2 == 1
    assert st.session_state.messages[1]["role"] == "assistant"
    assert st.session_state.messages[1]["feedback"] is None
    assert st.session_state.messages[1]["sources"] == [
        {"kind": "pdf", "file": "herbs.pdf", "page": 12}
    ]


def test_update_message_feedback_roundtrip(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("bob")
    state.append_message(role="user", content="q1", username="bob")
    assistant_idx = state.append_message(role="assistant", content="a1", username="bob")

    import streamlit as st

    chat_id = st.session_state.active_chat_id

    assert state.update_message_feedback(
        username="bob", chat_id=chat_id, message_index=assistant_idx, feedback="up"
    )
    assert st.session_state.messages[assistant_idx]["feedback"] == "up"

    # Toggling to None clears timestamp, stays valid.
    assert state.update_message_feedback(
        username="bob", chat_id=chat_id, message_index=assistant_idx, feedback=None
    )
    assert st.session_state.messages[assistant_idx]["feedback"] is None


def test_update_message_feedback_rejects_user_message(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("bob")
    state.append_message(role="user", content="q1", username="bob")

    import streamlit as st

    chat_id = st.session_state.active_chat_id
    assert not state.update_message_feedback(
        username="bob", chat_id=chat_id, message_index=0, feedback="up"
    )


def test_iter_all_feedback_ordered_newest_first(fresh_db, clean_streamlit_session):
    state = _import_state()

    state.init_session_state("alice")
    state.append_message(role="user", content="q-a", username="alice")
    ai_a = state.append_message(role="assistant", content="a-a", username="alice")

    import streamlit as st

    chat_a = st.session_state.active_chat_id
    state.update_message_feedback(
        username="alice", chat_id=chat_a, message_index=ai_a, feedback="up"
    )

    state.init_session_state("bob")
    state.append_message(role="user", content="q-b", username="bob")
    ai_b = state.append_message(role="assistant", content="a-b", username="bob")
    chat_b = st.session_state.active_chat_id
    state.update_message_feedback(
        username="bob", chat_id=chat_b, message_index=ai_b, feedback="down"
    )

    entries = state.iter_all_feedback()
    assert len(entries) == 2
    users = {e["username"] for e in entries}
    assert users == {"alice", "bob"}
    assert {e["feedback"] for e in entries} == {"up", "down"}
    # Ordered by feedback_at DESC: entries from the same second may tie,
    # so we just assert the order is reverse-chronological by that key.
    feedback_times = [e["feedback_at"] for e in entries]
    assert feedback_times == sorted(feedback_times, reverse=True)


def test_start_and_delete_chat(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("dana")
    state.append_message(role="user", content="orig", username="dana")

    import streamlit as st

    first_chat = st.session_state.active_chat_id
    new_chat = state.start_new_chat("dana")
    assert new_chat != first_chat
    assert st.session_state.active_chat_id == new_chat
    assert st.session_state.messages == []

    # Deleting the active chat leaves the user with at least one chat.
    assert state.delete_chat("dana", new_chat)
    assert st.session_state.active_chat_id == first_chat
    assert st.session_state.messages[0]["content"] == "orig"


def test_set_active_chat_switches_snapshot(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("erin")

    import streamlit as st

    state.append_message(role="user", content="msg1", username="erin")
    chat_one = st.session_state.active_chat_id

    chat_two = state.start_new_chat("erin")
    state.append_message(role="user", content="second-chat-msg", username="erin")
    assert st.session_state.active_chat_id == chat_two

    assert state.set_active_chat("erin", chat_one)
    assert st.session_state.active_chat_id == chat_one
    assert st.session_state.messages[0]["content"] == "msg1"


def test_get_chat_messages_honors_ownership(fresh_db, clean_streamlit_session):
    state = _import_state()
    state.init_session_state("owner")
    state.append_message(role="user", content="private", username="owner")

    import streamlit as st

    chat_id = st.session_state.active_chat_id
    # Different user should get nothing for a chat they don't own.
    assert state.get_chat_messages("intruder", chat_id) == []
    assert state.get_chat_messages("owner", chat_id)[0]["content"] == "private"


def test_ha_db_path_env_isolates_tests(fresh_db):
    # Belt-and-suspenders: the fixture must have set HA_DB_PATH.
    assert os.environ.get("HA_DB_PATH")
