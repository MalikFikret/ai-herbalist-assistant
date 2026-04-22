"""Session-state glue between Streamlit and the SQLite-backed repository.

The public function names and signatures here match the pre-DB version
so ``streamlit_app.py`` did not need to change when we migrated. All
reads/writes go through :mod:`herbalist_assistant.db.repository`; the
only thing we keep in ``st.session_state`` is a snapshot of the active
chat for fast rendering.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from herbalist_assistant.db import repository

# ---------------------------------------------------------------------------
# Session-state snapshot helpers
# ---------------------------------------------------------------------------


def _user_key(username: str) -> str:
    return (username or "").strip() or "guest"


def _refresh_snapshot(user_key: str) -> None:
    """Pull the user's active chat + chat list from the DB into session_state.

    This is how the sidebar chat list and the main chat pane stay in
    sync with the database after any mutation.
    """
    active = repository.load_active_chat(user_key)
    summaries = repository.get_user_chat_summaries(user_key)

    st.session_state.messages = list(active["messages"])
    st.session_state.messages_owner = user_key
    st.session_state.active_chat_id = active["active_chat_id"]
    st.session_state.user_chats = summaries


# ---------------------------------------------------------------------------
# Public API (called by streamlit_app.py)
# ---------------------------------------------------------------------------


def init_session_state(username: str) -> None:
    """Ensure session_state reflects the user's chats in the DB."""
    user_key = _user_key(username)
    _refresh_snapshot(user_key)


def append_message(
    *,
    role: str,
    content: str,
    username: str,
    sources: list[dict[str, Any]] | list[str] | None = None,
) -> int:
    """Append a message to the active chat. Returns its position (index)."""
    user_key = _user_key(username)
    if "messages" not in st.session_state or st.session_state.get("messages_owner") != user_key:
        _refresh_snapshot(user_key)

    active_chat_id = st.session_state.get("active_chat_id")
    result = repository.append_message(
        username=user_key,
        chat_id=active_chat_id,
        role=role,
        content=content,
        sources=list(sources) if sources else [],
    )
    _refresh_snapshot(user_key)
    return int(result["position"])


def update_message_feedback(
    *,
    username: str,
    chat_id: str,
    message_index: int,
    feedback: str | None,
) -> bool:
    user_key = _user_key(username)
    ok = repository.update_message_feedback(
        username=user_key,
        chat_id=chat_id,
        message_index=message_index,
        feedback=feedback,
    )
    if not ok:
        return False
    if (
        st.session_state.get("active_chat_id") == chat_id
        and st.session_state.get("messages_owner") == user_key
    ):
        _refresh_snapshot(user_key)
    return True


def iter_all_feedback() -> list[dict[str, Any]]:
    return repository.iter_all_feedback()


def start_new_chat(username: str) -> str:
    user_key = _user_key(username)
    chat_id = repository.start_new_chat(user_key)
    _refresh_snapshot(user_key)
    return chat_id


def get_user_chat_summaries(username: str) -> list[dict[str, Any]]:
    return repository.get_user_chat_summaries(_user_key(username))


def get_chat_messages(username: str, chat_id: str) -> list[dict[str, Any]]:
    return repository.get_chat_messages(_user_key(username), chat_id)


def set_active_chat(username: str, chat_id: str) -> bool:
    user_key = _user_key(username)
    ok = repository.set_active_chat(user_key, chat_id)
    if ok:
        _refresh_snapshot(user_key)
    return ok


def delete_chat(username: str, chat_id: str) -> bool:
    user_key = _user_key(username)
    ok = repository.delete_chat(user_key, chat_id)
    if ok:
        _refresh_snapshot(user_key)
    return ok
