"""High-level data operations used by the UI layer.

The repository is the only module outside ``db`` that the Streamlit UI
should import from -- it keeps ORM objects behind plain dicts so session
state never accidentally holds stale or detached rows.

All functions are transaction-scoped: every entry point opens its own
``session_scope`` so concurrent Streamlit reruns don't step on each
other's uncommitted state.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .engine import session_scope
from .models import AppSetting, ChatMessage, ChatSession, User

_logger = logging.getLogger("herbalist_assistant.db")


# ---------------------------------------------------------------------------
# Small serialization helpers
# ---------------------------------------------------------------------------


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_DEFAULT_PROFILE: dict[str, str] = {
    "name": "",
    "age": "",
    "gender": "",
    "allergies": "",
    "conditions": "",
}


def _load_profile(raw: str | None) -> dict[str, str]:
    if not raw:
        return dict(_DEFAULT_PROFILE)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return dict(_DEFAULT_PROFILE)
    if not isinstance(data, dict):
        return dict(_DEFAULT_PROFILE)
    merged = dict(_DEFAULT_PROFILE)
    for key in _DEFAULT_PROFILE:
        value = data.get(key)
        merged[key] = str(value).strip() if value is not None else ""
    return merged


def _dump_profile(profile: dict[str, str]) -> str:
    return json.dumps(
        {
            "name": (profile.get("name") or "").strip(),
            "age": (profile.get("age") or "").strip(),
            "gender": (profile.get("gender") or "").strip(),
            "allergies": (profile.get("allergies") or "").strip(),
            "conditions": (profile.get("conditions") or "").strip(),
        },
        ensure_ascii=False,
    )


def _load_sources(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _dump_sources(sources: list[Any] | None) -> str:
    if not sources:
        return "[]"
    # Keep legacy string sources intact; they're normalized in the UI.
    return json.dumps(sources, ensure_ascii=False)


def _message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp,
        "sources": _load_sources(msg.sources_json),
    }
    if msg.role == "assistant":
        out["feedback"] = msg.feedback
        out["feedback_at"] = msg.feedback_at
    return out


def _chat_summary(session: ChatSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title or "New Chat",
        "created_at": session.created_at or "-",
        "updated_at": session.updated_at or "-",
        "message_count": len(session.messages),
    }


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


def _user_key(username: str) -> str:
    return (username or "").strip() or "guest"


def _find_user(session: Session, username: str) -> User | None:
    stmt = select(User).where(User.username == _user_key(username))
    return session.execute(stmt).scalar_one_or_none()


def _ensure_user(session: Session, username: str) -> User:
    """Fetch a user row, creating a placeholder guest row if missing.

    The placeholder is only used for anonymous chats (``username`` empty
    or not yet registered); real registration still goes through
    :func:`create_user`.
    """
    user = _find_user(session, username)
    if user is not None:
        return user

    key = _user_key(username)
    user = User(
        username=key,
        password_hash="",
        salt="",
        role="guest" if key == "guest" else "user",
        health_profile_json=_dump_profile({}),
    )
    session.add(user)
    session.flush()
    return user


def _ensure_active_session(session: Session, user: User) -> ChatSession:
    """Return the user's active chat session, creating one if needed."""
    if user.active_session_id:
        current = session.get(ChatSession, user.active_session_id)
        if current is not None and current.user_id == user.id:
            return current

    if user.sessions:
        fallback = user.sessions[-1]
        user.active_session_id = fallback.id
        return fallback

    fresh = ChatSession(
        id=uuid.uuid4().hex,
        user_id=user.id,
        title="New Chat",
        created_at=_now_str(),
        updated_at=_now_str(),
    )
    session.add(fresh)
    session.flush()
    user.active_session_id = fresh.id
    return fresh


# ---------------------------------------------------------------------------
# Public: user auth + profile
# ---------------------------------------------------------------------------


def user_exists(username: str) -> bool:
    with session_scope() as session:
        return _find_user(session, username) is not None


def get_user_auth(username: str) -> dict[str, str] | None:
    """Return {password_hash, salt, role} for the given username, or None."""
    with session_scope() as session:
        user = _find_user(session, username)
        if user is None:
            return None
        return {
            "password_hash": user.password_hash,
            "salt": user.salt,
            "role": user.role,
        }


def create_user(
    *,
    username: str,
    password_hash: str,
    salt: str,
    role: str = "user",
) -> bool:
    """Insert a new user. Returns False if the username is already taken."""
    key = _user_key(username)
    with session_scope() as session:
        if _find_user(session, key) is not None:
            return False
        session.add(
            User(
                username=key,
                password_hash=password_hash,
                salt=salt,
                role=role,
                health_profile_json=_dump_profile({}),
            )
        )
    return True


def reset_user_password(*, username: str, password_hash: str, salt: str) -> bool:
    with session_scope() as session:
        user = _find_user(session, username)
        if user is None:
            return False
        user.password_hash = password_hash
        user.salt = salt
    return True


def get_user_profile(username: str) -> dict[str, str]:
    with session_scope() as session:
        user = _find_user(session, username)
        if user is None:
            return dict(_DEFAULT_PROFILE)
        return _load_profile(user.health_profile_json)


def save_user_profile(username: str, profile: dict[str, str]) -> bool:
    with session_scope() as session:
        user = _find_user(session, username)
        if user is None:
            return False
        user.health_profile_json = _dump_profile(profile)
    return True


def get_all_users() -> list[dict[str, Any]]:
    with session_scope() as session:
        stmt = select(User).order_by(User.created_at.desc())
        real_users = []
        for u in session.execute(stmt).scalars():
            un = (u.username or "").lower()
            if un == "guest" or un.startswith(("anon_", "guest_", "temp_", "test_")):
                continue
            real_users.append(
                {
                    "id": u.id,
                    "username": u.username,
                    "role": u.role,
                    "created_at": u.created_at,
                }
            )
        return real_users


def delete_user(username: str) -> bool:
    key = _user_key(username)
    with session_scope() as session:
        user = _find_user(session, key)
        if user is None:
            return False
        session.delete(user)
    return True


# ---------------------------------------------------------------------------
# Public: chat sessions + messages
# ---------------------------------------------------------------------------


def load_active_chat(username: str) -> dict[str, Any]:
    """Return the active chat (summary + messages) for ``username``.

    Creates the user and a first chat if they don't exist yet; this is
    the entry point called by ``init_session_state``.
    """
    with session_scope() as session:
        user = _ensure_user(session, username)
        active = _ensure_active_session(session, user)
        return {
            "active_chat_id": active.id,
            "messages": [_message_to_dict(m) for m in active.messages],
        }


def get_user_chat_summaries(username: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        user = _ensure_user(session, username)
        _ensure_active_session(session, user)
        return [_chat_summary(s) for s in user.sessions]


def get_chat_messages(username: str, chat_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        user = _ensure_user(session, username)
        target = session.get(ChatSession, chat_id)
        if target is None or target.user_id != user.id:
            return []
        return [_message_to_dict(m) for m in target.messages]


def start_new_chat(username: str) -> str:
    with session_scope() as session:
        user = _ensure_user(session, username)
        fresh = ChatSession(
            id=uuid.uuid4().hex,
            user_id=user.id,
            title="New Chat",
            created_at=_now_str(),
            updated_at=_now_str(),
        )
        session.add(fresh)
        session.flush()
        user.active_session_id = fresh.id
        return fresh.id


def set_active_chat(username: str, chat_id: str) -> bool:
    with session_scope() as session:
        user = _ensure_user(session, username)
        target = session.get(ChatSession, chat_id)
        if target is None or target.user_id != user.id:
            return False
        user.active_session_id = target.id
    return True


def delete_chat(username: str, chat_id: str) -> bool:
    with session_scope() as session:
        user = _ensure_user(session, username)
        target = session.get(ChatSession, chat_id)
        if target is None or target.user_id != user.id:
            return False
        session.delete(target)
        session.flush()
        # Never leave a user without a chat: pick (or create) the next one.
        active = _ensure_active_session(session, user)
        user.active_session_id = active.id
    return True


def append_message(
    *,
    username: str,
    chat_id: str | None,
    role: str,
    content: str,
    sources: list[Any] | None = None,
) -> dict[str, Any]:
    """Append a message to ``chat_id`` (or the active chat) for ``username``.

    Returns a dict with the new message's ``position`` and parent
    ``chat_id`` so the UI can locate it later (for feedback updates).
    """
    now = _now_str()
    with session_scope() as session:
        user = _ensure_user(session, username)
        target: ChatSession | None
        if chat_id:
            target = session.get(ChatSession, chat_id)
            if target is None or target.user_id != user.id:
                target = _ensure_active_session(session, user)
        else:
            target = _ensure_active_session(session, user)

        assert target is not None  # _ensure_active_session always returns one
        position = len(target.messages)
        message = ChatMessage(
            session_id=target.id,
            position=position,
            role=role,
            content=content,
            timestamp=now,
            sources_json=_dump_sources(sources),
        )
        session.add(message)

        # Promote the first real user utterance to the chat title.
        if role == "user" and (target.title or "New Chat") == "New Chat":
            preview = (content or "").strip().splitlines()[0][:50]
            if preview:
                target.title = preview

        target.updated_at = now
        user.active_session_id = target.id
        session.flush()

        return {
            "chat_id": target.id,
            "position": position,
            "message": _message_to_dict(message),
        }


def update_message_feedback(
    *,
    username: str,
    chat_id: str,
    message_index: int,
    feedback: str | None,
) -> bool:
    if feedback not in (None, "up", "down"):
        return False

    with session_scope() as session:
        user = _ensure_user(session, username)
        target = session.get(ChatSession, chat_id)
        if target is None or target.user_id != user.id:
            return False
        if message_index < 0 or message_index >= len(target.messages):
            return False

        message = target.messages[message_index]
        if message.role != "assistant":
            return False

        message.feedback = feedback
        message.feedback_at = _now_str() if feedback else None
        target.updated_at = _now_str()
    return True


def iter_all_feedback() -> list[dict[str, Any]]:
    """Flat, newest-first list of every message that has feedback attached.

    This is the backing query for the Admin panel's feedback viewer.
    """
    entries: list[dict[str, Any]] = []
    with session_scope() as session:
        stmt = (
            select(ChatMessage, ChatSession, User)
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .join(User, ChatSession.user_id == User.id)
            .where(ChatMessage.feedback.in_(("up", "down")))
        )
        for message, chat, user in session.execute(stmt).all():
            question = ""
            # Grab the user utterance immediately preceding this answer (if any).
            if message.position > 0:
                prev = next(
                    (m for m in chat.messages if m.position == message.position - 1),
                    None,
                )
                if prev is not None and prev.role == "user":
                    question = prev.content or ""
            entries.append(
                {
                    "username": user.username,
                    "chat_id": chat.id,
                    "chat_title": chat.title or "New Chat",
                    "message_index": message.position,
                    "feedback": message.feedback,
                    "feedback_at": message.feedback_at or message.timestamp or "",
                    "question": question,
                    "answer": message.content or "",
                    "timestamp": message.timestamp or "",
                    "sources": _load_sources(message.sources_json),
                }
            )
    entries.sort(key=lambda e: e.get("feedback_at") or "", reverse=True)
    return entries


# ---------------------------------------------------------------------------
# Counters (used by docs / admin diagnostics)
# ---------------------------------------------------------------------------


def stats() -> dict[str, int]:
    from sqlalchemy import func

    with session_scope() as session:
        users = session.execute(select(func.count()).select_from(User)).scalar_one()
        chats = session.execute(select(func.count()).select_from(ChatSession)).scalar_one()
        messages = session.execute(select(func.count()).select_from(ChatMessage)).scalar_one()
        return {"users": int(users), "chats": int(chats), "messages": int(messages)}


def get_app_setting(key: str) -> str | None:
    key = (key or "").strip()
    if not key:
        return None
    with session_scope() as session:
        row = session.get(AppSetting, key)
        return row.value if row is not None else None


def set_app_setting(*, key: str, value: str) -> bool:
    key = (key or "").strip()
    if not key:
        return False
    with session_scope() as session:
        row = session.get(AppSetting, key)
        if row is None:
            row = AppSetting(key=key, value=value)
            session.add(row)
        else:
            row.value = value
    return True
