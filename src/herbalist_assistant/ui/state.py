import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

CHAT_HISTORY_FILE = Path(".chat_history.json")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_messages() -> List[Dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your 🌿 AI Herbalist Assistant. "
                "Ask me about herbs, traditional remedies, and general wellness support. "
                "I base my answers on the herbal PDFs you provide in the local `data/` folder.\n\n"
                "_Note: This is for educational purposes only and not medical advice._"
            ),
            "timestamp": _now_str(),
            "sources": [],
        }
    ]


def _new_chat(*, title: str = "New Chat") -> Dict[str, Any]:
    timestamp = _now_str()
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": _default_messages(),
    }


def _load_history_store() -> Dict[str, Any]:
    if not CHAT_HISTORY_FILE.exists():
        return {}
    try:
        with CHAT_HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_history_store(store: Dict[str, Any]) -> None:
    with CHAT_HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=True, indent=2)


def _normalize_user_history(user_history: Any) -> Dict[str, Any]:
    # Backward compatibility: old format was just a list of messages.
    if isinstance(user_history, list):
        migrated_chat = _new_chat(title="Previous Chat")
        migrated_chat["messages"] = user_history if user_history else _default_messages()
        return {
            "active_chat_id": migrated_chat["id"],
            "chats": [migrated_chat],
        }

    if not isinstance(user_history, dict):
        chat = _new_chat()
        return {"active_chat_id": chat["id"], "chats": [chat]}

    chats = user_history.get("chats")
    if not isinstance(chats, list) or not chats:
        chat = _new_chat()
        user_history["chats"] = [chat]
        user_history["active_chat_id"] = chat["id"]
        return user_history

    for chat in chats:
        chat.setdefault("id", uuid.uuid4().hex)
        chat.setdefault("title", "New Chat")
        chat.setdefault("created_at", _now_str())
        chat.setdefault("updated_at", chat.get("created_at", _now_str()))
        chat.setdefault("messages", _default_messages())

    if user_history.get("active_chat_id") not in {c["id"] for c in chats}:
        user_history["active_chat_id"] = chats[-1]["id"]

    return user_history


def _get_or_create_user_history(store: Dict[str, Any], username: str) -> Dict[str, Any]:
    user_key = username.strip() or "guest"
    normalized = _normalize_user_history(store.get(user_key))
    store[user_key] = normalized
    return normalized


def _set_active_chat_session(username: str, user_history: Dict[str, Any]) -> None:
    active_chat_id = user_history.get("active_chat_id")
    active_chat = next(
        (chat for chat in user_history.get("chats", []) if chat.get("id") == active_chat_id),
        None,
    )
    if active_chat is None:
        active_chat = user_history["chats"][-1]
        user_history["active_chat_id"] = active_chat["id"]

    st.session_state.messages = active_chat.get("messages", _default_messages())
    st.session_state.messages_owner = username
    st.session_state.active_chat_id = active_chat["id"]
    st.session_state.user_chats = [
        {
            "id": chat["id"],
            "title": chat.get("title", "New Chat"),
            "created_at": chat.get("created_at", "-"),
            "updated_at": chat.get("updated_at", "-"),
            "message_count": len(chat.get("messages", [])),
        }
        for chat in user_history.get("chats", [])
    ]


def init_session_state(username: str) -> None:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)
    _save_history_store(history_store)

    if st.session_state.get("messages_owner") != user_key or "messages" not in st.session_state:
        _set_active_chat_session(user_key, user_history)
    else:
        # Refresh chat list metadata for sidebar/history views.
        _set_active_chat_session(user_key, user_history)


def append_message(*, role: str, content: str, username: str, sources: List[str] | None = None) -> None:
    if "messages" not in st.session_state:
        init_session_state(username)

    message = {
        "role": role,
        "content": content,
        "timestamp": _now_str(),
        "sources": sources or [],
    }
    st.session_state.messages.append(message)

    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)
    active_chat_id = st.session_state.get("active_chat_id", user_history["active_chat_id"])

    for chat in user_history["chats"]:
        if chat["id"] == active_chat_id:
            chat["messages"] = st.session_state.messages
            chat["updated_at"] = _now_str()
            if role == "user" and chat.get("title", "New Chat") == "New Chat":
                preview = content.strip().splitlines()[0][:50]
                chat["title"] = preview or "New Chat"
            break

    user_history["active_chat_id"] = active_chat_id
    history_store[user_key] = user_history
    _save_history_store(history_store)
    _set_active_chat_session(user_key, user_history)


def start_new_chat(username: str) -> str:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)

    new_chat = _new_chat()
    user_history["chats"].append(new_chat)
    user_history["active_chat_id"] = new_chat["id"]
    history_store[user_key] = user_history
    _save_history_store(history_store)
    _set_active_chat_session(user_key, user_history)
    return new_chat["id"]


def get_user_chat_summaries(username: str) -> List[Dict[str, Any]]:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)
    history_store[user_key] = user_history
    _save_history_store(history_store)
    return [
        {
            "id": chat["id"],
            "title": chat.get("title", "New Chat"),
            "created_at": chat.get("created_at", "-"),
            "updated_at": chat.get("updated_at", "-"),
            "message_count": len(chat.get("messages", [])),
        }
        for chat in user_history.get("chats", [])
    ]


def get_chat_messages(username: str, chat_id: str) -> List[Dict[str, Any]]:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)
    chat = next((item for item in user_history["chats"] if item["id"] == chat_id), None)
    return chat.get("messages", []) if chat else []


def set_active_chat(username: str, chat_id: str) -> bool:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)
    if chat_id not in {chat["id"] for chat in user_history["chats"]}:
        return False

    user_history["active_chat_id"] = chat_id
    history_store[user_key] = user_history
    _save_history_store(history_store)
    _set_active_chat_session(user_key, user_history)
    return True


def delete_chat(username: str, chat_id: str) -> bool:
    user_key = username.strip() or "guest"
    history_store = _load_history_store()
    user_history = _get_or_create_user_history(history_store, user_key)

    original_count = len(user_history["chats"])
    user_history["chats"] = [chat for chat in user_history["chats"] if chat.get("id") != chat_id]
    if len(user_history["chats"]) == original_count:
        return False

    # Never leave the user without a chat.
    if not user_history["chats"]:
        fresh = _new_chat()
        user_history["chats"] = [fresh]
        user_history["active_chat_id"] = fresh["id"]
    elif user_history.get("active_chat_id") == chat_id:
        user_history["active_chat_id"] = user_history["chats"][-1]["id"]

    history_store[user_key] = user_history
    _save_history_store(history_store)
    _set_active_chat_session(user_key, user_history)
    return True

