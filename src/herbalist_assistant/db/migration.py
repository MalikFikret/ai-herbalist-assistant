"""One-time migration from the legacy JSON store into SQLite.

The migration is safe to run on every startup: it only does work if
``.users.json`` / ``.chat_history.json`` still exist AND the DB has no
rows for those usernames yet. After a successful copy, the JSON files
are renamed to ``<file>.migrated-backup-<timestamp>`` so they stay
available as a rollback artifact but don't get re-imported.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from herbalist_assistant import config

from .engine import init_db, session_scope
from .models import ChatMessage, ChatSession, User

_logger = logging.getLogger("herbalist_assistant.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        _logger.warning("Legacy store %s exists but is unreadable; skipping", path)
        return None


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_suffix(path.suffix + f".migrated-backup-{stamp}")


def _rename_to_backup(path: Path) -> None:
    if not path.exists():
        return
    target = _backup_path(path)
    try:
        path.rename(target)
        _logger.info("Backed up legacy store %s -> %s", path, target)
    except OSError:
        _logger.exception("Could not rename legacy store %s", path)


# ---------------------------------------------------------------------------
# The import itself
# ---------------------------------------------------------------------------


def _import_users(users_payload: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Insert users that don't already exist. Returns {username: user_id}."""
    imported: dict[str, int] = {}
    with session_scope() as session:
        for username, record in (users_payload or {}).items():
            if not isinstance(record, dict):
                continue
            username = (username or "").strip()
            if not username:
                continue
            existing = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            if existing is not None:
                imported[username] = existing.id
                continue
            profile_dict = record.get("profile") or {}
            if not isinstance(profile_dict, dict):
                profile_dict = {}
            user = User(
                username=username,
                password_hash=str(record.get("password_hash", "")),
                salt=str(record.get("salt", "")),
                role=str(record.get("role", "user")) or "user",
                health_profile_json=json.dumps(
                    {
                        "name": str(profile_dict.get("name", profile_dict.get("full_name", ""))),
                        "age": str(profile_dict.get("age", "")),
                        "gender": str(profile_dict.get("gender", "")),
                        "allergies": str(profile_dict.get("allergies", "")),
                        "conditions": str(
                            profile_dict.get("conditions", profile_dict.get("notes", ""))
                        ),
                    },
                    ensure_ascii=False,
                ),
            )
            session.add(user)
            session.flush()
            imported[username] = user.id
    return imported


def _import_chat_history(history_payload: dict[str, Any]) -> int:
    """Insert chat sessions + messages. Returns the number of sessions imported."""
    sessions_imported = 0
    with session_scope() as session:
        for username, user_history in (history_payload or {}).items():
            username = (username or "").strip() or "guest"
            user = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            if user is None:
                # Guest-only chats may exist without a user record; create a
                # placeholder so FK integrity holds.
                user = User(
                    username=username,
                    password_hash="",
                    salt="",
                    role="guest" if username == "guest" else "user",
                    health_profile_json="{}",
                )
                session.add(user)
                session.flush()

            if not isinstance(user_history, dict):
                continue
            chats: list[dict[str, Any]] = user_history.get("chats") or []
            if not isinstance(chats, list):
                continue
            existing_ids = {c.id for c in user.sessions}
            for chat in chats:
                if not isinstance(chat, dict):
                    continue
                chat_id = str(chat.get("id") or "").strip()
                if not chat_id or chat_id in existing_ids:
                    continue
                new_session = ChatSession(
                    id=chat_id,
                    user_id=user.id,
                    title=str(chat.get("title") or "New Chat"),
                    created_at=str(chat.get("created_at") or ""),
                    updated_at=str(chat.get("updated_at") or chat.get("created_at") or ""),
                )
                session.add(new_session)
                session.flush()
                sessions_imported += 1

                messages = chat.get("messages") or []
                if not isinstance(messages, list):
                    messages = []
                for idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        continue
                    role = str(msg.get("role") or "").strip()
                    content = str(msg.get("content") or "")
                    if role not in {"user", "assistant", "system"}:
                        continue
                    sources = msg.get("sources") or []
                    if not isinstance(sources, list):
                        sources = []
                    feedback = msg.get("feedback")
                    if feedback not in ("up", "down"):
                        feedback = None
                    session.add(
                        ChatMessage(
                            session_id=new_session.id,
                            position=idx,
                            role=role,
                            content=content,
                            timestamp=str(msg.get("timestamp") or ""),
                            sources_json=json.dumps(sources, ensure_ascii=False),
                            feedback=feedback,
                            feedback_at=(
                                str(msg.get("feedback_at") or msg.get("timestamp") or "")
                                if feedback
                                else None
                            ),
                        )
                    )

            # Preserve the user's previously-active chat if possible.
            active_id = user_history.get("active_chat_id")
            if isinstance(active_id, str) and active_id:
                user.active_session_id = active_id
    return sessions_imported


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def migrate_json_to_sqlite(
    *,
    users_path: Path | None = None,
    history_path: Path | None = None,
    backup: bool = True,
) -> dict[str, int]:
    """Perform the one-time migration.

    Parameters
    ----------
    users_path, history_path :
        Override the default :mod:`config` locations. Tests use these.
    backup :
        When True (the default), rename the JSON files out of the way
        after a successful copy so subsequent starts don't redo the work.

    Returns
    -------
    dict with ``users_imported`` and ``chats_imported`` counts.
    """
    users_path = users_path or config.LEGACY_USERS_JSON
    history_path = history_path or config.LEGACY_CHAT_HISTORY_JSON

    users_payload = _read_json_file(users_path) or {}
    history_payload = _read_json_file(history_path) or {}

    if not users_payload and not history_payload:
        return {"users_imported": 0, "chats_imported": 0}

    imported_users = _import_users(users_payload if isinstance(users_payload, dict) else {})
    imported_chats = _import_chat_history(
        history_payload if isinstance(history_payload, dict) else {}
    )

    if backup:
        _rename_to_backup(users_path)
        _rename_to_backup(history_path)

    _logger.info(
        "JSON -> SQLite migration done: %d user rows touched, %d chat sessions imported",
        len(imported_users),
        imported_chats,
    )
    return {"users_imported": len(imported_users), "chats_imported": imported_chats}


def ensure_database_ready() -> None:
    """Idempotent startup hook.

    Creates the schema if needed, then runs the JSON migration (also
    idempotent). Safe to call on every Streamlit rerun; the heavy work
    only happens once thanks to the backup-rename step.

    Safety guard
    ------------
    If ``HA_DB_PATH`` points outside the current working directory
    (e.g. a scratch DB under ``/tmp`` used for smoke tests), we refuse
    to rename the legacy JSON files. The data still migrates, but the
    source files stay put so the real deployment DB can re-import them.
    This prevents the "I pointed the DB at /tmp and my prod files got
    renamed" footgun.
    """
    init_db()

    cwd = Path.cwd().resolve()
    env_db = os.environ.get("HA_DB_PATH", "").strip()
    safe_to_backup = True
    if env_db:
        try:
            resolved = Path(env_db).resolve()
            # Only allow the rename if the DB is inside the current workspace.
            resolved.relative_to(cwd)
        except (ValueError, OSError):
            safe_to_backup = False
            _logger.warning(
                "HA_DB_PATH=%r is outside the current workspace; the JSON "
                "migration will import data but will NOT rename the legacy "
                "files (safety guard against smoke-test footguns).",
                env_db,
            )

    try:
        migrate_json_to_sqlite(backup=safe_to_backup)
    except Exception:
        _logger.exception("JSON -> SQLite migration failed; DB is still usable")
