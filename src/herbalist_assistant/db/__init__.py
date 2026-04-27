"""SQLite + SQLAlchemy persistence layer.

The public surface of this package is intentionally tiny: callers should
go through :mod:`herbalist_assistant.db.repository` for all reads/writes.
:mod:`engine` owns connection + session lifecycle, :mod:`models` declares
the ORM mapping, and :mod:`migration` runs the idempotent schema bootstrap.
"""

from __future__ import annotations

from .engine import get_engine, init_db, session_scope
from .migration import ensure_database_ready
from .models import Base, ChatMessage, ChatSession, User

__all__ = [
    "Base",
    "ChatMessage",
    "ChatSession",
    "User",
    "ensure_database_ready",
    "get_engine",
    "init_db",
    "session_scope",
]
