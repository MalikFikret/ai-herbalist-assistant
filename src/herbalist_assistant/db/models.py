"""SQLAlchemy ORM models for users, chat sessions, and chat messages.

Design notes
------------
- ``User.active_session_id`` is *not* a foreign key, to avoid the
  circular FK cycle with ``ChatSession.user_id``. It's a plain string
  that we validate at the repository level when loading a user.
- ``ChatMessage.position`` is maintained by the repository so we can
  render messages in a stable order without depending on auto-increment
  id ordering across databases.
- Timestamps are stored as ``YYYY-MM-DD HH:MM:SS`` strings to match the
  existing JSON format. This avoids an extra parse/format hop when we
  migrate the legacy store and keeps display code unchanged.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for every ORM model in this package."""


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    # JSON blob for {name, age, gender, allergies, conditions}. Stored as
    # TEXT so SQLite stays happy on every platform without the JSON1
    # extension.
    health_profile_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # Tracks which chat the user had open last, equivalent to the old
    # per-user ``active_chat_id`` in .chat_history.json.
    active_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=_now_str)

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatSession.created_at",
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    # uuid4 hex, matches the existing chat id shape so existing URL and
    # session_state references keep working after the migration.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=_now_str)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False, default=_now_str)

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.position",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 0-based ordinal within the session; stable ordering even if we
    # ever switch to a DB whose auto-increment ids aren't monotonic.
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False, default=_now_str)
    # JSON list of source dicts (kind/file/page or kind/url/title).
    sources_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # 'up' | 'down' | NULL. Matches the UI's tri-state toggle.
    feedback: Mapped[str | None] = mapped_column(String(8), nullable=True)
    feedback_at: Mapped[str | None] = mapped_column(String(32), nullable=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
