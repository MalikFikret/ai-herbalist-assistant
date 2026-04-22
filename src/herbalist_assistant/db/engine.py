"""Engine / session-factory / init_db helpers.

A single SQLAlchemy engine is cached per-process. Callers should use
:func:`session_scope` as a context manager rather than creating sessions
directly -- it commits on success, rolls back on exceptions, and always
closes the session.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from herbalist_assistant import config

from .models import Base

_logger = logging.getLogger("herbalist_assistant.db")


def _resolve_db_path() -> Path:
    """Where to put the SQLite file.

    Env var ``HA_DB_PATH`` wins (useful for containers or tests); otherwise
    we use ``config.DB_PATH`` relative to the current working directory,
    matching how the legacy JSON files were located.
    """
    env_path = os.environ.get("HA_DB_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return Path(config.DB_PATH)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine.

    Streamlit re-executes the script on every interaction from multiple
    threads. ``check_same_thread=False`` lets those threads share the
    connection, and ``sessionmaker`` (see :func:`_session_factory`)
    gives each call its own Session on top.
    """
    db_path = _resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )

    # Enable foreign-key enforcement on SQLite (off by default).
    @event.listens_for(engine, "connect")
    def _fk_pragma_on_connect(dbapi_connection, _connection_record):  # type: ignore[no-redef]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call repeatedly."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _logger.debug("init_db: ensured schema at %s", engine.url)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope around a series of operations.

    Usage::

        with session_scope() as session:
            user = session.get(User, 1)
            ...

    Commits on clean exit, rolls back on exceptions.
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Drop the cached engine/session factory.

    Used by tests that swap the DB path (via ``HA_DB_PATH`` + monkeypatch);
    production code never calls this.
    """
    get_engine.cache_clear()
    _session_factory.cache_clear()
