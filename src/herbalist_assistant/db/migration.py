"""Database bootstrap hook.

Creates the SQLite schema (tables) on first startup if they don't already
exist.  Safe to call on every Streamlit rerun — the heavy work only happens
once.
"""

from __future__ import annotations

import logging

from .engine import init_db

_logger = logging.getLogger("herbalist_assistant.db")


def ensure_database_ready() -> None:
    """Idempotent startup hook — creates the schema if needed."""
    init_db()
    _logger.debug("Database schema is up-to-date")
