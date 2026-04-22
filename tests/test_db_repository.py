"""End-to-end tests for the DB repository + JSON-to-SQLite migration."""

from __future__ import annotations

import json
import sys

import pytest


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "herbalist.db"
    monkeypatch.setenv("HA_DB_PATH", str(db_file))
    # Isolate the JSON migration from the real repo: point it at empty
    # paths inside tmp_path so a stray legacy file in the workspace
    # can't accidentally get renamed.
    from herbalist_assistant import config as _config

    monkeypatch.setattr(_config, "LEGACY_USERS_JSON", tmp_path / ".users.json", raising=False)
    monkeypatch.setattr(
        _config, "LEGACY_CHAT_HISTORY_JSON", tmp_path / ".chat_history.json", raising=False
    )
    for mod in [
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
    yield tmp_path
    reset_engine_cache()


def _repo():
    from herbalist_assistant.db import repository

    return repository


def test_create_and_authenticate_user(fresh_db):
    repo = _repo()
    assert repo.create_user(
        username="alice", password_hash="hash-a", salt="salt-a", role="user"
    )
    # Duplicate creation is rejected.
    assert not repo.create_user(username="alice", password_hash="x", salt="y")

    auth = repo.get_user_auth("alice")
    assert auth == {"password_hash": "hash-a", "salt": "salt-a", "role": "user"}
    assert repo.get_user_auth("nobody") is None


def test_profile_round_trip_defaults_and_persists(fresh_db):
    repo = _repo()
    repo.create_user(username="alice", password_hash="h", salt="s")

    default = repo.get_user_profile("alice")
    assert default == {
        "name": "",
        "age": "",
        "gender": "",
        "allergies": "",
        "conditions": "",
    }

    assert repo.save_user_profile(
        "alice",
        {"name": "Alice", "age": "30", "gender": "F", "allergies": "nettle", "conditions": ""},
    )
    profile = repo.get_user_profile("alice")
    assert profile["name"] == "Alice"
    assert profile["age"] == "30"
    assert profile["allergies"] == "nettle"


def test_chat_lifecycle(fresh_db):
    repo = _repo()
    repo.create_user(username="bob", password_hash="h", salt="s")

    active_before = repo.load_active_chat("bob")
    assert active_before["messages"] == []
    chat_id = active_before["active_chat_id"]
    assert isinstance(chat_id, str) and chat_id

    result1 = repo.append_message(
        username="bob",
        chat_id=chat_id,
        role="user",
        content="Hello!",
    )
    assert result1["position"] == 0
    assert result1["chat_id"] == chat_id

    result2 = repo.append_message(
        username="bob",
        chat_id=chat_id,
        role="assistant",
        content="Hi there",
        sources=[{"kind": "pdf", "file": "ref.pdf", "page": 3}],
    )
    assert result2["position"] == 1

    messages = repo.get_chat_messages("bob", chat_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"] == [{"kind": "pdf", "file": "ref.pdf", "page": 3}]
    assert messages[1]["feedback"] is None


def test_title_promotion_from_first_user_message(fresh_db):
    repo = _repo()
    repo.create_user(username="cara", password_hash="h", salt="s")
    snap = repo.load_active_chat("cara")
    chat_id = snap["active_chat_id"]

    repo.append_message(
        username="cara",
        chat_id=chat_id,
        role="user",
        content="Benefits of chamomile tea for headache?",
    )
    summaries = repo.get_user_chat_summaries("cara")
    assert summaries[0]["title"].startswith("Benefits of chamomile")


def test_feedback_update_and_aggregation(fresh_db):
    repo = _repo()
    repo.create_user(username="dan", password_hash="h", salt="s")
    snap = repo.load_active_chat("dan")
    chat_id = snap["active_chat_id"]

    repo.append_message(username="dan", chat_id=chat_id, role="user", content="q")
    repo.append_message(username="dan", chat_id=chat_id, role="assistant", content="a")

    assert repo.update_message_feedback(
        username="dan", chat_id=chat_id, message_index=1, feedback="up"
    )
    assert not repo.update_message_feedback(
        username="dan", chat_id=chat_id, message_index=0, feedback="up"
    )  # can't rate a user message
    assert not repo.update_message_feedback(
        username="dan", chat_id=chat_id, message_index=99, feedback="up"
    )
    assert not repo.update_message_feedback(
        username="dan", chat_id=chat_id, message_index=1, feedback="maybe"
    )

    entries = repo.iter_all_feedback()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["username"] == "dan"
    assert entry["feedback"] == "up"
    assert entry["question"] == "q"
    assert entry["answer"] == "a"


def test_start_delete_and_set_active(fresh_db):
    repo = _repo()
    repo.create_user(username="evy", password_hash="h", salt="s")
    first = repo.load_active_chat("evy")["active_chat_id"]

    new_id = repo.start_new_chat("evy")
    assert new_id != first
    summaries = repo.get_user_chat_summaries("evy")
    assert {s["id"] for s in summaries} == {first, new_id}

    assert repo.set_active_chat("evy", first)
    assert repo.load_active_chat("evy")["active_chat_id"] == first

    assert repo.delete_chat("evy", first)
    # Deleting the active chat reassigns to the remaining one (never empty).
    assert repo.load_active_chat("evy")["active_chat_id"] == new_id


def test_ownership_enforced(fresh_db):
    repo = _repo()
    repo.create_user(username="owner", password_hash="h", salt="s")
    repo.create_user(username="intruder", password_hash="h", salt="s")
    chat_id = repo.load_active_chat("owner")["active_chat_id"]
    repo.append_message(username="owner", chat_id=chat_id, role="user", content="mine")

    assert repo.get_chat_messages("intruder", chat_id) == []
    assert not repo.set_active_chat("intruder", chat_id)
    assert not repo.delete_chat("intruder", chat_id)


# ---------------------------------------------------------------------------
# JSON -> SQLite migration
# ---------------------------------------------------------------------------


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_json_migration_imports_users_and_chats(tmp_path, monkeypatch):
    db_file = tmp_path / "herbalist.db"
    users_path = tmp_path / ".users.json"
    history_path = tmp_path / ".chat_history.json"

    _write_json(
        users_path,
        {
            "alice": {
                "password_hash": "hash-a",
                "salt": "salt-a",
                "role": "user",
                "profile": {
                    "name": "Alice",
                    "age": "30",
                    "gender": "F",
                    "allergies": "chamomile",
                    "conditions": "migraine",
                },
            }
        },
    )
    _write_json(
        history_path,
        {
            "alice": {
                "active_chat_id": "chat-1",
                "chats": [
                    {
                        "id": "chat-1",
                        "title": "Headache remedies",
                        "created_at": "2024-01-01 10:00:00",
                        "updated_at": "2024-01-01 10:05:00",
                        "messages": [
                            {
                                "role": "user",
                                "content": "What helps headache?",
                                "timestamp": "2024-01-01 10:00:00",
                            },
                            {
                                "role": "assistant",
                                "content": "Try peppermint tea.",
                                "timestamp": "2024-01-01 10:00:10",
                                "sources": [
                                    {"kind": "pdf", "file": "herbs.pdf", "page": 5}
                                ],
                                "feedback": "up",
                                "feedback_at": "2024-01-01 10:01:00",
                            },
                        ],
                    }
                ],
            },
            "guest": {
                "active_chat_id": "chat-guest",
                "chats": [
                    {
                        "id": "chat-guest",
                        "title": "Guest thoughts",
                        "created_at": "2024-01-02 09:00:00",
                        "updated_at": "2024-01-02 09:00:00",
                        "messages": [],
                    }
                ],
            },
        },
    )

    monkeypatch.setenv("HA_DB_PATH", str(db_file))
    for mod in [
        "herbalist_assistant.db",
        "herbalist_assistant.db.engine",
        "herbalist_assistant.db.migration",
        "herbalist_assistant.db.repository",
    ]:
        sys.modules.pop(mod, None)
    from herbalist_assistant.db import init_db
    from herbalist_assistant.db import repository as repo
    from herbalist_assistant.db.engine import reset_engine_cache
    from herbalist_assistant.db.migration import migrate_json_to_sqlite

    reset_engine_cache()
    init_db()
    result = migrate_json_to_sqlite(
        users_path=users_path,
        history_path=history_path,
        backup=True,
    )
    assert result == {"users_imported": 1, "chats_imported": 2}

    # Auth + profile survived.
    auth = repo.get_user_auth("alice")
    assert auth == {"password_hash": "hash-a", "salt": "salt-a", "role": "user"}
    profile = repo.get_user_profile("alice")
    assert profile["name"] == "Alice"
    assert profile["allergies"] == "chamomile"

    # Chat history + feedback survived end-to-end.
    messages = repo.get_chat_messages("alice", "chat-1")
    assert len(messages) == 2
    assert messages[1]["sources"] == [{"kind": "pdf", "file": "herbs.pdf", "page": 5}]
    assert messages[1]["feedback"] == "up"

    # Guest-only chats also come across.
    guest_summaries = repo.get_user_chat_summaries("guest")
    assert {s["id"] for s in guest_summaries} == {"chat-guest"}

    # Active chat pointer is preserved.
    snap = repo.load_active_chat("alice")
    assert snap["active_chat_id"] == "chat-1"

    # Original JSON files have been renamed to .migrated-backup-* so a
    # second startup does not re-import them.
    assert not users_path.exists()
    assert not history_path.exists()
    assert list(tmp_path.glob(".users.json.migrated-backup-*"))
    assert list(tmp_path.glob(".chat_history.json.migrated-backup-*"))

    reset_engine_cache()


def test_migration_is_idempotent_when_no_legacy_files(tmp_path, monkeypatch):
    db_file = tmp_path / "herbalist.db"
    users_path = tmp_path / ".users.json"
    history_path = tmp_path / ".chat_history.json"

    monkeypatch.setenv("HA_DB_PATH", str(db_file))
    for mod in [
        "herbalist_assistant.db",
        "herbalist_assistant.db.engine",
        "herbalist_assistant.db.migration",
        "herbalist_assistant.db.repository",
    ]:
        sys.modules.pop(mod, None)
    from herbalist_assistant.db import init_db
    from herbalist_assistant.db.engine import reset_engine_cache
    from herbalist_assistant.db.migration import migrate_json_to_sqlite

    reset_engine_cache()
    init_db()

    result = migrate_json_to_sqlite(users_path=users_path, history_path=history_path)
    assert result == {"users_imported": 0, "chats_imported": 0}

    reset_engine_cache()


def test_stats_reflect_inserts(fresh_db):
    repo = _repo()
    repo.create_user(username="u1", password_hash="h", salt="s")
    repo.create_user(username="u2", password_hash="h", salt="s")
    snap = repo.load_active_chat("u1")
    repo.append_message(
        username="u1", chat_id=snap["active_chat_id"], role="user", content="hi"
    )
    stats = repo.stats()
    assert stats["users"] >= 2
    assert stats["chats"] >= 1
    assert stats["messages"] >= 1
