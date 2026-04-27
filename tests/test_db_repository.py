"""End-to-end tests for the DB repository."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "herbalist.db"
    monkeypatch.setenv("HA_DB_PATH", str(db_file))
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
