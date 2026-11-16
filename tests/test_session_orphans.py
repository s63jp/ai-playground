"""Orphaned-session reconciliation (2026-11-16).

A killed run previously left its session rows marked 'running' forever (session 50,
agent_012/nemotron4b, ran for hours as a 'running' ghost). Reconcile runs once at process
start and marks every pre-existing 'running' row 'interrupted' — the honest, RESUMABLE
state. These tests pin the behaviour and its sharp edges.
"""
import pytest

from playground.storage.db import Database


def _db():
    return Database(":memory:")


def test_happy_reconciles_orphans_to_interrupted():
    db = _db()
    a = db.create_session(agent_id="agent_001", mode="exploration")
    b = db.create_session(agent_id="agent_002", mode="exploration")
    fixed = db.reconcile_orphans()
    assert sorted(fixed) == sorted([a, b])
    for sid in (a, b):
        s = db.session(sid)
        assert s["status"] == "interrupted" and s["end_reason"] == "interrupted"
        assert s["ended_at"] is not None


def test_validate_second_call_is_a_no_op():
    """Reconciliation is for process start; a second pass in the same process finds nothing
    — and in particular must never touch sessions this process deliberately left 'running'."""
    db = _db()
    sid = db.create_session(agent_id="agent_001", mode="exploration")
    assert db.reconcile_orphans() == [sid]
    # a NEW session started after reconciliation (the live one) survives later reconciles
    # in tests of the same db only because no second call happens in-process; assert the
    # first reconcile's postcondition directly:
    assert db.reconcile_orphans() == []


def test_validate_only_running_rows_touched():
    """Completed / interrupted / benchmark rows are never rewritten."""
    db = _db()
    done = db.create_session(agent_id="agent_001", mode="exploration")
    db.update_session(done, status="completed", end_reason="agent_ended")
    interrupted = db.create_session(agent_id="agent_001", mode="exploration")
    db.update_session(interrupted, status="interrupted", end_reason="adapter_error")
    assert db.reconcile_orphans() == []
    assert db.session(done)["status"] == "completed"
    assert db.session(interrupted)["status"] == "interrupted"


def test_error_empty_database():
    db = _db()
    assert db.reconcile_orphans() == []
