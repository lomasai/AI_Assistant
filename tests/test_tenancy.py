"""Two orgs, deliberately identical names, must never see each other.

This is the test that makes the multi-tenancy claim real. If it ever goes
red, stop and fix it before anything else - a leak here is the kind of bug
that ends a school contract.
"""
from __future__ import annotations

from array import array

import pytest

from lomas_store import (
    STORES,
    AnswerRepo,
    ClassRepo,
    ConsentRepo,
    EmbeddingRepo,
    EventRepo,
    OrgRepo,
    SchoolRepo,
    ScopeError,
    SessionRepo,
    StudentRepo,
    TenantScope,
    migrate,
)

SEED_TIME = 1_700_000_000.0
COLLIDING_NAME = "Ananya Sharma"
COLLIDING_ROLL = "12"


class Ticker:
    """Deterministic wall clock so ordering assertions are stable."""

    def __init__(self) -> None:
        self.t = SEED_TIME

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


@pytest.fixture
def store():
    backing = STORES.create("memory")
    migrate(backing, SEED_TIME)
    yield backing
    backing.close()


@pytest.fixture
def repos(store):
    tick = Ticker()
    return {
        "org": OrgRepo(store, tick),
        "school": SchoolRepo(store, tick),
        "klass": ClassRepo(store, tick),
        "student": StudentRepo(store, tick),
        "consent": ConsentRepo(store, tick),
        "embedding": EmbeddingRepo(store, tick),
        "session": SessionRepo(store, tick),
        "answer": AnswerRepo(store, tick),
        "event": EventRepo(store, tick),
    }


def seed_tenant(repos, org_id: str) -> TenantScope:
    """Same school name, same class, same student, same roll number."""
    scope = TenantScope(org_id=org_id)
    repos["org"].create(scope, "Sunrise Public School Group")

    scope = scope.narrowed(school_id=f"{org_id}-school")
    repos["school"].create(scope, "Sunrise Public School", "en")

    scope = scope.narrowed(class_id=f"{org_id}-class")
    repos["klass"].create(scope, "6", "B", "science")

    student_id = repos["student"].create(scope, COLLIDING_NAME, COLLIDING_ROLL)
    repos["consent"].grant(scope, student_id, "face_recognition", "principal")

    session_id = repos["session"].open(scope, "en", "photosynthesis", "Mrs Rao")
    repos["session"].mark_present(scope, session_id, student_id)
    repos["answer"].record(scope, session_id, student_id, "q1", "sunlight", True, 2400)
    repos["event"].append(scope, session_id, "session.opened", {"org": org_id})

    vector = array("f", [0.1, 0.2, 0.3, 0.4]).tobytes()
    repos["embedding"].add(scope, student_id, vector, 4, "float32", 0.9, "centre")
    return scope


@pytest.fixture
def two_orgs(repos):
    return seed_tenant(repos, "org-alpha"), seed_tenant(repos, "org-beta")


def test_students_do_not_leak(repos, two_orgs):
    alpha, beta = two_orgs
    for scope, expected in ((alpha, "org-alpha"), (beta, "org-beta")):
        students = repos["student"].list_for_class(scope)
        assert len(students) == 1
        assert students[0]["org_id"] == expected
        assert students[0]["name"] == COLLIDING_NAME


def test_identical_roll_numbers_stay_separate(repos, two_orgs):
    alpha, beta = two_orgs
    a = repos["student"].by_roll(alpha, COLLIDING_ROLL)
    b = repos["student"].by_roll(beta, COLLIDING_ROLL)
    assert a is not None and b is not None
    assert a["id"] != b["id"]


def test_cross_org_read_returns_nothing(repos, two_orgs):
    alpha, beta = two_orgs
    beta_student = repos["student"].by_roll(beta, COLLIDING_ROLL)
    assert repos["student"].get(alpha, beta_student["id"]) is None


def test_sessions_answers_events_are_scoped(repos, two_orgs):
    alpha, beta = two_orgs
    for scope in (alpha, beta):
        sessions = repos["session"].recent(scope, 10)
        assert len(sessions) == 1
        assert sessions[0]["org_id"] == scope.org_id

        session_id = sessions[0]["id"]
        assert len(repos["answer"].for_session(scope, session_id)) == 1
        assert len(repos["event"].for_session(scope, session_id)) == 1

    other_session = repos["session"].recent(beta, 10)[0]["id"]
    assert repos["session"].get(alpha, other_session) is None
    assert repos["answer"].for_session(alpha, other_session) == []


def test_embeddings_are_scoped(repos, two_orgs):
    alpha, beta = two_orgs
    beta_student = repos["student"].by_roll(beta, COLLIDING_ROLL)
    assert repos["embedding"].for_student(alpha, beta_student["id"]) == []
    assert len(repos["embedding"].all_for_class(beta)) == 1


def test_consent_is_scoped(repos, two_orgs):
    alpha, beta = two_orgs
    beta_student = repos["student"].by_roll(beta, COLLIDING_ROLL)
    assert repos["consent"].is_granted(beta, beta_student["id"], "face_recognition")
    assert not repos["consent"].is_granted(alpha, beta_student["id"], "face_recognition")


def test_revoked_consent_stops_counting(repos, two_orgs):
    alpha, _ = two_orgs
    student = repos["student"].by_roll(alpha, COLLIDING_ROLL)
    repos["consent"].revoke(alpha, student["id"], "face_recognition")
    assert not repos["consent"].is_granted(alpha, student["id"], "face_recognition")


def test_delete_cannot_cross_orgs(repos, two_orgs):
    alpha, beta = two_orgs
    beta_student = repos["student"].by_roll(beta, COLLIDING_ROLL)
    repos["student"].delete(alpha, beta_student["id"])
    assert repos["student"].get(beta, beta_student["id"]) is not None


def test_missing_scope_field_raises(repos):
    """A class-scoped repo called with only an org must refuse, not widen."""
    with pytest.raises(ScopeError, match="class_id"):
        repos["student"].list_for_class(TenantScope(org_id="org-alpha"))


def test_scope_is_a_required_argument(repos):
    with pytest.raises(TypeError):
        repos["student"].list_for_class()


def test_embedding_vector_round_trips(repos, two_orgs):
    alpha, _ = two_orgs
    student = repos["student"].by_roll(alpha, COLLIDING_ROLL)
    stored = repos["embedding"].for_student(alpha, student["id"])[0]

    restored = array("f")
    restored.frombytes(stored["vector"])
    assert stored["dim"] == 4
    assert stored["dtype"] == "float32"
    assert [round(v, 4) for v in restored] == [0.1, 0.2, 0.3, 0.4]
