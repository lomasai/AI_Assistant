-- Every table carries org_id. There is deliberately no column anywhere for
-- an image, a frame or a video: enrolment produces vectors and discards the
-- pictures, and a future contributor cannot casually change that because
-- there is nowhere to put them.

CREATE TABLE IF NOT EXISTS orgs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS schools (
    id                TEXT PRIMARY KEY,
    org_id            TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name              TEXT NOT NULL,
    default_language  TEXT NOT NULL,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_schools_org ON schools(org_id);

CREATE TABLE IF NOT EXISTS classes (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    school_id  TEXT NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
    grade      TEXT NOT NULL,
    section    TEXT NOT NULL,
    subject    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_classes_org_school ON classes(org_id, school_id);

CREATE TABLE IF NOT EXISTS students (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    school_id  TEXT NOT NULL,
    class_id   TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    roll_no    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_students_org_class ON students(org_id, class_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_students_roll ON students(org_id, class_id, roll_no);

-- Enrolment is refused unless a row exists here for the student.
CREATE TABLE IF NOT EXISTS consent (
    id           TEXT PRIMARY KEY,
    org_id       TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    student_id   TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    granted_by   TEXT NOT NULL,
    granted_at   REAL NOT NULL,
    document_ref TEXT,
    revoked_at   REAL
);
CREATE INDEX IF NOT EXISTS ix_consent_student ON consent(org_id, student_id);

-- Vectors only. dim and dtype let a caller rebuild the array without this
-- package needing to know about numpy.
CREATE TABLE IF NOT EXISTS embeddings (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    student_id  TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    dtype       TEXT NOT NULL,
    quality     REAL NOT NULL,
    angle       TEXT NOT NULL,
    captured_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_embeddings_student ON embeddings(org_id, student_id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    school_id  TEXT NOT NULL,
    class_id   TEXT NOT NULL,
    teacher    TEXT,
    topic      TEXT,
    language   TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL,
    status     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sessions_org_class ON sessions(org_id, class_id, started_at);

CREATE TABLE IF NOT EXISTS session_students (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    joined_at  REAL,
    left_at    REAL,
    present    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_session_student
    ON session_students(org_id, session_id, student_id);

CREATE TABLE IF NOT EXISTS answers (
    id           TEXT PRIMARY KEY,
    org_id       TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id   TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    question_ref TEXT NOT NULL,
    response     TEXT,
    correct      INTEGER,
    latency_ms   INTEGER,
    answered_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_answers_session ON answers(org_id, session_id);
CREATE INDEX IF NOT EXISTS ix_answers_student ON answers(org_id, student_id);

-- Append-only. Every report reads from here, so there is no second code path
-- that could disagree with what actually happened.
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id     TEXT NOT NULL,
    session_id TEXT,
    name       TEXT NOT NULL,
    payload    TEXT,
    at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_session ON events(org_id, session_id, at);
