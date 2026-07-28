"""SQLite persistence for student registration and local recognition."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from server.config import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudentStoreError(Exception):
    """Raised when student persistence cannot complete a requested action."""


class SQLiteStudentRepository:
    """Local SQLite store for student profiles, consent and face embeddings."""

    SCHEMA_VERSION = 2

    def __init__(self, sqlite_path: str) -> None:
        self.path = self._resolve_path(sqlite_path)
        self._lock = asyncio.Lock()
        self._initialized = False

    @staticmethod
    def _resolve_path(sqlite_path: str) -> Path:
        path = Path(sqlite_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            if not self._initialized:
                await asyncio.to_thread(self._initialize_sync)
                self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_sync(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS students (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL COLLATE NOCASE,
                    grade_level TEXT,
                    language TEXT,
                    consent_given INTEGER NOT NULL DEFAULT 0,
                    consent_timestamp_utc TEXT,
                    registration_status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    deleted_at_utc TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_students_active_name
                    ON students(display_name)
                    WHERE deleted_at_utc IS NULL;

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    quality REAL NOT NULL,
                    pose_label TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS registration_sessions (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required_samples INTEGER NOT NULL,
                    accepted_samples INTEGER NOT NULL DEFAULT 0,
                    rejected_samples INTEGER NOT NULL DEFAULT 0,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS teaching_sessions (
                    id TEXT PRIMARY KEY,
                    student_id TEXT,
                    topic TEXT NOT NULL,
                    state TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(teaching_sessions)").fetchall()}
            if "state" not in columns:
                conn.execute("ALTER TABLE teaching_sessions ADD COLUMN state TEXT")
            if "active" not in columns:
                conn.execute("ALTER TABLE teaching_sessions ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at_utc) VALUES (?, ?)",
                (self.SCHEMA_VERSION, utc_now()),
            )

    async def create_student(
        self,
        display_name: str,
        grade_level: str | None,
        language: str | None,
        consent_given: bool,
        status: str = "pending",
    ) -> dict[str, Any]:
        await self._ensure_initialized()
        clean_name = display_name.strip()
        if not clean_name:
            raise StudentStoreError("Student name is required.")
        now = utc_now()
        student = {
            "id": str(uuid4()),
            "display_name": clean_name,
            "grade_level": (grade_level or "").strip() or None,
            "language": (language or "").strip() or None,
            "consent_given": consent_given,
            "consent_timestamp_utc": now if consent_given else None,
            "registration_status": status,
            "created_at_utc": now,
            "updated_at_utc": now,
            "deleted_at_utc": None,
        }
        async with self._lock:
            try:
                await asyncio.to_thread(self._insert_student_sync, student)
            except sqlite3.IntegrityError as exc:
                raise StudentStoreError("A student with this name already exists.") from exc
        return dict(student)

    def _insert_student_sync(self, student: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO students (
                    id, display_name, grade_level, language, consent_given,
                    consent_timestamp_utc, registration_status, created_at_utc,
                    updated_at_utc, deleted_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student["id"],
                    student["display_name"],
                    student["grade_level"],
                    student["language"],
                    1 if student["consent_given"] else 0,
                    student["consent_timestamp_utc"],
                    student["registration_status"],
                    student["created_at_utc"],
                    student["updated_at_utc"],
                    student["deleted_at_utc"],
                ),
            )

    async def list_students(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._list_students_sync, include_deleted)

    def _list_students_sync(self, include_deleted: bool) -> list[dict[str, Any]]:
        query = "SELECT * FROM students"
        if not include_deleted:
            query += " WHERE deleted_at_utc IS NULL"
        query += " ORDER BY display_name"
        with closing(self._connect()) as conn, conn:
            return [self._student_row_to_dict(row) for row in conn.execute(query).fetchall()]

    async def get_student(self, student_id: str) -> dict[str, Any] | None:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._get_student_sync, student_id)

    def _get_student_sync(self, student_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM students WHERE id = ? AND deleted_at_utc IS NULL",
                (student_id,),
            ).fetchone()
        return self._student_row_to_dict(row) if row else None

    async def delete_student(self, student_id: str) -> bool:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._delete_student_sync, student_id)

    def _delete_student_sync(self, student_id: str) -> bool:
        now = utc_now()
        with closing(self._connect()) as conn, conn:
            result = conn.execute(
                """
                UPDATE students
                SET deleted_at_utc = ?, registration_status = 'deleted', updated_at_utc = ?
                WHERE id = ? AND deleted_at_utc IS NULL
                """,
                (now, now, student_id),
            )
            conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        return result.rowcount > 0

    async def update_student_status(self, student_id: str, status: str) -> None:
        await self._ensure_initialized()
        async with self._lock:
            await asyncio.to_thread(self._update_student_status_sync, student_id, status)

    def _update_student_status_sync(self, student_id: str, status: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE students SET registration_status = ?, updated_at_utc = ? WHERE id = ?",
                (status, utc_now(), student_id),
            )

    async def save_embedding(self, student_id: str, embedding: list[float], quality: float, pose_label: str) -> dict[str, Any]:
        await self._ensure_initialized()
        record = {
            "id": str(uuid4()),
            "student_id": student_id,
            "embedding": list(embedding),
            "quality": quality,
            "pose_label": pose_label,
            "created_at_utc": utc_now(),
        }
        async with self._lock:
            await asyncio.to_thread(self._save_embedding_sync, record)
        return record

    def _save_embedding_sync(self, record: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO face_embeddings (id, student_id, embedding_json, quality, pose_label, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["student_id"],
                    json.dumps(record["embedding"], separators=(",", ":")),
                    record["quality"],
                    record["pose_label"],
                    record["created_at_utc"],
                ),
            )

    async def get_embeddings(self, student_id: str | None = None) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._get_embeddings_sync, student_id)

    def _get_embeddings_sync(self, student_id: str | None) -> list[dict[str, Any]]:
        query = """
            SELECT e.*, s.display_name, s.registration_status
            FROM face_embeddings e
            JOIN students s ON s.id = e.student_id
            WHERE s.deleted_at_utc IS NULL
        """
        args: tuple[Any, ...] = ()
        if student_id:
            query += " AND e.student_id = ?"
            args = (student_id,)
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(query, args).fetchall()
        return [
            {
                "id": row["id"],
                "student_id": row["student_id"],
                "embedding": json.loads(row["embedding_json"]),
                "quality": float(row["quality"]),
                "pose_label": row["pose_label"],
                "created_at_utc": row["created_at_utc"],
                "display_name": row["display_name"],
                "registration_status": row["registration_status"],
            }
            for row in rows
        ]

    async def start_registration_session(self, student_id: str, required_samples: int) -> dict[str, Any]:
        await self._ensure_initialized()
        session = {
            "id": str(uuid4()),
            "student_id": student_id,
            "status": "capturing",
            "required_samples": required_samples,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }
        async with self._lock:
            await asyncio.to_thread(self._insert_registration_session_sync, session)
        return dict(session)

    def _insert_registration_session_sync(self, session: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO registration_sessions (
                    id, student_id, status, required_samples, accepted_samples,
                    rejected_samples, created_at_utc, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session["id"],
                    session["student_id"],
                    session["status"],
                    session["required_samples"],
                    session["accepted_samples"],
                    session["rejected_samples"],
                    session["created_at_utc"],
                    session["updated_at_utc"],
                ),
            )

    async def get_registration_session(self, registration_id: str) -> dict[str, Any] | None:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._get_registration_session_sync, registration_id)

    def _get_registration_session_sync(self, registration_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT * FROM registration_sessions WHERE id = ?", (registration_id,)).fetchone()
        return dict(row) if row else None

    async def update_registration_session(
        self,
        registration_id: str,
        *,
        status: str | None = None,
        accepted_delta: int = 0,
        rejected_delta: int = 0,
    ) -> dict[str, Any]:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(
                self._update_registration_session_sync,
                registration_id,
                status,
                accepted_delta,
                rejected_delta,
            )

    def _update_registration_session_sync(
        self,
        registration_id: str,
        status: str | None,
        accepted_delta: int,
        rejected_delta: int,
    ) -> dict[str, Any]:
        with closing(self._connect()) as conn, conn:
            current = conn.execute("SELECT * FROM registration_sessions WHERE id = ?", (registration_id,)).fetchone()
            if current is None:
                raise StudentStoreError("Registration session not found.")
            next_status = status or current["status"]
            conn.execute(
                """
                UPDATE registration_sessions
                SET status = ?,
                    accepted_samples = accepted_samples + ?,
                    rejected_samples = rejected_samples + ?,
                    updated_at_utc = ?
                WHERE id = ?
                """,
                (next_status, accepted_delta, rejected_delta, utc_now(), registration_id),
            )
            row = conn.execute("SELECT * FROM registration_sessions WHERE id = ?", (registration_id,)).fetchone()
        return dict(row)

    async def create_session(self, student_id: str | None, topic: str) -> dict[str, Any]:
        await self._ensure_initialized()
        session = {
            "id": str(uuid4()),
            "student_id": student_id,
            "topic": topic,
            "created_at_utc": utc_now(),
        }
        await self.save_session(session)
        return dict(session)

    async def save_session(self, session: dict[str, Any]) -> None:
        await self._ensure_initialized()
        session_id = str(session.get("id") or uuid4())
        payload = dict(session)
        payload["id"] = session_id
        now = utc_now()
        created = str(payload.get("created_at_utc") or now)
        async with self._lock:
            await asyncio.to_thread(self._save_session_sync, payload, created, now)

    def _save_session_sync(self, session: dict[str, Any], created: str, updated: str) -> None:
        state = str(session.get("state") or "")
        active = 0 if state == "session_complete" else 1
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO teaching_sessions (id, student_id, topic, state, active, payload_json, created_at_utc, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    student_id = excluded.student_id,
                    topic = excluded.topic,
                    state = excluded.state,
                    active = excluded.active,
                    payload_json = excluded.payload_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    str(session["id"]),
                    session.get("student_id"),
                    str(session.get("topic") or session.get("config", {}).get("topic") or ""),
                    state,
                    active,
                    json.dumps(session, separators=(",", ":"), default=str),
                    created,
                    updated,
                ),
            )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._get_session_sync, session_id)

    def _get_session_sync(self, session_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT payload_json FROM teaching_sessions WHERE id = ?", (session_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    async def list_active_sessions(self) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._list_active_sessions_sync)

    def _list_active_sessions_sync(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT payload_json FROM teaching_sessions WHERE active = 1 ORDER BY updated_at_utc DESC"
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    async def schema_version(self) -> int:
        await self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._schema_version_sync)

    def _schema_version_sync(self) -> int:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
        return int(row["version"] or 0)

    @staticmethod
    def _student_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["consent_given"] = bool(data["consent_given"])
        return data
