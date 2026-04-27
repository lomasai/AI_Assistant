"""Memory system with SQLite logs and vector-placeholder summary retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class MemoryError(Exception):
    """Raised when memory operations fail."""


class SummarizerProtocol(Protocol):
    """Optional protocol for LLM-based summarization."""

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        """Generate summary text."""


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for memory storage and retrieval behavior."""

    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "logs.db")
    embedding_dim: int = 128
    recent_window_size: int = 8
    summary_window_size: int = 12
    summary_min_chars: int = 24
    summary_max_items: int = 6
    important_keywords: set[str] = field(
        default_factory=lambda: {
            "name",
            "age",
            "medicine",
            "medication",
            "allergy",
            "pain",
            "doctor",
            "hospital",
            "fall",
            "emergency",
            "reminder",
            "schedule",
            "appointment",
            "preference",
            "favorite",
            "address",
            "phone",
            "sleep",
            "water",
            "track",
            "tracking",
            "follow me",
            "routine",
        }
    )


@dataclass(slots=True)
class SummaryMatch:
    """A retrieved summary with similarity score."""

    id: int
    summary: str
    source: str | None
    score: float
    metadata: dict[str, Any]
    created_at: str


class MemoryService:
    """Conversation + summary memory service.

    - Conversation logs stored in SQLite (`conversation_logs`)
    - Summaries stored in SQLite (`summary_memory`) with placeholder vectors
    - Relevant context retrieval via cosine similarity + recent logs
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        summarizer: SummarizerProtocol | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.summarizer = summarizer
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema if needed."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._init_db_sync)
            self._initialized = True

    async def store_conversation_log(
        self,
        role: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist a conversation message and return row id."""
        await self.initialize()
        role_clean = role.strip().lower()
        text_clean = self._normalize_text(message)
        if not role_clean or not text_clean:
            raise MemoryError("role and message are required.")

        metadata_json = json.dumps(metadata or {}, ensure_ascii=True, default=str)
        created_at = self._utc_now()
        return await asyncio.to_thread(
            self._insert_conversation_sync,
            role_clean,
            text_clean,
            metadata_json,
            created_at,
        )

    async def store_summary(
        self,
        summary: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist summary with vector placeholder and return row id."""
        await self.initialize()
        summary_clean = self._normalize_text(summary)
        if not summary_clean:
            raise MemoryError("summary cannot be empty.")

        vector = self._embed_text(summary_clean, dim=self.config.embedding_dim)
        row_metadata = dict(metadata or {})
        row_metadata.setdefault("embedding_provider", "placeholder_hash")
        created_at = self._utc_now()

        return await asyncio.to_thread(
            self._insert_summary_sync,
            summary_clean,
            source,
            json.dumps(vector, ensure_ascii=True),
            json.dumps(row_metadata, ensure_ascii=True, default=str),
            created_at,
        )

    async def retrieve_relevant_context(
        self,
        query: str,
        top_k: int = 3,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant summaries + recent conversation context."""
        await self.initialize()
        query_clean = self._normalize_text(query)
        if not query_clean:
            raise MemoryError("query cannot be empty.")

        query_vec = self._embed_text(query_clean, dim=self.config.embedding_dim)
        recent_limit = recent_k if recent_k is not None else self.config.recent_window_size

        recent_logs, summary_rows = await asyncio.to_thread(
            self._fetch_context_sync,
            max(1, recent_limit),
        )
        summary_matches = self._rank_summaries(query_vec, summary_rows, top_k=max(1, top_k))

        return {
            "query": query_clean,
            "recent_logs": recent_logs,
            "summary_matches": [self._summary_match_to_dict(item) for item in summary_matches],
        }

    async def get_recent_logs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch most recent conversation logs."""
        await self.initialize()
        return await asyncio.to_thread(self._fetch_recent_logs_sync, max(1, limit))

    async def summarize_and_store_recent(
        self,
        limit: int | None = None,
        source: str = "conversation",
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """Summarize recent logs and store only useful information."""
        await self.initialize()
        window = limit if limit is not None else self.config.summary_window_size
        logs = await self.get_recent_logs(limit=max(1, window))

        important_logs = self._extract_important_logs(logs)
        if not important_logs:
            return {
                "stored": False,
                "reason": "no_important_info",
                "summary": None,
                "important_log_ids": [],
            }

        if use_llm and self.summarizer is not None:
            summary_text = await self._summarize_with_llm(important_logs)
            summary_method = "llm"
        else:
            summary_text = self._summarize_with_rules(important_logs)
            summary_method = "rule_based"

        if not self._is_useful_summary(summary_text):
            return {
                "stored": False,
                "reason": "summary_not_useful",
                "summary": summary_text,
                "important_log_ids": [log["id"] for log in important_logs],
            }

        summary_id = await self.store_summary(
            summary=summary_text,
            source=source,
            metadata={
                "summary_method": summary_method,
                "important_log_ids": [log["id"] for log in important_logs],
            },
        )
        return {
            "stored": True,
            "summary_id": summary_id,
            "summary": summary_text,
            "important_log_ids": [log["id"] for log in important_logs],
        }

    def _init_db_sync(self) -> None:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    source TEXT,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _insert_conversation_sync(self, role: str, message: str, metadata_json: str, created_at: str) -> int:
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_logs (role, message, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (role, message, metadata_json, created_at),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def _insert_summary_sync(
        self,
        summary: str,
        source: str | None,
        embedding_json: str,
        metadata_json: str,
        created_at: str,
    ) -> int:
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO summary_memory (summary, source, embedding_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (summary, source, embedding_json, metadata_json, created_at),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def _fetch_recent_logs_sync(self, limit: int) -> list[dict[str, Any]]:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, role, message, metadata_json, created_at
                FROM conversation_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        ordered = list(reversed(rows))
        result: list[dict[str, Any]] = []
        for row in ordered:
            result.append(
                {
                    "id": int(row["id"]),
                    "role": row["role"],
                    "message": row["message"],
                    "metadata": self._safe_json_load(row["metadata_json"], default={}),
                    "created_at": row["created_at"],
                }
            )
        return result

    def _fetch_context_sync(self, recent_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        recent_logs = self._fetch_recent_logs_sync(recent_limit)
        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, summary, source, embedding_json, metadata_json, created_at
                FROM summary_memory
                """
            ).fetchall()

        summaries: list[dict[str, Any]] = []
        for row in rows:
            summaries.append(
                {
                    "id": int(row["id"]),
                    "summary": row["summary"],
                    "source": row["source"],
                    "embedding": self._safe_json_load(row["embedding_json"], default=[]),
                    "metadata": self._safe_json_load(row["metadata_json"], default={}),
                    "created_at": row["created_at"],
                }
            )
        return recent_logs, summaries

    def _extract_important_logs(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Select logs containing important or actionable information."""
        selected: list[dict[str, Any]] = []
        for log in logs:
            message = str(log.get("message", ""))
            role = str(log.get("role", "")).lower()
            if self._importance_score(role=role, message=message) >= 3:
                selected.append(log)

        # Keep only the most recent important subset.
        if len(selected) > self.config.summary_max_items:
            return selected[-self.config.summary_max_items :]
        return selected

    def _importance_score(self, role: str, message: str) -> int:
        text = message.lower()
        score = 0

        if role == "user":
            score += 1
        if any(keyword in text for keyword in self.config.important_keywords):
            score += 2
        if re.search(r"\b\d{1,2}(:\d{2})?\s?(am|pm)?\b", text):
            score += 1
        if re.search(r"\b(remind|remember|schedule|track|follow|call|take)\b", text):
            score += 1
        if len(text.split()) >= 14:
            score += 1
        return score

    def _summarize_with_rules(self, important_logs: list[dict[str, Any]]) -> str:
        """Compress important dialogue into a short factual summary."""
        compact_lines: list[str] = []
        for log in important_logs:
            role = str(log.get("role", "unknown")).lower()
            prefix = "User" if role == "user" else "Assistant"
            compact = self._compress_message(str(log.get("message", "")))
            if compact:
                compact_lines.append(f"{prefix}: {compact}")

        return " | ".join(compact_lines)

    async def _summarize_with_llm(self, important_logs: list[dict[str, Any]]) -> str:
        """Use injected LLM summarizer when available."""
        if self.summarizer is None:
            return self._summarize_with_rules(important_logs)

        transcript = "\n".join(
            f"{str(log.get('role', '')).lower()}: {str(log.get('message', ''))}"
            for log in important_logs
        )
        prompt = (
            "Summarize only important long-term memory facts from this conversation. "
            "Keep it short and factual. Exclude greetings and filler.\n\n"
            f"{transcript}"
        )
        try:
            summary = await self.summarizer.generate(prompt=prompt)
        except Exception:  # noqa: BLE001
            return self._summarize_with_rules(important_logs)
        return self._normalize_text(summary)

    def _is_useful_summary(self, summary: str) -> bool:
        text = self._normalize_text(summary)
        if len(text) < self.config.summary_min_chars:
            return False
        if text.lower() in {"hello", "hi", "thanks", "thank you", "ok", "okay"}:
            return False
        return True

    @staticmethod
    def _compress_message(message: str, max_words: int = 18) -> str:
        text = " ".join(message.strip().split())
        if not text:
            return ""
        # Trim low-signal filler phrases.
        text = re.sub(r"\b(please|kindly|actually|basically|just)\b", "", text, flags=re.IGNORECASE)
        words = [word for word in text.split() if word]
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + "..."

    def _rank_summaries(
        self,
        query_embedding: list[float],
        summaries: list[dict[str, Any]],
        top_k: int,
    ) -> list[SummaryMatch]:
        ranked: list[SummaryMatch] = []
        for row in summaries:
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                continue
            vector = [float(v) for v in embedding]
            score = self._cosine_similarity(query_embedding, vector)
            ranked.append(
                SummaryMatch(
                    id=int(row["id"]),
                    summary=str(row["summary"]),
                    source=row.get("source"),
                    score=score,
                    metadata=row.get("metadata", {}),
                    created_at=str(row["created_at"]),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _summary_match_to_dict(match: SummaryMatch) -> dict[str, Any]:
        return {
            "id": match.id,
            "summary": match.summary,
            "source": match.source,
            "score": round(match.score, 6),
            "metadata": match.metadata,
            "created_at": match.created_at,
        }

    @staticmethod
    def _embed_text(text: str, dim: int) -> list[float]:
        """Deterministic hash-based embedding placeholder."""
        vector = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % dim
            vector[idx] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            return 0.0
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))

    @staticmethod
    def _safe_json_load(raw: Any, default: Any) -> Any:
        if isinstance(raw, (dict, list)):
            return raw
        if not isinstance(raw, str):
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


memory_service = MemoryService()
