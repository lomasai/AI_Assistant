from __future__ import annotations

import json
from typing import Any

from lomas_core.contracts import (
    LESSON_SEGMENT,
    QUESTION_ANSWERED,
    QUESTION_ASKED,
    QUIZ_POSED,
)
from lomas_core.errors import LomasError
from lomas_core.schema import Config
from lomas_store import TenantScope

from app.content import ContentLibrary

CORRECT = "correct"
PRESENT = "present"
NO_ANSWER = None


class ReportBuilder:
    """What happened, read back from the append-only log.

    There is no attention score in here, no ranking and no percentage per
    child. A number that says how much a ten year old looked at a robot is
    not a fact about her, and once it is in a document a parent can read it
    becomes one. Coverage and answers are facts; the rest is not.
    """

    def __init__(self, cfg: Config, repos: dict[str, Any], content: ContentLibrary) -> None:
        self.cfg = cfg
        self.repos = repos
        self.content = content

    def recent(self, scope: TenantScope) -> list[dict]:
        return [
            {
                "id": row["id"],
                "topic": row["topic"],
                "language": row["language"],
                "teacher": row["teacher"] or "",
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "status": row["status"],
            }
            for row in self.repos["session"].recent(scope, self.cfg.teacher.recent_sessions)
        ]

    def build(self, scope: TenantScope, session_id: str) -> dict:
        session = self.repos["session"].get(scope, session_id)
        if session is None:
            raise LomasError(f"no session {session_id} in org '{scope.org_id}'")

        events = self.repos["event"].for_session(scope, session_id)
        roster = self.repos["session"].roster(scope, session_id)
        answers = self.repos["answer"].for_session(scope, session_id)

        return {
            "session": {
                "id": session_id,
                "topic": session["topic"],
                "language": session["language"],
                "teacher": session["teacher"] or "",
                "started_at": session["started_at"],
                "ended_at": session["ended_at"],
                "minutes": _minutes(session),
                "status": session["status"],
            },
            "attendance": self._attendance(roster),
            "coverage": self._coverage(events, session),
            "questions": self._questions(events),
            "quiz": self._quiz(roster, events, answers),
        }

    def _attendance(self, roster: list[dict]) -> dict:
        present = [
            {"id": row["student_id"], "name": row["name"], "roll_no": row["roll_no"]}
            for row in roster
            if row[PRESENT]
        ]
        return {"present": present, "count": len(present)}

    def _coverage(self, events: list[dict], session: dict) -> dict:
        """How much of the lesson was actually taught. The number a head of
        department asks for, and the only one on here that ranks anything."""
        taught = [_body(e) for e in events if e["name"] == LESSON_SEGMENT]
        total = max((int(p.get("total") or 0) for p in taught), default=0)

        if not total:
            pack = self.content.load(session["language"])
            lesson = pack.lessons.get(session["topic"])
            total = len(lesson.segments) if lesson else 0

        return {
            "taught": len({int(p["index"]) for p in taught if p.get("index") is not None}),
            "total": total,
            "segments": [p.get("segment_id", "") for p in taught],
        }

    def _questions(self, events: list[dict]) -> list[dict]:
        """What the class wanted to know. The most useful page in the report
        and the one nothing else in the system produces."""
        answers = {
            _body(e).get("question", ""): _body(e).get("answer", "")
            for e in events
            if e["name"] == QUESTION_ANSWERED
        }
        asked = []
        for event in events:
            if event["name"] != QUESTION_ASKED:
                continue
            body = _body(event)
            text = body.get("text", "")
            asked.append(
                {
                    "text": text,
                    "asked_by": body.get("student_name", ""),
                    "answered": answers.get(text, ""),
                    "at": event["at"],
                }
            )
        return asked

    def _quiz(self, roster: list[dict], events: list[dict], answers: list[dict]) -> dict:
        """Per student, in roll order.

        Roll order, not score order. A report sorted by result is a ranking
        whatever the column headings say.
        """
        posed = [_body(e) for e in events if e["name"] == QUIZ_POSED]
        questions = {p["question_id"]: p.get("text", "") for p in posed if p.get("question_id")}

        by_student: dict[str, list[dict]] = {}
        for row in answers:
            by_student.setdefault(row["student_id"], []).append(row)

        students = []
        for row in roster:
            given = by_student.get(row["student_id"], [])
            students.append(
                {
                    "id": row["student_id"],
                    "name": row["name"],
                    "roll_no": row["roll_no"],
                    "answered": len(given),
                    "correct": sum(1 for a in given if a[CORRECT]),
                    "unmarked": sum(1 for a in given if a[CORRECT] is NO_ANSWER),
                    "responses": [
                        {
                            "question": questions.get(a["question_ref"], a["question_ref"]),
                            "response": a["response"],
                            "correct": None if a[CORRECT] is None else bool(a[CORRECT]),
                        }
                        for a in given
                    ],
                }
            )

        return {"asked": len(questions), "students": students}


def _body(event: dict) -> dict:
    try:
        payload = json.loads(event["payload"])
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _minutes(session: dict) -> float:
    ended = session["ended_at"]
    if not ended:
        return 0.0
    return round((ended - session["started_at"]) / 60.0, 1)
