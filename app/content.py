from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lomas_core.errors import LomasError
from lomas_core.schema import ContentConfig

# Keys in the lesson and quiz JSON. Naming them keeps the shape of the file
# format in one place.
SEGMENTS_KEY = "segments"
QUESTIONS_KEY = "questions"

NOT_A_WORD = re.compile(r"[^a-z0-9]+")


def topic_key(topic: str) -> str:
    """A topic as a lesson id. "Solar System!", "solar system" and
    "solar-system" are one lesson, because one of them was typed by a
    teacher, one was said out loud, and one is the file on disk."""
    return NOT_A_WORD.sub("-", topic.strip().lower()).strip("-")


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    say: str
    display: str = ""


@dataclass(frozen=True, slots=True)
class Lesson:
    id: str
    title: str
    language: str
    segments: tuple[Segment, ...] = ()
    # Written on the spot for a topic a child asked for, rather than taken
    # from a reviewed pack. The report says so; a school has to be able to
    # tell the two apart.
    written: bool = False

    def __len__(self) -> int:
        return len(self.segments)


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    id: str
    ask: str
    options: tuple[str, ...] = ()
    answer: int | None = None

    def is_correct(self, response: str) -> bool | None:
        if self.answer is None or not self.options:
            return None
        expected = self.options[self.answer].strip().lower()
        given = response.strip().lower()
        return given == expected or given == str(self.answer + 1)


@dataclass(frozen=True, slots=True)
class Quiz:
    id: str
    lesson: str
    language: str
    questions: tuple[QuizQuestion, ...] = ()


@dataclass(slots=True)
class ContentPack:
    lessons: dict[str, Lesson] = field(default_factory=dict)
    quizzes: dict[str, Quiz] = field(default_factory=dict)

    def lesson_for(self, topic: str) -> Lesson:
        if topic in self.lessons:
            return self.lessons[topic]

        wanted = topic_key(topic)
        for lesson in self.lessons.values():
            if wanted in (topic_key(lesson.id), topic_key(lesson.title)):
                return lesson

        known = ", ".join(sorted(self.lessons)) or "none"
        raise LomasError(f"no lesson '{topic}'. Available: {known}")

    def add(self, lesson: Lesson, quiz: Quiz | None = None) -> Lesson:
        """A lesson written during the class, kept for the rest of it: the
        quiz step looks its questions up by lesson id."""
        self.lessons[lesson.id] = lesson
        if quiz is not None:
            self.quizzes[quiz.id] = quiz
        return lesson

    def quiz_for(self, lesson_id: str) -> Quiz | None:
        for quiz in self.quizzes.values():
            if quiz.lesson == lesson_id:
                return quiz
        return None


class ContentLibrary:
    """Lessons and quizzes as JSON under content/<language>/<grade>/<subject>.

    Grade, subject and language are config, so the same engine runs a Grade 4
    Hindi storytelling session by changing three keys.
    """

    def __init__(self, cfg: ContentConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.pack_path)
        # Lessons written during this run, by language. Every load is a fresh
        # read of the folder, and an agent asked for the lesson's text after
        # the session had already started teaching it.
        self._written: dict[str, list[tuple[Lesson, Quiz | None]]] = {}

    def remember(self, lesson: Lesson, quiz: Quiz | None, language: str | None = None) -> Lesson:
        self._written.setdefault(language or self.cfg.language, []).append((lesson, quiz))
        return lesson

    def folder(self, language: str | None = None) -> Path:
        return self.root / (language or self.cfg.language) / str(self.cfg.grade) / self.cfg.subject

    def load(self, language: str | None = None) -> ContentPack:
        folder = self.folder(language)
        if not folder.is_dir():
            raise LomasError(
                f"no content at {folder}. Check content.language, content.grade "
                "and content.subject."
            )

        pack = ContentPack()
        for path in sorted(folder.glob("*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            if SEGMENTS_KEY in body:
                pack.lessons[body["id"]] = _lesson(body)
            elif QUESTIONS_KEY in body:
                pack.quizzes[body["id"]] = _quiz(body)

        if not pack.lessons:
            raise LomasError(f"{folder} holds no lessons")

        for lesson, quiz in self._written.get(language or self.cfg.language, []):
            pack.add(lesson, quiz)
        return pack


def _lesson(body: dict) -> Lesson:
    return Lesson(
        id=body["id"],
        title=body.get("title", body["id"]),
        language=body.get("language", ""),
        segments=tuple(
            Segment(id=s["id"], say=s["say"], display=s.get("display", ""))
            for s in body["segments"]
        ),
    )


def _quiz(body: dict) -> Quiz:
    return Quiz(
        id=body["id"],
        lesson=body.get("lesson", ""),
        language=body.get("language", ""),
        questions=tuple(
            QuizQuestion(
                id=q["id"],
                ask=q["ask"],
                options=tuple(q.get("options", ())),
                answer=q.get("answer"),
            )
            for q in body["questions"]
        ),
    )
