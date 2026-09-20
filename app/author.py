from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.schema import AuthorConfig, ContentConfig

from app.content import Lesson, Quiz, QuizQuestion, Segment, topic_key as slug

FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)
SEGMENT = "s"
QUESTION = "q"

# JSON's own punctuation, named so the mender reads as what it is.
OPENS = "{["
SHUTS = "}]"
ENDS_A_VALUE = '}]"'
QUOTE = '"'
BACKSLASH = "\\"


def as_json(text: str) -> dict:
    """What the model returned, as a dict.

    Models fence their JSON and sometimes say "here you go" first, so the
    outermost braces are taken rather than the whole reply. A reply that ran
    out of tokens mid-sentence is mended rather than thrown away: five
    segments of a lesson in front of a class beat an exception.
    """
    body = FENCE.sub("", text).strip()
    start = body.find("{")
    if start < 0:
        raise LomasError("the lesson writer did not return JSON")

    body = body[start:]
    end = body.rfind("}")
    if end > 0:
        try:
            return json.loads(body[: end + 1])
        except json.JSONDecodeError:
            pass

    mended = mend(body)
    if mended is not None:
        return mended
    raise LomasError("the lesson writer returned JSON that could not be read")


def mend(body: str) -> dict | None:
    """A truncated reply, closed off at the last thing that was whole.

    Walks back to each complete item and tries to shut the brackets that are
    still open. Returns None when there is nothing usable in it at all.
    """
    for cut in range(len(body), 0, -1):
        if body[cut - 1] not in ENDS_A_VALUE and not body[cut - 1].isdigit():
            continue
        candidate = body[:cut].rstrip().rstrip(",")
        try:
            return json.loads(candidate + closers(candidate))
        except json.JSONDecodeError:
            continue
    return None


def closers(body: str) -> str:
    """The brackets left open, in the order they have to be shut."""
    open_now: list[str] = []
    quoted = False
    escaped = False
    for char in body:
        if escaped:
            escaped = False
            continue
        if char == BACKSLASH:
            escaped = True
        elif char == QUOTE:
            quoted = not quoted
        elif quoted:
            continue
        elif char in OPENS:
            open_now.append(char)
        elif char in SHUTS and open_now:
            open_now.pop()
    return "".join(SHUTS[OPENS.index(char)] for char in reversed(open_now))


def clean_topic(said: str, cfg) -> str:
    """The subject, out of a sentence a child said.

    "My name is Akshay, so today we want to learn about machine learning, so
    let us go ahead and see" is a lesson on machine learning, and was sent to
    the writer whole.
    """
    topic = " ".join(said.split()).strip(" .,!?").lower()

    # The introduction first: a name is not a subject, and the cue takes the
    # name after it with it.
    for cue in sorted(cfg.topic_name_cues, key=len, reverse=True):
        if topic.startswith(cue + " "):
            topic = " ".join(topic[len(cue) + 1 :].split()[1:]).lstrip(" ,.")
            break

    for lead in sorted(cfg.topic_lead_ins, key=len, reverse=True):
        at = topic.find(lead + " ")
        if at >= 0:
            topic = topic[at + len(lead) + 1 :]
            break
    # Repeatedly: "machine learning so, let us go ahead and see" sheds the
    # invitation and then the "so" that was holding it on.
    trimming = True
    while trimming:
        trimming = False
        for tail in sorted(cfg.topic_tail_offs, key=len, reverse=True):
            if topic.endswith(" " + tail) or topic == tail:
                topic = topic[: -len(tail)].rstrip(" ,.")
                trimming = True
                break
    words = topic.strip(" .,!?").split()
    return " ".join(words[: cfg.topic_max_words])


class LessonWriter:
    """Writes a lesson about whatever a child asks for.

    The content packs are the reviewed material and stay the first choice; a
    topic nobody wrote a pack for used to be an error, which made the robot a
    demo of one lesson. What comes back here is marked as written on the
    spot, so a report never presents it as approved content.
    """

    def __init__(self, cfg: ContentConfig, prompts: Any, llm: Any) -> None:
        self.cfg = cfg
        self.settings: AuthorConfig = cfg.author
        self.prompts = prompts
        self.llm = llm
        self.log = log.get("author")

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def cached(self, topic: str, language: str) -> tuple[Lesson, Quiz | None] | None:
        path = self._path(topic, language)
        if not path.exists():
            return None
        try:
            return self._build(json.loads(path.read_text(encoding="utf-8")), topic, language)
        except (LomasError, OSError, json.JSONDecodeError, KeyError) as exc:
            self.log.debug("ignoring cached lesson %s: %s", path, exc)
            return None

    def write(self, topic: str, language: str) -> tuple[Lesson, Quiz | None]:
        """One call for the lesson and its questions. Two would be twice the
        wait in front of a class, and the questions have to be about the
        lesson that was actually written rather than the topic in general."""
        messages = self.prompts.messages(
            self.settings.prompt,
            language,
            topic=topic,
            grade=self.cfg.grade,
            subject=self.cfg.subject,
            vocabulary_level=self.cfg.vocabulary_level,
            language=language,
            segments=self.settings.segments,
            questions=self.settings.questions,
        )
        options = {"max_tokens": self.settings.max_tokens or None}
        if self.settings.json_mode:
            options["response_format"] = {"type": "json_object"}
        written = self.llm.complete(messages, language=language, **options)
        if not written:
            raise LomasError(f"nothing came back for '{topic}'")

        body = as_json(written.text)
        lesson, quiz = self._build(body, topic, language)
        self._remember(body, topic, language)
        self.log.info("wrote a lesson on %s: %d parts", topic, len(lesson))
        return lesson, quiz

    def _build(self, body: dict, topic: str, language: str) -> tuple[Lesson, Quiz | None]:
        parts = body.get("segments") or []
        if not parts:
            raise LomasError(f"the lesson on '{topic}' came back with no parts")

        lesson_id = slug(topic)
        lesson = Lesson(
            id=lesson_id,
            title=body.get("title") or topic,
            language=language,
            segments=tuple(
                Segment(id=f"{SEGMENT}{n}", say=str(part.get("say", "")).strip(),
                        display=str(part.get("display", "")).strip())
                for n, part in enumerate(parts[: self.settings.segments], start=1)
                if str(part.get("say", "")).strip()
            ),
            written=True,
        )
        if not len(lesson):
            raise LomasError(f"the lesson on '{topic}' came back empty")

        asked = body.get("questions") or []
        quiz = Quiz(
            id=f"{lesson_id}-quiz",
            lesson=lesson_id,
            language=language,
            questions=tuple(
                QuizQuestion(
                    id=f"{QUESTION}{n}",
                    ask=str(question.get("ask", "")).strip(),
                    options=tuple(str(option) for option in question.get("options", ())),
                    answer=question.get("answer") if isinstance(question.get("answer"), int) else None,
                )
                for n, question in enumerate(asked[: self.settings.questions], start=1)
                if str(question.get("ask", "")).strip()
            ),
        ) if asked else None

        return lesson, quiz

    def _remember(self, body: dict, topic: str, language: str) -> None:
        """Kept on disk, so the same topic tomorrow costs nothing and works
        with the internet down. Not in content/: those are the reviewed
        lessons and this is not one."""
        if not self.settings.cache_dir:
            return
        path = self._path(topic, language)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self.log.debug("could not keep the lesson: %s", exc)

    def _path(self, topic: str, language: str) -> Path:
        return Path(self.settings.cache_dir) / language / f"{slug(topic)}.json"
