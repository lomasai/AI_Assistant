from __future__ import annotations

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import (
    CARD_SEEN,
    QUIZ_ANSWERED,
    QUIZ_POSED,
    CardSeen,
    QuizAnswered,
    QuizPosed,
)
from lomas_core.events import EventBus
from lomas_core.schema import Config

MILLISECONDS = 1000.0
NOBODY = ""


class Answering:
    """A whole class answering a question by holding up a card.

    One microphone in a room of forty is the problem this is for. Spoken
    answers cost a turn each, arrive one at a time, and come back through a
    transcriber that mishears a child at three metres. Cards arrive in one
    frame, already attributed, with nothing to transcribe - so every child
    answers every question instead of the one who was asked.

    Which edge means which answer is config, and so is whether this is on at
    all: a school with no cards printed loses nothing.
    """

    def __init__(self, cfg: Config, bus: EventBus, clock: Clock) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.log = log.get("answering")

        self.recorded = 0
        self._open: QuizPosed | None = None
        self._asked_at = 0.0
        self._answered: set[str] = set()

        bus.subscribe(QUIZ_POSED, self._on_posed)
        bus.subscribe(CARD_SEEN, self._on_card)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.signs.enabled and self.cfg.signs.cards.answers)

    def _on_posed(self, _event: str, posed: QuizPosed) -> None:
        # A new question closes the last one. A card still held up from the
        # previous answer is not an answer to this.
        self._open = posed
        self._asked_at = self.clock.now()
        self._answered.clear()

    def _on_card(self, _event: str, seen: CardSeen) -> None:
        if not self.enabled or self._open is None or not seen.student_id:
            return
        if seen.student_id in self._answered:
            # A child may change their mind while the question is open, but
            # a card held steady must not count forty times.
            return
        if not seen.answer:
            return

        self._answered.add(seen.student_id)
        self.recorded += 1
        chosen = self._index(seen.answer)
        self.bus.publish(QUIZ_ANSWERED, QuizAnswered(
            session_id=self._open.session_id,
            question_id=self._open.question_id,
            student_id=seen.student_id,
            response=self._said(seen.answer, chosen),
            correct=None,
            latency_ms=int((self.clock.now() - self._asked_at) * MILLISECONDS),
        ))
        self.log.info("%s answered %s with a card", seen.student_name or seen.student_id,
                      seen.answer)

    def _index(self, answer: str) -> int:
        answers = self.cfg.signs.cards.answers
        return answers.index(answer) if answer in answers else -1

    def _said(self, answer: str, chosen: int) -> str:
        """The option itself where the question has options, and the letter
        where it does not - so a marker reads an answer and not a letter."""
        options = self._open.options if self._open else ()
        if 0 <= chosen < len(options):
            return str(options[chosen])
        return answer
