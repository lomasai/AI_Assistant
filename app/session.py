from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lomas_core.clock import Clock
from lomas_core.contracts import ROBOT_SAY, Utterance
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_store import TenantScope

from app.content import ContentPack, Lesson


@dataclass(slots=True)
class SessionContext:
    """Everything a step is allowed to touch.

    A step publishes events and reads this. It has no route to another step,
    no handle on the orchestrator and no agent to call, which is what makes
    any one of them safe to remove from flow.sequence.
    """

    session_id: str
    scope: TenantScope
    cfg: Config
    bus: EventBus
    clock: Clock
    language: str
    topic: str
    content: ContentPack
    lesson: Lesson
    repos: dict[str, Any] = field(default_factory=dict)
    roster: list[dict] = field(default_factory=list)
    present: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def say(self, text: str, student_name: str = "", reason: str = "") -> None:
        """Publish speech. The voice service is what actually makes a sound,
        so a step never touches an audio device."""
        if not text.strip():
            return
        self.bus.publish(
            ROBOT_SAY,
            Utterance(
                text=text.strip(),
                language=self.language,
                session_id=self.session_id,
                student_name=student_name,
                reason=reason,
            ),
        )

    def repo(self, name: str):
        return self.repos[name]
