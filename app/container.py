from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock, RealClock
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_face import DETECTORS, EMBEDDERS
from lomas_llm import PROVIDERS, PromptLibrary, Router
from lomas_speech import STT_ENGINES, TTS_ENGINES, WAKE_WORDS, DuplexGate, InputSet
from lomas_store import (
    STORES,
    AnswerRepo,
    ClassRepo,
    ConsentRepo,
    EmbeddingRepo,
    EventRepo,
    OrgRepo,
    SchoolRepo,
    SessionRepo,
    StudentRepo,
    migrate,
)
from lomas_vision import FrameBus, build_sources

from app.content import ContentLibrary
from app.flow.machine import Machine
from app.flow.step import STEPS
from app.orchestrator import Orchestrator
from app.voice import Voice

from app.flow import steps as _steps  # noqa: F401

STEPS.discover("app.flow.steps")


@dataclass(slots=True)
class System:
    """Everything, wired. Handed to run.py and to the tests."""

    cfg: Config
    bus: EventBus
    clock: Clock
    store: Any
    repos: dict[str, Any]
    prompts: PromptLibrary
    llm: Any
    router: Router
    tts: Any
    stt: Any
    wake: Any
    voice: Voice
    content: ContentLibrary
    orchestrator: Orchestrator
    frames: FrameBus | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        if self.frames is not None:
            self.frames.stop()
        self.voice.stop()
        self.store.close()


def build(cfg: Config, clock: Clock | None = None, bus: EventBus | None = None) -> System:
    """The only place in the repository that names concrete classes.

    Everything above this reads interfaces; everything below registers itself
    under a config string. Swapping a provider, a camera or a speech engine
    happens here and nowhere else.
    """
    clock = clock or RealClock()
    bus = bus or _bus_for(cfg)
    logger = log.get("container")

    store = STORES.create(cfg.storage.backend, cfg.storage.path, cfg.storage.busy_timeout_ms)
    migrate(store, clock.now())
    repos = _repos(store)

    prompts = PromptLibrary(cfg.llm.prompts_path, cfg.llm.fallback_language)
    llm = PROVIDERS.create(cfg.llm.provider, cfg.llm)
    router = Router(cfg.llm.router)

    tts = TTS_ENGINES.create(cfg.speech.tts.engine, cfg.speech.tts)
    stt = STT_ENGINES.create(cfg.speech.stt.engine, cfg.speech.stt)
    wake = WAKE_WORDS.create(cfg.speech.wake.engine, cfg.speech.wake)
    gate = DuplexGate(cfg.speech.audio, clock)
    voice = Voice(tts, gate, bus)

    content = ContentLibrary(cfg.content)
    steps = [STEPS.create(name, cfg) for name in cfg.flow.sequence]
    machine = Machine(steps, cfg.flow, bus, clock)

    orchestrator = Orchestrator(
        cfg=cfg, bus=bus, clock=clock, machine=machine, repos=repos,
        prompts=prompts, llm=llm, content=content,
    )

    logger.debug(
        "built: store=%s llm=%s tts=%s stt=%s wake=%s steps=%s",
        cfg.storage.backend, cfg.llm.provider, cfg.speech.tts.engine,
        cfg.speech.stt.engine, cfg.speech.wake.engine, ",".join(cfg.flow.sequence),
    )

    return System(
        cfg=cfg, bus=bus, clock=clock, store=store, repos=repos, prompts=prompts,
        llm=llm, router=router, tts=tts, stt=stt, wake=wake, voice=voice,
        content=content, orchestrator=orchestrator,
        extras={"gate": gate, "machine": machine, "inputs": InputSet(cfg.speech.audio)},
    )


def build_vision(cfg: Config, clock: Clock) -> FrameBus:
    """Kept separate because the flow does not need a camera to run, and P8
    is what joins frames to faces."""
    return FrameBus(
        build_sources(cfg.sources),
        buffer_size=cfg.vision.buffer_size,
        clock=clock,
        read_timeout_ms=cfg.vision.read_timeout_ms,
    )


def _bus_for(cfg: Config) -> EventBus:
    logger = log.get("events")

    def on_error(event: str, exc: BaseException) -> None:
        logger.error("handler failed on %s: %s", event, exc, exc_info=exc)

    return EventBus(
        replay_size=cfg.runtime.event_replay_size,
        on_error=None if cfg.runtime.raise_on_handler_error else on_error,
    )


def _repos(store) -> dict[str, Any]:
    return {
        "org": OrgRepo(store),
        "school": SchoolRepo(store),
        "class": ClassRepo(store),
        "student": StudentRepo(store),
        "consent": ConsentRepo(store),
        "embedding": EmbeddingRepo(store),
        "session": SessionRepo(store),
        "answer": AnswerRepo(store),
        "event": EventRepo(store),
    }


def available() -> dict[str, list[str]]:
    """What can be selected in config right now. Used by the debug view and
    by the error message when someone names something that does not exist."""
    return {
        "storage": STORES.keys(),
        "llm": PROVIDERS.keys(),
        "tts": TTS_ENGINES.keys(),
        "stt": STT_ENGINES.keys(),
        "wake": WAKE_WORDS.keys(),
        "detector": DETECTORS.keys(),
        "embedder": EMBEDDERS.keys(),
        "steps": STEPS.keys(),
    }
