from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock, RealClock
from lomas_core.contracts import SESSION_CLOSED, SESSION_OPENED
from lomas_core.events import EventBus
from lomas_core.schema import AgentConfig, Config
from lomas_face import (
    DETECTORS,
    EMBEDDERS,
    AttentionMonitor,
    IdentityMatcher,
    Tracker,
)
from lomas_llm import PROVIDERS, PromptLibrary, Router
from lomas_speech import STT_ENGINES, TTS_ENGINES, WAKE_WORDS, DuplexGate, InputSet
from lomas_store import (
    STORES,
    TenantScope,
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

from app.agents.base import AGENTS, Agent, AgentDeps, AgentRunner
from app.agents.safety import Safety
from app.content import ContentLibrary
from app.enrolment import EnrolmentService
from app.context.assembler import ContextAssembler
from app.context.mcp_server import ContextServer
from app.flow.machine import Machine
from app.flow.step import STEPS
from app.orchestrator import Orchestrator
from app.report import ReportBuilder
from app.pipeline import VisionPipeline, vectors_by_student
from app.web.server import WebServer
from app.voice import Voice, allow_everything

from app import agents as _agents  # noqa: F401
from app.flow import steps as _steps  # noqa: F401

STEPS.discover("app.flow.steps")
AGENTS.discover("app.agents")


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
    agents: AgentRunner | None = None
    mcp: ContextServer | None = None
    vision: VisionPipeline | None = None
    enrolment: EnrolmentService | None = None
    report: ReportBuilder | None = None
    web: WebServer | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> FrameBus | None:
        return self.vision.frames if self.vision else None

    def close(self) -> None:
        if self.web is not None:
            self.web.stop()
        # Signal the pipeline first, then close the bus that wakes it. The
        # other order leaves the vision thread parked on an idle camera.
        if self.vision is not None:
            self.vision.stop()
            self.vision.frames.stop()
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

    content = ContentLibrary(cfg.content)
    assembler = ContextAssembler(cfg, repos, content)
    built = _agents_from(cfg, bus, clock, repos, prompts, llm)
    runner = AgentRunner(built, assembler, bus, clock, cfg) if built else None

    # The filter is asked before a sound is made, so it is an argument to the
    # voice rather than another subscriber racing it.
    guard = next((a.approve for a in built if isinstance(a, Safety)), allow_everything)
    voice = Voice(tts, gate, bus, guard=guard)
    steps = [STEPS.create(name, cfg) for name in cfg.flow.sequence]
    machine = Machine(steps, cfg.flow, bus, clock)

    orchestrator = Orchestrator(
        cfg=cfg, bus=bus, clock=clock, machine=machine, repos=repos,
        prompts=prompts, llm=llm, content=content,
    )

    vision = build_vision(cfg, bus, clock, repos)
    report = ReportBuilder(cfg, repos, content)
    enrolment = _enrolment(cfg, bus, clock, vision, repos)

    logger.debug(
        "built: store=%s llm=%s tts=%s stt=%s wake=%s steps=%s agents=%s",
        cfg.storage.backend, cfg.llm.provider, cfg.speech.tts.engine,
        cfg.speech.stt.engine, cfg.speech.wake.engine, ",".join(cfg.flow.sequence),
        ",".join(runner.names()) if runner else "none",
    )

    system = System(
        cfg=cfg, bus=bus, clock=clock, store=store, repos=repos, prompts=prompts,
        llm=llm, router=router, tts=tts, stt=stt, wake=wake, voice=voice,
        content=content, orchestrator=orchestrator, vision=vision,
        agents=runner, mcp=ContextServer(assembler),
        enrolment=enrolment, report=report,
        extras={"gate": gate, "machine": machine, "inputs": InputSet(cfg.speech.audio)},
    )

    # Last, because every surface is a view of the finished system. It is
    # built but not started: nothing listens until someone asks it to.
    if cfg.web.enabled:
        system.web = WebServer(system)
    return system


def build_vision(
    cfg: Config, bus: EventBus, clock: Clock, repos: dict[str, Any]
) -> VisionPipeline | None:
    """Cameras, detector, tracker, matcher and attention, assembled.

    Returns None when vision is switched off, and the class still runs: that
    is rule four, and it is the difference between a demo and a product.
    """
    if not (cfg.vision.pipeline.enabled and cfg.face.enabled):
        return None

    frames = FrameBus(
        build_sources(cfg.sources),
        buffer_size=cfg.vision.buffer_size,
        clock=clock,
        read_timeout_ms=cfg.vision.read_timeout_ms,
    )
    pipeline = VisionPipeline(
        cfg=cfg,
        bus=bus,
        clock=clock,
        frames=frames,
        detector=DETECTORS.create(cfg.face.detector, cfg.face),
        tracker=Tracker(cfg.face),
        matcher=IdentityMatcher(EMBEDDERS.create(cfg.face.embedder, cfg.face), cfg.face),
        attention=AttentionMonitor(cfg.attention),
    )
    _follow_the_session(pipeline, bus, repos)
    return pipeline


def _enrolment(
    cfg: Config, bus: EventBus, clock: Clock, vision: VisionPipeline | None, repos: dict[str, Any]
) -> EnrolmentService | None:
    """Its own detector and its own embedder.

    Borrowing the pipeline's would mean two threads sharing one model with
    one input size, which is a fault you only meet on the Pi and only under
    load - so it is not worth the object it saves.
    """
    if not cfg.teacher.enabled:
        return None
    return EnrolmentService(
        cfg=cfg,
        bus=bus,
        clock=clock,
        frames=vision.frames if vision else None,
        detector=DETECTORS.create(cfg.face.detector, cfg.face),
        embedder=EMBEDDERS.create(cfg.face.embedder, cfg.face),
        repos=repos,
    )


def _follow_the_session(pipeline: VisionPipeline, bus: EventBus, repos: dict[str, Any]) -> None:
    """The camera starts when a class opens, not when the process does.

    Through events, so the orchestrator keeps its promise of knowing nothing
    about cameras and vision can be removed without it noticing.
    """

    def on_open(_event: str, opened) -> None:
        scope = TenantScope(opened.org_id, opened.school_id, opened.class_id)
        pipeline.load(vectors_by_student(repos["embedding"].all_for_class(scope)))
        pipeline.start()

    def on_close(_event: str, _closed) -> None:
        pipeline.attention.reset()  # nudge budgets are per session

    bus.subscribe(SESSION_OPENED, on_open)
    bus.subscribe(SESSION_CLOSED, on_close)


def _bus_for(cfg: Config) -> EventBus:
    logger = log.get("events")

    def on_error(event: str, exc: BaseException) -> None:
        logger.error("handler failed on %s: %s", event, exc, exc_info=exc)

    return EventBus(
        replay_size=cfg.runtime.event_replay_size,
        on_error=None if cfg.runtime.raise_on_handler_error else on_error,
    )


def _agents_from(
    cfg: Config,
    bus: EventBus,
    clock: Clock,
    repos: dict[str, Any],
    prompts: PromptLibrary,
    shared: Any,
) -> list[Agent]:
    """One object per name in `agents.enabled`. Drop a name and that agent
    is gone; nothing else in the system notices."""
    built: list[Agent] = []
    for name in cfg.agents.enabled:
        settings = cfg.agents.settings.get(name) or AgentConfig(prompt=name)
        deps = AgentDeps(
            bus=bus,
            clock=clock,
            prompts=prompts,
            llm=_provider_for(cfg, settings, shared),
            repos=repos,
        )
        built.append(AGENTS.create(name, cfg, settings, deps))
    return built


def _provider_for(cfg: Config, settings: AgentConfig, shared: Any) -> Any:
    """An empty provider inherits llm.*, so pinning the safety filter to a
    small fast model leaves the tutor on the one that can teach."""
    if not settings.provider:
        return shared
    overrides = {"provider": settings.provider}
    if settings.model:
        overrides["model"] = settings.model
    return PROVIDERS.create(settings.provider, cfg.llm.model_copy(update=overrides))


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
        "agents": AGENTS.keys(),
    }
