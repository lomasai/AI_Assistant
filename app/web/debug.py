from __future__ import annotations

from fastapi import APIRouter

from app import container

ENABLED = "enabled"


def router(system) -> APIRouter:
    """Read-only, all of it.

    There is no endpoint here that changes anything. Diagnostics are
    subscribers, never participants, and the moment one of these could poke
    the flow it would start being used to poke the flow.
    """
    api = APIRouter()

    @api.get("/metrics")
    def metrics() -> dict:
        return system.metrics.snapshot(system)

    @api.get("/plugins")
    def plugins() -> dict:
        """What config could select, next to what it did. The answer to
        'why is it using that' without reading any code."""
        chosen = {
            "storage": system.cfg.storage.backend,
            "llm": system.cfg.llm.provider,
            "tts": system.cfg.speech.tts.engine,
            "stt": system.cfg.speech.stt.engine,
            "wake": system.cfg.speech.wake.engine,
            "detector": system.cfg.face.detector,
            "embedder": system.cfg.face.embedder,
            "steps": system.cfg.flow.sequence,
            "agents": system.agents.names() if system.agents else [],
        }
        return {"available": container.available(), "chosen": chosen}

    @api.get("/config")
    def config() -> dict:
        """The resolved config, after every layer. Half of debugging this
        system is finding out which file won."""
        return system.cfg.model_dump()

    @api.get("/events")
    def events() -> dict:
        snapshot = system.metrics.snapshot(system)["events"]
        return {"counts": snapshot["counts"], "recent": snapshot["recent"]}

    return api
