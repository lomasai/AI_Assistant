from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lomas_core import logging as log
from lomas_core.errors import LomasError

from app.pipeline import source_for
from app.web import api as api_module
from app.web import teacher as teacher_module
from app.web import ws as ws_module
from app.web.stream import CONTENT_TYPE, mjpeg

UI = Path(__file__).parent / "ui"
FACE = "face"
BOARD = "board"
INDEX = "index.html"
API_PREFIX = "/api"
NO_CAMERA = b""
BAD_REQUEST = 400


def create_app(system) -> FastAPI:
    """Every surface is a subscriber and only a subscriber.

    The face, the board and whatever comes after read the same event stream
    the agents do. None of them can reach a step or the orchestrator except
    through the handful of controls in api.py.
    """
    hub = ws_module.EventHub(system.bus, system.cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Until the loop exists there is nobody to forward events to, so the
        # hub simply drops them.
        hub.bind(asyncio.get_running_loop())
        yield

    app = FastAPI(title="LomasAI", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.hub = hub
    app.state.system = system

    app.include_router(api_module.router(system), prefix=API_PREFIX)
    if system.cfg.teacher.enabled:
        app.include_router(teacher_module.router(system), prefix=API_PREFIX)

    @app.exception_handler(LomasError)
    async def refused(_request: Request, exc: LomasError) -> JSONResponse:
        """Everything this system raises deliberately is a bad request, not a
        crash. A refused enrolment must read as a refusal in the browser."""
        return JSONResponse(status_code=BAD_REQUEST, content={"error": str(exc)})

    @app.websocket("/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await ws_module.pump(websocket, hub)
        except WebSocketDisconnect:
            pass

    @app.get("/camera.mjpeg")
    def camera() -> StreamingResponse:
        frames = system.frames
        if frames is None:
            return StreamingResponse(iter([NO_CAMERA]), media_type=CONTENT_TYPE)
        source = system.cfg.web.mjpeg_source or source_for(system.cfg)
        return StreamingResponse(mjpeg(frames, system.cfg, source), media_type=CONTENT_TYPE)

    surfaces = [s for s in system.cfg.web.surfaces if (UI / s).is_dir()]
    for surface in surfaces:
        app.mount(f"/{surface}", StaticFiles(directory=UI / surface, html=True), name=surface)

    @app.get("/")
    def home() -> RedirectResponse:
        return RedirectResponse(f"/{surfaces[0]}/" if surfaces else API_PREFIX)

    return app


class WebServer:
    """uvicorn on its own thread, so the class does not wait on the browser.

    The flow is the product; the screens are a view of it. If the server were
    the main loop, a hung request would be a hung lesson.
    """

    def __init__(self, system) -> None:
        self.system = system
        self.cfg = system.cfg.web
        self.log = log.get("web")
        self.app = create_app(system)
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = self.cfg.host
        shown = "localhost" if host in ("0.0.0.0", "::") else host
        return f"http://{shown}:{self.cfg.port}/"

    def start(self) -> None:
        if self._thread is not None or not self.cfg.enabled:
            return

        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.cfg.host,
            port=self.cfg.port,
            log_level=self.system.cfg.runtime.log_level.lower(),
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="web", daemon=True)
        self._thread.start()
        self.log.info("surfaces on %s (%s)", self.url, ", ".join(self.cfg.surfaces))

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.shutdown_seconds)
            self._thread = None
