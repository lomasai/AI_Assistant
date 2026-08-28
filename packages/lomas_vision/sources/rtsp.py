from __future__ import annotations

from lomas_core.errors import LomasError
from lomas_vision.source import CAMERA_SOURCES
from lomas_vision.sources.capture import OpenCvSource


@CAMERA_SOURCES.register("rtsp")
class RtspSource(OpenCvSource):
    """Network camera. This is the seam for classroom CCTV.

    Untested against real school hardware - RTSP in the field brings
    reconnection, latency and codec problems that this does not yet handle.
    Treat it as a starting point, not a finished backend.
    """

    def _target(self) -> str:
        if not self.spec.url:
            raise LomasError(f"source '{self.source_id}': rtsp kind needs a url")
        return self.spec.url

    def _configure(self) -> None:
        return  # the stream dictates its own format
