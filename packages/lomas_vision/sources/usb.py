from __future__ import annotations

from lomas_core.schema import SourceConfig
from lomas_vision.source import CAMERA_SOURCES
from lomas_vision.sources.capture import OpenCvSource

DEFAULT_DEVICE = 0


@CAMERA_SOURCES.register("usb")
class UsbSource(OpenCvSource):
    def _target(self) -> int | str:
        device = self.spec.device
        if device is None:
            return DEFAULT_DEVICE
        return int(device) if str(device).isdigit() else device
