from __future__ import annotations

import numpy as np

from lomas_core import logging as log
from lomas_core.errors import LomasError
from lomas_core.schema import SourceConfig
from lomas_vision.source import CAMERA_SOURCES, BaseSource
from lomas_vision.zoom import crop_rectangle

MIN_ZOOM = 1.0
MAX_ZOOM = 2.5

SETTLE_POLL = 0.1
AE_LOCKED = "AeLocked"
AUTO = 0  # a control left at zero is one the sensor decides for itself


@CAMERA_SOURCES.register("picamera2")
class PiCameraSource(BaseSource):
    """The Raspberry Pi camera via libcamera.

    The import is deliberately deferred: this module must import cleanly on a
    laptop so the package stays testable off-device, and only complain when
    someone actually tries to open the camera.
    """

    def __init__(self, spec: SourceConfig) -> None:
        super().__init__(spec)
        self._camera = None
        self._sensor_size: tuple[int, int] | None = None
        self.log = log.get("camera")

    def _start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise LomasError(
                "picamera2 is not installed. On Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2. On a laptop, use "
                "kind: usb or kind: mock instead."
            ) from exc

        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (self.spec.width, self.spec.height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()
        self._sensor_size = tuple(self._camera.camera_properties["PixelArraySize"])

        self._apply_controls()
        self._wait_for_exposure()
        self.set_zoom(self.zoom)

    def _apply_controls(self) -> None:
        """Everything libcamera will take. Absent keys are left alone rather
        than forced to a default, so a sensible sensor stays sensible."""
        wanted = self.spec.controls
        controls: dict = {
            "AeEnable": wanted.auto_exposure,
            "AwbEnable": wanted.auto_white_balance,
            "Brightness": wanted.brightness,
            "Contrast": wanted.contrast,
            "Saturation": wanted.saturation,
        }
        if wanted.auto_exposure:
            controls["ExposureValue"] = wanted.exposure_value
        if wanted.exposure_time_us > AUTO:
            controls["ExposureTime"] = wanted.exposure_time_us
        if wanted.analogue_gain > AUTO:
            controls["AnalogueGain"] = wanted.analogue_gain

        try:
            self._camera.set_controls(controls)
        except (RuntimeError, KeyError) as exc:
            # An older libcamera may not know one of these. A camera that
            # works with default colour beats one that refuses to open.
            self.log.warning("camera controls not applied: %s", exc)

    def _wait_for_exposure(self) -> None:
        """Give auto-exposure real frames to converge on, and stop as soon as
        it says it has. A bright room takes a moment; a dim one takes longer,
        and grabbing before either is why a good sensor looks dark."""
        import time

        deadline = time.monotonic() + self.spec.controls.warmup_seconds
        while time.monotonic() < deadline:
            try:
                if self._camera.capture_metadata().get(AE_LOCKED):
                    return
            except (RuntimeError, KeyError):
                return
            time.sleep(SETTLE_POLL)

    def set_zoom(self, factor: float) -> None:
        super().set_zoom(factor)
        if self._camera is None or self._sensor_size is None:
            return
        width, height = self._sensor_size
        self._camera.set_controls({"ScalerCrop": crop_rectangle(factor, width, height, MIN_ZOOM, MAX_ZOOM)})

    def _grab(self) -> np.ndarray | None:
        return self._camera.capture_array()

    def _stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
