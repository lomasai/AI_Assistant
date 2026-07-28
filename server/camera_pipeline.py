"""Backend-owned camera drivers and latest-frame pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from server.config import CameraConfig
from server.interfaces import Frame


CameraState = Literal["disabled", "starting", "ready", "error", "stopped"]


class CameraPipelineError(Exception):
    """Raised when the backend camera pipeline cannot operate."""


class Picamera2UnavailableError(CameraPipelineError):
    """Raised when Picamera2 is configured but unavailable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_jpeg(frame_data: Any, quality: int) -> bytes:
    """Encode a frame-like object to JPEG bytes."""
    if isinstance(frame_data, (bytes, bytearray)):
        return bytes(frame_data)
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise CameraPipelineError("OpenCV is required for JPEG encoding.") from exc

    ok, encoded = cv2.imencode(".jpg", frame_data, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise CameraPipelineError("Failed to encode camera frame.")
    return encoded.tobytes()


class MockCameraDriver:
    """Mock camera driver that produces deterministic JPEG frames."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.started = False
        self.sequence = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def frames(self) -> AsyncIterator[Frame]:
        while self.started:
            self.sequence += 1
            yield self._make_frame(self.sequence)
            await asyncio.sleep(1 / max(1, self.config.preview_fps))

    def _make_frame(self, sequence: int) -> Frame:
        try:
            import cv2  # type: ignore
            import numpy as np

            image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            color = int((sequence * 17) % 255)
            image[:, :] = (30, color, 90)
            cv2.putText(
                image,
                f"MOCK {sequence}",
                (24, max(48, self.config.height // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            jpeg = encode_jpeg(image, self.config.jpeg_quality)
            data: Any = image
        except Exception:
            jpeg = _minimal_jpeg()
            data = None
        return Frame(
            data=data,
            width=self.config.width,
            height=self.config.height,
            timestamp_utc=utc_now(),
            source="mock",
            sequence=sequence,
            jpeg_bytes=jpeg,
        )


class BrowserCameraDriver:
    """No-op driver for the existing browser-owned camera workflow."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def frames(self) -> AsyncIterator[Frame]:
        if False:
            yield Frame(data=None, width=self.config.width, height=self.config.height, timestamp_utc=utc_now(), source="browser")
        return


class DisabledCameraDriver(BrowserCameraDriver):
    async def start(self) -> None:
        self.started = False


class Picamera2CameraDriver:
    """Raspberry Pi CSI camera driver using Picamera2 when installed."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.started = False
        self.sequence = 0
        self._picam2: Any | None = None

    async def start(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise Picamera2UnavailableError("Picamera2 is not installed or not available on this platform.") from exc

        self._picam2 = Picamera2()
        video_config = self._picam2.create_video_configuration(
            main={"size": (self.config.width, self.config.height), "format": "RGB888"}
        )
        self._picam2.configure(video_config)
        self._picam2.start()
        self.started = True
        await asyncio.sleep(min(self.config.warmup_timeout_seconds, 2.0))

    async def stop(self) -> None:
        self.started = False
        if self._picam2 is not None:
            picam2 = self._picam2
            self._picam2 = None
            await asyncio.to_thread(picam2.stop)

    async def frames(self) -> AsyncIterator[Frame]:
        if self._picam2 is None:
            await self.start()
        while self.started and self._picam2 is not None:
            self.sequence += 1
            frame_data = await asyncio.to_thread(self._picam2.capture_array)
            frame_data = self._transform(frame_data)
            jpeg = encode_jpeg(frame_data, self.config.jpeg_quality)
            height = int(getattr(frame_data, "shape", [self.config.height, self.config.width])[0])
            width = int(getattr(frame_data, "shape", [self.config.height, self.config.width])[1])
            yield Frame(
                data=frame_data,
                width=width,
                height=height,
                timestamp_utc=utc_now(),
                source="picamera2",
                sequence=self.sequence,
                jpeg_bytes=jpeg,
            )
            await asyncio.sleep(1 / max(1, self.config.preview_fps))

    def _transform(self, frame_data: Any) -> Any:
        if not (self.config.flip_horizontal or self.config.flip_vertical or self.config.rotation_degrees):
            return frame_data
        try:
            import cv2  # type: ignore
        except ImportError:
            return frame_data
        output = frame_data
        if self.config.rotation_degrees == 90:
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation_degrees == 180:
            output = cv2.rotate(output, cv2.ROTATE_180)
        elif self.config.rotation_degrees == 270:
            output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.config.flip_horizontal and self.config.flip_vertical:
            output = cv2.flip(output, -1)
        elif self.config.flip_horizontal:
            output = cv2.flip(output, 1)
        elif self.config.flip_vertical:
            output = cv2.flip(output, 0)
        return output


@dataclass(slots=True)
class CameraStatus:
    state: CameraState
    provider: str
    sequence: int
    width: int
    height: int
    timestamp_utc: str | None
    preview_clients: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "provider": self.provider,
            "sequence": self.sequence,
            "width": self.width,
            "height": self.height,
            "timestamp_utc": self.timestamp_utc,
            "preview_clients": self.preview_clients,
            "error": self.error,
        }


class FramePipeline:
    """Own one camera driver and expose only the latest frame."""

    def __init__(self, driver: Any, config: CameraConfig) -> None:
        self.driver = driver
        self.config = config
        self.state: CameraState = "disabled" if config.provider == "disabled" else "stopped"
        self._latest: Frame | None = None
        self._lock = asyncio.Lock()
        self._new_frame = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._preview_clients = 0
        self._error: str | None = None

    @property
    def latest_frame(self) -> Frame | None:
        return self._latest

    @property
    def preview_clients(self) -> int:
        return self._preview_clients

    async def start(self) -> None:
        if self.config.provider in {"browser", "disabled"}:
            self.state = "disabled" if self.config.provider == "disabled" else "stopped"
            return
        async with self._lock:
            if self._task and not self._task.done():
                return
            self.state = "starting"
            self._error = None
            try:
                await self.driver.start()
            except Picamera2UnavailableError:
                self.state = "error"
                self._error = "Picamera2 is unavailable. Install Raspberry Pi camera packages or use mock/browser mode."
                raise
            except Exception as exc:
                self.state = "error"
                self._error = "Camera failed to start."
                raise CameraPipelineError("Camera failed to start.") from exc
            self._task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        async with self._lock:
            task = self._task
            self._task = None
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self.driver.stop()
            self.state = "disabled" if self.config.provider == "disabled" else "stopped"
            async with self._new_frame:
                self._new_frame.notify_all()

    async def _capture_loop(self) -> None:
        try:
            async for frame in self.driver.frames():
                if frame.jpeg_bytes is None:
                    frame.jpeg_bytes = encode_jpeg(frame.data, self.config.jpeg_quality)
                self._latest = frame
                self.state = "ready"
                async with self._new_frame:
                    self._new_frame.notify_all()
        except asyncio.CancelledError:
            raise
        except Picamera2UnavailableError:
            self.state = "error"
            self._error = "Picamera2 is unavailable. Install Raspberry Pi camera packages or use mock/browser mode."
        except Exception:
            self.state = "error"
            self._error = "Camera stream failed."

    async def wait_for_frame(self, last_sequence: int = 0, timeout: float = 2.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        while True:
            frame = self._latest
            if frame is not None and frame.sequence > last_sequence:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return frame
            async with self._new_frame:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._new_frame.wait(), timeout=remaining)

    async def mjpeg_frames(self) -> AsyncIterator[bytes]:
        self._preview_clients += 1
        last_sequence = 0
        try:
            while self.state in {"starting", "ready", "stopped"}:
                frame = await self.wait_for_frame(last_sequence=last_sequence, timeout=2.0)
                if frame is None or frame.jpeg_bytes is None:
                    if self.state in {"error", "disabled"}:
                        break
                    continue
                last_sequence = frame.sequence
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"X-Frame-Sequence: {frame.sequence}\r\n\r\n".encode("ascii")
                    + frame.jpeg_bytes
                    + b"\r\n"
                )
        finally:
            self._preview_clients = max(0, self._preview_clients - 1)

    async def status_events(self) -> AsyncIterator[str]:
        last_sequence = -1
        while True:
            frame = self._latest
            sequence = frame.sequence if frame else 0
            if sequence != last_sequence:
                last_sequence = sequence
                yield self.sse_payload()
            await asyncio.sleep(max(0.2, 1 / max(1, self.config.analysis_fps)))

    def sse_payload(self) -> str:
        import json

        return f"event: camera_status\ndata: {json.dumps(self.status().as_dict(), ensure_ascii=True)}\n\n"

    def status(self) -> CameraStatus:
        frame = self._latest
        return CameraStatus(
            state=self.state,
            provider=self.config.provider,
            sequence=frame.sequence if frame else 0,
            width=frame.width if frame else self.config.width,
            height=frame.height if frame else self.config.height,
            timestamp_utc=frame.timestamp_utc if frame else None,
            preview_clients=self._preview_clients,
            error=self._error,
        )


def build_camera_driver(config: CameraConfig) -> Any:
    if config.provider == "picamera2":
        return Picamera2CameraDriver(config)
    if config.provider == "mock":
        return MockCameraDriver(config)
    if config.provider == "disabled":
        return DisabledCameraDriver(config)
    return BrowserCameraDriver(config)


def _minimal_jpeg() -> bytes:
    # 1x1 black JPEG.
    return bytes.fromhex(
        "ffd8ffe000104a46494600010101006000600000ffdb004300"
        "0302020302020303030304030304050805050404050a07070608"
        "0c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b10161011131415"
        "15150c0f171816141812141514ffdb0043010304040504050905"
        "0509140d0b0d1414141414141414141414141414141414141414"
        "1414141414141414141414141414141414141414141414141414"
        "141414141414ffc00011080001000103012200021101031101ff"
        "c4001400010000000000000000000000000000000000000000ff"
        "c4001410010000000000000000000000000000000000000000ff"
        "da000c03010002110311003f00d2cf20ffd9"
    )
