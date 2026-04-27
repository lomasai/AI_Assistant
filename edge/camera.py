"""Raspberry Pi camera module.

Responsibilities:
- Capture frames from camera device
- Send frames to processing functions
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable


class CameraError(Exception):
    """Raised when camera operations fail."""


ProcessorFn = Callable[[Any], Any | Awaitable[Any]]


@dataclass(slots=True)
class CameraConfig:
    """Configuration for edge camera capture."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 24


class EdgeCamera:
    """Camera capture wrapper with frame processing pipeline."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: Any = None
        self._opened = False

    def open(self) -> None:
        """Open camera device with configured stream settings."""
        if self._opened:
            return

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraError("OpenCV is not installed. Install `opencv-python`.") from exc

        capture = cv2.VideoCapture(self.config.device_index)
        if not capture.isOpened():
            raise CameraError(f"Unable to open camera device index {self.config.device_index}.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._capture = capture
        self._opened = True

    def close(self) -> None:
        """Release camera device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._opened = False

    def capture_frame(self) -> Any:
        """Capture one frame from camera and return image array."""
        if not self._opened:
            self.open()

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture frame from camera.")
        return frame

    def capture_frame_to_file(self, output_path: str | Path) -> Path:
        """Capture one frame and store it as an image file."""
        frame = self.capture_frame()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraError("OpenCV is not installed. Install `opencv-python`.") from exc

        written = cv2.imwrite(str(target), frame)
        if not written:
            raise CameraError(f"Failed to write frame to: {target}")
        return target

    async def stream_frames(
        self,
        max_frames: int | None = None,
        interval_seconds: float = 0.0,
    ):
        """Yield frames asynchronously from camera."""
        frame_count = 0
        try:
            while max_frames is None or frame_count < max_frames:
                frame = await asyncio.to_thread(self.capture_frame)
                frame_count += 1
                yield frame
                if interval_seconds > 0:
                    await asyncio.sleep(interval_seconds)
        finally:
            self.close()

    async def process_frames(
        self,
        processors: Iterable[ProcessorFn],
        max_frames: int | None = None,
        interval_seconds: float = 0.0,
        stop_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """Capture frames and send each frame to processing functions."""
        results: list[dict[str, Any]] = []
        processor_list = list(processors)
        if not processor_list:
            raise CameraError("At least one frame processor is required.")

        frame_index = 0
        async for frame in self.stream_frames(max_frames=max_frames, interval_seconds=interval_seconds):
            frame_index += 1
            frame_result: dict[str, Any] = {
                "frame_index": frame_index,
                "timestamp_unix": time.time(),
                "processor_outputs": [],
            }

            for processor in processor_list:
                try:
                    output = processor(frame)
                    if inspect.isawaitable(output):
                        output = await output
                    frame_result["processor_outputs"].append(
                        {
                            "processor": getattr(processor, "__name__", processor.__class__.__name__),
                            "output": output,
                            "error": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    frame_result["processor_outputs"].append(
                        {
                            "processor": getattr(processor, "__name__", processor.__class__.__name__),
                            "output": None,
                            "error": str(exc),
                        }
                    )
                    if stop_on_error:
                        results.append(frame_result)
                        raise CameraError(
                            f"Processor '{getattr(processor, '__name__', processor.__class__.__name__)}' failed: {exc}"
                        ) from exc

            results.append(frame_result)

        return results


edge_camera = EdgeCamera()
