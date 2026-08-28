from __future__ import annotations

import time
from typing import Iterator

from lomas_core import logging as log
from lomas_core.schema import Config
from lomas_vision import Frame, FrameBus

BOUNDARY = "lomasframe"
CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
JPEG = ".jpg"
HEADER = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: ".encode()
BREAK = b"\r\n\r\n"
END = b"\r\n"


def encode(frame: Frame, quality: int) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(JPEG, frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else b""


def mjpeg(frames: FrameBus, cfg: Config, source_id: str) -> Iterator[bytes]:
    """The camera, as it is, with nothing drawn on it.

    Face boxes are HTML positioned over the video element. Painting them here
    would mean decoding, drawing and re-encoding every frame on the Pi's CPU,
    which is the most expensive mistake available in this file - and it would
    also tie the overlay's frame rate to the video's.
    """
    logger = log.get("web")
    web = cfg.web
    interval = 1.0 / web.mjpeg_fps
    last_seq = 0
    sent = 0

    try:
        while True:
            started = time.monotonic()
            frame = frames.latest(source_id)

            if frame is not None and frame.seq != last_seq:
                last_seq = frame.seq
                body = encode(frame, web.mjpeg_quality)
                if body:
                    yield HEADER + str(len(body)).encode() + BREAK + body + END
                    sent += 1

            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except GeneratorExit:
        logger.debug("stream closed after %s frames", sent)
        raise
