from lomas_core.schema import SourceConfig, VisionConfig
from lomas_vision.bus import ANY_SOURCE, FrameBus
from lomas_vision.frame import Frame
from lomas_vision.source import CAMERA_SOURCES, BaseSource, CameraSource
from lomas_vision.zoom import crop_rectangle

# Importing the package registers every backend.
from lomas_vision import sources as _sources  # noqa: F401

CAMERA_SOURCES.discover("lomas_vision.sources")


def build_sources(specs: list[SourceConfig]) -> list[CameraSource]:
    """Turn the `sources` config list into live objects, skipping disabled
    entries. Adding a camera is a config entry, never a code change."""
    return [CAMERA_SOURCES.create(spec.kind, spec) for spec in specs if spec.enabled]


__all__ = [
    "ANY_SOURCE",
    "CAMERA_SOURCES",
    "BaseSource",
    "CameraSource",
    "Frame",
    "FrameBus",
    "SourceConfig",
    "VisionConfig",
    "build_sources",
    "crop_rectangle",
]
