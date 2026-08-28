from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# This is the only file in the repository where a default value may be written.
# Every other module reads its numbers from a Config instance.

Strict = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    model_config = Strict

    mode: Literal["debug", "user"] = "user"
    locale: str = "en"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sinks: list[Literal["console", "jsonl"]] = Field(default_factory=lambda: ["console"])
    log_dir: str = "data/logs"
    event_replay_size: int = Field(default=512, ge=1)
    raise_on_handler_error: bool = True


class TenancyConfig(BaseModel):
    model_config = Strict

    org_id: str = "lomas-demo"
    school_id: str = "sunrise-nashik"
    class_id: str = "grade-6b"
    scratch_org_id: str = "scratch"


class StorageConfig(BaseModel):
    model_config = Strict

    backend: Literal["sqlite", "memory"] = "sqlite"
    path: str = "data/lomas.db"
    retention_days: int = Field(default=180, ge=1)
    purge_on_term_end: bool = True
    busy_timeout_ms: int = Field(default=5000, ge=0)


class SourceConfig(BaseModel):
    """One camera. `sources` is a list from the first commit, even with a
    single entry, so classroom CCTV and multi-angle capture arrive as config
    rather than as a redesign."""

    model_config = Strict

    id: str
    kind: Literal["picamera2", "usb", "rtsp", "file", "folder", "mock"] = "mock"
    zone: str = "front"
    enabled: bool = True
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)
    fps: int = Field(default=15, ge=0)  # 0 means capture as fast as the device allows
    zoom: float = Field(default=1.0, ge=1.0, le=2.5)
    rotation: Literal[0, 90, 180, 270] = 0
    device: str | int | None = None  # usb index or device path
    path: str | None = None  # file or folder source
    url: str | None = None  # rtsp source
    loop: bool = True  # replay sources start over at the end


class VisionConfig(BaseModel):
    model_config = Strict

    buffer_size: int = Field(default=4, ge=1)
    read_timeout_ms: int = Field(default=200, ge=1)


def _default_sources() -> list[SourceConfig]:
    return [SourceConfig(id="head")]


class Config(BaseModel):
    model_config = Strict

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sources: list[SourceConfig] = Field(default_factory=_default_sources)
    vision: VisionConfig = Field(default_factory=VisionConfig)

    @property
    def is_debug(self) -> bool:
        return self.runtime.mode == "debug"

    @property
    def active_org_id(self) -> str:
        """Debug runs write to a scratch tenant so bench testing never lands
        in a real school's data."""
        return self.tenancy.scratch_org_id if self.is_debug else self.tenancy.org_id
