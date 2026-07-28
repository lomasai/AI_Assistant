"""Admin registration and local face recognition services."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.config import RecognitionConfig
from server.face_providers import FaceProviderError, build_face_embedding_provider
from server.student_store import SQLiteStudentRepository, StudentStoreError


class RegistrationError(Exception):
    """Raised for controlled registration and recognition failures."""


class RegistrationStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=80)
    grade_level: str | None = Field(default=None, max_length=40)
    language: str | None = Field(default=None, max_length=20)
    consent_given: bool = False


class RegistrationSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(default="", max_length=2_000_000)
    pose_label: str | None = Field(default=None, max_length=20)
    embedding_seed: str | None = Field(default=None, max_length=200)
    quality_override: str | None = Field(default=None, max_length=40)
    brightness: float | None = Field(default=None, ge=0, le=255)
    blur_score: float | None = Field(default=None, ge=0)
    face_count: int | None = Field(default=None, ge=0, le=8)


class RecognitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(default="", max_length=2_000_000)
    embedding_seed: str | None = Field(default=None, max_length=200)
    quality_override: str | None = Field(default=None, max_length=40)
    brightness: float | None = Field(default=None, ge=0, le=255)
    blur_score: float | None = Field(default=None, ge=0)
    face_count: int | None = Field(default=None, ge=0, le=8)


@dataclass(slots=True)
class SampleQuality:
    accepted: bool
    reason: str
    brightness: float
    blur_score: float
    face_count: int
    quality: float


class FaceRegistrationService:
    """Coordinates consent, sample quality gates, local embeddings and recognition."""

    def __init__(self, store: SQLiteStudentRepository, config: RecognitionConfig) -> None:
        self.store = store
        self.config = config
        self.embedding_provider = build_face_embedding_provider(config)
        self._last_recognition_at = 0.0
        self._last_result: dict[str, Any] | None = None
        self.temporary_frames_retained = 0

    async def initialize(self) -> None:
        try:
            await self.embedding_provider.initialize()
        except FaceProviderError:
            raise

    def health(self) -> dict[str, Any]:
        return self.embedding_provider.health()

    async def start_registration(self, payload: RegistrationStart) -> dict[str, Any]:
        if not payload.consent_given:
            raise RegistrationError("Explicit consent is required before registration.")
        try:
            student = await self.store.create_student(
                display_name=payload.display_name,
                grade_level=payload.grade_level,
                language=payload.language,
                consent_given=True,
                status="registering",
            )
            session = await self.store.start_registration_session(
                student_id=student["id"],
                required_samples=self.config.registration_sample_count,
            )
        except StudentStoreError as exc:
            raise RegistrationError(str(exc)) from exc
        return {
            "registration": self._registration_view(session),
            "student": self._student_view(student),
            "next_guidance": self._guidance_for(0),
        }

    async def submit_sample(self, registration_id: str, sample: RegistrationSample) -> dict[str, Any]:
        session = await self.store.get_registration_session(registration_id)
        if not session:
            raise RegistrationError("Registration session not found.")
        if session["status"] not in {"capturing", "ready_to_verify"}:
            raise RegistrationError("Registration is not accepting samples.")

        quality = self._evaluate_quality(sample)
        if not quality.accepted:
            updated = await self.store.update_registration_session(registration_id, rejected_delta=1)
            return {
                "accepted": False,
                "reason": quality.reason,
                "registration": self._registration_view(updated),
                "next_guidance": self._guidance_for(updated["accepted_samples"]),
            }

        student = await self.store.get_student(session["student_id"])
        if not student:
            raise RegistrationError("Student profile not found.")
        pose = sample.pose_label or self._guidance_for(session["accepted_samples"])
        embedding = self._embedding_for_sample(sample, fallback_seed=f"{student['id']}:{pose}")
        await self.store.save_embedding(student["id"], embedding, quality.quality, pose)
        accepted_total = session["accepted_samples"] + 1
        status = "ready_to_verify" if accepted_total >= session["required_samples"] else "capturing"
        updated = await self.store.update_registration_session(registration_id, status=status, accepted_delta=1)
        if not self.config.retain_temporary_media:
            self.temporary_frames_retained = 0
        return {
            "accepted": True,
            "reason": "accepted",
            "registration": self._registration_view(updated),
            "next_guidance": self._guidance_for(updated["accepted_samples"]),
        }

    async def registration_status(self, registration_id: str) -> dict[str, Any]:
        session = await self.store.get_registration_session(registration_id)
        if not session:
            raise RegistrationError("Registration session not found.")
        student = await self.store.get_student(session["student_id"])
        return {
            "registration": self._registration_view(session),
            "student": self._student_view(student) if student else None,
            "next_guidance": self._guidance_for(session["accepted_samples"]),
        }

    async def complete_registration(self, registration_id: str) -> dict[str, Any]:
        session = await self.store.get_registration_session(registration_id)
        if not session:
            raise RegistrationError("Registration session not found.")
        if session["accepted_samples"] < session["required_samples"]:
            raise RegistrationError("More accepted samples are required before verification.")
        embeddings = await self.store.get_embeddings(session["student_id"])
        if not embeddings:
            raise RegistrationError("No local embeddings are available for verification.")
        verified = self._best_match(embeddings[0]["embedding"], embeddings)
        if verified is None:
            raise RegistrationError("Registration verification failed.")
        await self.store.update_student_status(session["student_id"], "registered")
        updated = await self.store.update_registration_session(registration_id, status="completed")
        student = await self.store.get_student(session["student_id"])
        return {
            "verified": True,
            "registration": self._registration_view(updated),
            "student": self._student_view(student) if student else None,
        }

    async def cancel_registration(self, registration_id: str) -> dict[str, Any]:
        session = await self.store.get_registration_session(registration_id)
        if not session:
            raise RegistrationError("Registration session not found.")
        updated = await self.store.update_registration_session(registration_id, status="cancelled")
        student = await self.store.get_student(session["student_id"])
        if student and student["registration_status"] != "registered":
            await self.store.delete_student(student["id"])
        self.temporary_frames_retained = 0
        return {"cancelled": True, "registration": self._registration_view(updated)}

    async def recognize_current_student(self, payload: RecognitionRequest) -> dict[str, Any]:
        now = time.monotonic()
        if self._last_result and now - self._last_recognition_at < self.config.recognition_interval_seconds:
            return dict(self._last_result)

        quality = self._evaluate_quality(payload)
        if not quality.accepted:
            result = self._guest_result(reason=quality.reason)
            self._remember_result(now, result)
            return result

        embeddings = [item for item in await self.store.get_embeddings() if item["registration_status"] == "registered"]
        if not embeddings:
            result = self._guest_result(reason="no_registered_students")
            self._remember_result(now, result)
            return result

        try:
            probe = self._embedding_for_sample(payload, fallback_seed="current-frame")
        except RegistrationError:
            probe = []
        if not probe:
            result = self._guest_result(reason="no_face")
            self._remember_result(now, result)
            return result
        match = self._best_match(probe, embeddings)
        result = match or self._guest_result(reason="below_threshold")
        self._remember_result(now, result)
        return result

    def _remember_result(self, now: float, result: dict[str, Any]) -> None:
        self._last_recognition_at = now
        self._last_result = dict(result)

    def _evaluate_quality(self, sample: RegistrationSample | RecognitionRequest) -> SampleQuality:
        override = (sample.quality_override or "").strip().lower()
        face_count = sample.face_count
        brightness = sample.brightness
        blur_score = sample.blur_score
        if override == "dark":
            brightness = 0.0
        if override == "blurry":
            blur_score = 0.0
        if override == "no_face":
            face_count = 0
        if override == "multi_face":
            face_count = 2
        if override in {"ok", "good", ""}:
            pass

        resolved_face_count = 1 if face_count is None else face_count
        resolved_brightness = 128.0 if brightness is None else brightness
        resolved_blur = self.config.blur_threshold + 20.0 if blur_score is None else blur_score

        if resolved_face_count == 0:
            return SampleQuality(False, "no_face", resolved_brightness, resolved_blur, resolved_face_count, 0.0)
        if resolved_face_count > 1:
            return SampleQuality(False, "multi_face", resolved_brightness, resolved_blur, resolved_face_count, 0.0)
        if resolved_brightness < self.config.brightness_min or resolved_brightness > self.config.brightness_max:
            return SampleQuality(False, "dark_or_overexposed", resolved_brightness, resolved_blur, resolved_face_count, 0.0)
        if resolved_blur < self.config.blur_threshold:
            return SampleQuality(False, "blurry", resolved_brightness, resolved_blur, resolved_face_count, 0.0)

        brightness_score = 1.0 - min(abs(128.0 - resolved_brightness) / 128.0, 1.0)
        blur_score_norm = min(resolved_blur / max(self.config.blur_threshold, 1.0), 2.0) / 2.0
        return SampleQuality(True, "accepted", resolved_brightness, resolved_blur, resolved_face_count, (brightness_score + blur_score_norm) / 2.0)

    def _embedding_for_sample(self, sample: RegistrationSample | RecognitionRequest, fallback_seed: str) -> list[float]:
        if getattr(self.embedding_provider, "provider_name", "") == "mock":
            seed = sample.embedding_seed or sample.image_base64 or fallback_seed
            return self.embedding_provider.embed_seed(seed)
        embedding, face_count = self.embedding_provider.embed_image(sample.image_base64)
        if face_count != 1:
            raise RegistrationError("Exactly one face is required.")
        return embedding

    def _best_match(self, probe: list[float], embeddings: list[dict[str, Any]]) -> dict[str, Any] | None:
        best: tuple[float, dict[str, Any]] | None = None
        for item in embeddings:
            score = self._cosine(probe, item["embedding"])
            if best is None or score > best[0]:
                best = (score, item)
        if best is None or best[0] < self.config.face_match_threshold:
            return None
        return {
            "recognized": True,
            "student_id": best[1]["student_id"],
            "display_name": best[1]["display_name"],
            "label": best[1]["display_name"],
            "confidence_label": "matched",
            "reason": "threshold_met",
        }

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        dot = sum(left[index] * right[index] for index in range(size))
        left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
        return dot / (left_norm * right_norm)

    def _guest_result(self, reason: str) -> dict[str, Any]:
        return {
            "recognized": False,
            "student_id": None,
            "display_name": self.config.unknown_label,
            "label": self.config.unknown_label,
            "confidence_label": "unknown",
            "reason": reason,
        }

    def _guidance_for(self, accepted_samples: int) -> str:
        poses = ["center", "left", "right", "center", "left"]
        return poses[min(accepted_samples, len(poses) - 1)]

    @staticmethod
    def _student_view(student: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": student["id"],
            "display_name": student["display_name"],
            "grade_level": student.get("grade_level"),
            "language": student.get("language"),
            "consent_given": bool(student.get("consent_given")),
            "consent_timestamp_utc": student.get("consent_timestamp_utc"),
            "registration_status": student.get("registration_status"),
            "created_at_utc": student.get("created_at_utc"),
            "updated_at_utc": student.get("updated_at_utc"),
        }

    @staticmethod
    def _registration_view(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": session["id"],
            "student_id": session["student_id"],
            "status": session["status"],
            "required_samples": session["required_samples"],
            "accepted_samples": session["accepted_samples"],
            "rejected_samples": session["rejected_samples"],
            "created_at_utc": session["created_at_utc"],
            "updated_at_utc": session["updated_at_utc"],
        }
