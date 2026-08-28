from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from lomas_core import logging as log
from lomas_core.clock import Clock
from lomas_core.contracts import STUDENT_ENROLLED, StudentEnrolled
from lomas_core.errors import LomasError
from lomas_core.events import EventBus
from lomas_core.schema import Config
from lomas_face import EnrolmentSession, FrameFeedback
from lomas_face.types import Detection
from lomas_store import TenantScope
from lomas_store.repos.base import new_id
from lomas_vision import FrameBus

from app.pipeline import downscale, source_for

NO_FACE = "no face in front of the camera"
NO_CAMERA = "no camera frame; is vision switched on?"


@dataclass(slots=True)
class Enrolling:
    """One child, mid-sweep. Vectors only - there is no field here that could
    hold a picture, which is the point."""

    id: str
    student_id: str
    name: str
    sweep: EnrolmentSession
    started: float
    touched: float


class EnrolmentService:
    """Faces in, vectors out, and never a picture in between.

    It holds its own detector rather than borrowing the pipeline's. The
    pipeline is running on another thread with its own input size, and two
    threads sharing one detector is a fault you only meet on the Pi.
    """

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        clock: Clock,
        frames: FrameBus | None,
        detector: Any,
        embedder: Any,
        repos: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.frames = frames
        self.detector = detector
        self.embedder = embedder
        self.repos = repos
        self.log = log.get("enrol")
        self._open: dict[str, Enrolling] = {}
        self._lock = threading.RLock()

    # --- the sweep --------------------------------------------------------

    def start(
        self,
        scope: TenantScope,
        name: str,
        roll_no: str,
        granted_by: str,
        document_ref: str = "",
    ) -> dict:
        """Consent first, and on the server.

        A checkbox in a browser is a claim, not a record. Nothing here runs
        until a consent row exists with a person's name against it.
        """
        if not granted_by.strip():
            raise LomasError(
                "enrolment needs the name of the person giving consent for this child"
            )
        if not name.strip() or not roll_no.strip():
            raise LomasError("enrolment needs a name and a roll number")

        student = self.repos["student"].by_roll(scope, roll_no)
        student_id = student["id"] if student else self.repos["student"].create(scope, name, roll_no)

        kind = self.cfg.enrolment.consent_kind
        if not self.repos["consent"].is_granted(scope, student_id, kind):
            self.repos["consent"].grant(scope, student_id, kind, granted_by.strip(), document_ref)

        self._forget_abandoned()
        enrolling = Enrolling(
            id=new_id(),
            student_id=student_id,
            name=name.strip(),
            sweep=EnrolmentSession(self.embedder, self.cfg.enrolment, self.cfg.face.pose),
            started=self.clock.now(),
            touched=self.clock.now(),
        )
        with self._lock:
            self._open[enrolling.id] = enrolling

        return {
            "enrolment_id": enrolling.id,
            "student_id": student_id,
            "name": enrolling.name,
            "needed": len(self.cfg.enrolment.required_angles) * self.cfg.enrolment.keep_best,
            "angles": list(self.cfg.enrolment.required_angles),
            "sweep_seconds": self.cfg.enrolment.sweep_seconds,
        }

    def add_frame(self, enrolment_id: str) -> FrameFeedback:
        """One frame from the robot's own camera, embedded and dropped.

        The browser sends no image and receives none. The frame exists inside
        this call and nowhere else.
        """
        enrolling = self._require(enrolment_id)
        frame = self._latest()
        detection = self._largest_face(frame.image)
        if detection is None:
            raise LomasError(NO_FACE)

        enrolling.touched = self.clock.now()
        return enrolling.sweep.add_frame(frame.image, detection)

    def finish(self, scope: TenantScope, enrolment_id: str) -> dict:
        enrolling = self._require(enrolment_id)
        kind = self.cfg.enrolment.consent_kind

        # Checked again, because a revoke during the sweep has to mean the
        # vectors are never written rather than written and tidied up later.
        if not self.repos["consent"].is_granted(scope, enrolling.student_id, kind):
            self.cancel(enrolment_id)
            raise LomasError("consent was withdrawn during enrolment; nothing was stored")

        result = enrolling.sweep.finish()
        rows = result.as_rows()
        for vector, dim, dtype, quality, angle in rows:
            self.repos["embedding"].add(
                scope, enrolling.student_id, vector, dim, dtype, quality, angle
            )

        self.cancel(enrolment_id)
        stored = {
            "student_id": enrolling.student_id,
            "name": enrolling.name,
            "vectors": len(rows),
            "angles": list(result.angles),
            "quality": round(result.quality, 3),
        }
        self.bus.publish(
            STUDENT_ENROLLED,
            StudentEnrolled(
                student_id=enrolling.student_id,
                name=enrolling.name,
                vectors=len(rows),
                angles=tuple(result.angles),
                quality=result.quality,
            ),
        )
        self.log.info("%s enrolled with %d vectors", enrolling.name, len(rows))
        return stored

    def cancel(self, enrolment_id: str) -> None:
        with self._lock:
            self._open.pop(enrolment_id, None)

    def forget(self, scope: TenantScope, student_id: str) -> dict:
        """Un-enrolling has to be one call. A school that cannot remove a
        child's face data on request does not have consent, it has a form."""
        kind = self.cfg.enrolment.consent_kind
        stored = len(self.repos["embedding"].for_student(scope, student_id))
        self.repos["embedding"].delete_for_student(scope, student_id)
        self.repos["consent"].revoke(scope, student_id, kind)
        self.log.info("forgot %d vectors for %s", stored, student_id)
        return {"student_id": student_id, "removed": stored}

    # --- what the teacher screen reads ------------------------------------

    def enrolled(self, scope: TenantScope) -> list[dict]:
        kind = self.cfg.enrolment.consent_kind
        counts: dict[str, int] = {}
        for row in self.repos["embedding"].all_for_class(scope):
            counts[row["student_id"]] = counts.get(row["student_id"], 0) + 1

        return [
            {
                "id": student["id"],
                "name": student["name"],
                "roll_no": student["roll_no"],
                "vectors": counts.get(student["id"], 0),
                "consented": self.repos["consent"].is_granted(scope, student["id"], kind),
            }
            for student in self.repos["student"].list_for_class(scope)
        ]

    # --- internals --------------------------------------------------------

    def _require(self, enrolment_id: str) -> Enrolling:
        self._forget_abandoned()
        with self._lock:
            enrolling = self._open.get(enrolment_id)
        if enrolling is None:
            raise LomasError(f"no enrolment {enrolment_id}; it may have timed out")
        return enrolling

    def _forget_abandoned(self) -> None:
        cutoff = self.clock.now() - self.cfg.enrolment.abandoned_after_seconds
        with self._lock:
            stale = [k for k, v in self._open.items() if v.touched < cutoff]
            for key in stale:
                del self._open[key]
        if stale:
            self.log.debug("dropped %d abandoned enrolments", len(stale))

    def _latest(self):
        if self.frames is None:
            raise LomasError(NO_CAMERA)
        source = self.cfg.web.mjpeg_source or source_for(self.cfg)
        frame = self.frames.latest(source)
        if frame is None:
            raise LomasError(NO_CAMERA)
        return frame

    def _largest_face(self, image) -> Detection | None:
        small, factor = downscale(image, self.cfg.face.downscale_width)
        found = [d.scaled(factor) for d in self.detector.detect(small)]
        usable = [d for d in found if d.confidence >= self.cfg.face.min_confidence]
        return max(usable, key=lambda d: d.area) if usable else None
