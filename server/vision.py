"""Browser frame multimodal vision analysis service."""

from __future__ import annotations

import base64
import binascii
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from edge.tracking import FaceBox, TrackingError, face_tracker
from server.vision_analyzer import estimate_age_range, estimate_apparent_gender, estimate_expression


class VisionError(Exception):
    """Raised when an image payload cannot be analyzed."""


@dataclass(slots=True)
class VisionRuntimeState:
    frames_analyzed: int = 0
    last_analysis_at: str | None = None
    last_face_seen_at: float | None = None
    last_latency_ms: float | None = None
    backend: str = "mock"
    last_error: str | None = None
    last_attention_state: str = "unknown"
    looking_away_frames: int = 0


@dataclass(slots=True)
class VisionService:
    """Decode frames and return safe robot-vision intelligence."""

    no_face_alert_seconds: float = 5.0
    state: VisionRuntimeState = field(default_factory=VisionRuntimeState)

    def analyze(
        self,
        image_base64: str,
        timestamp: str | None = None,
        include_decision: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        analyzed_at = timestamp or datetime.now(timezone.utc).isoformat()
        ctx = context or {}
        frame: Any | None = None
        frame_size: tuple[int, int] | None = None

        try:
            frame = self._decode_image(image_base64)
            frame_size = self._frame_size(frame)
            raw_faces = self._detect_faces(frame)
            self.state.backend = "opencv"
            self.state.last_error = None
        except (ModuleNotFoundError, ImportError) as exc:
            raw_faces = []
            self.state.backend = "mock"
            self.state.last_error = str(exc)
        except VisionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raw_faces = []
            self.state.backend = "mock"
            self.state.last_error = str(exc)

        now = time.time()
        face_boxes = [self._face_to_dict(face) for face in raw_faces]
        if face_boxes:
            self.state.last_face_seen_at = now

        face = self._build_face(face_boxes, frame, frame_size)
        eyes_attention = self._build_attention(face, frame_size)
        body_posture = self._build_body_posture(face, ctx, frame_size)
        tracking = self._build_tracking(face, frame_size)
        sensors = self._build_sensors(ctx.get("sensor_data"))
        health_behavior = self._build_health_behavior(body_posture, eyes_attention, ctx)
        decision = (
            self._build_decision(face, eyes_attention, body_posture, tracking, health_behavior, sensors, ctx, now)
            if include_decision
            else None
        )

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.state.frames_analyzed += 1
        self.state.last_analysis_at = analyzed_at
        self.state.last_latency_ms = latency_ms

        return {
            "ok": True,
            "timestamp": analyzed_at,
            "latency_ms": latency_ms,
            "face": face,
            "eyes_attention": eyes_attention,
            "body_posture": body_posture,
            "tracking": tracking,
            "health_behavior": health_behavior,
            "sensors": sensors,
            "decision": decision,
            "overlays": self._build_overlays(face, body_posture, eyes_attention, tracking),
        }

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": self.state.backend,
            "frames_analyzed": self.state.frames_analyzed,
            "last_analysis_at": self.state.last_analysis_at,
            "last_latency_ms": self.state.last_latency_ms,
            "last_error": self.state.last_error,
        }

    @staticmethod
    def _decode_image(image_base64: str) -> Any:
        if not image_base64 or not image_base64.strip():
            raise VisionError("image_base64 is required.")
        raw = image_base64.strip()
        if "," in raw and raw.lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VisionError(f"Invalid base64 image payload: {exc}") from exc
        if not image_bytes:
            raise VisionError("Decoded image payload is empty.")
        try:
            import cv2  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise ModuleNotFoundError("OpenCV is not installed; using mock vision response.") from exc
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise VisionError("Image payload could not be decoded.")
        return frame

    @staticmethod
    def _detect_faces(frame: Any) -> list[FaceBox]:
        try:
            return face_tracker.detect_faces(frame)
        except TrackingError:
            return []

    @staticmethod
    def _frame_size(frame: Any) -> tuple[int, int] | None:
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            return None
        return int(shape[1]), int(shape[0])

    @staticmethod
    def _face_to_dict(face: FaceBox) -> dict[str, float | int]:
        return {
            "x": int(face.x),
            "y": int(face.y),
            "width": int(face.w),
            "height": int(face.h),
            "confidence": round(float(face.confidence), 4),
        }

    def _build_face(self, boxes: list[dict[str, Any]], frame: Any | None, frame_size: tuple[int, int] | None) -> dict[str, Any]:
        primary = self._primary_box(boxes)
        center_x = center_y = None
        crop = None
        if primary and frame_size:
            center_x = round((primary["x"] + primary["width"] / 2) / max(1, frame_size[0]), 4)
            center_y = round((primary["y"] + primary["height"] / 2) / max(1, frame_size[1]), 4)
            if frame is not None:
                x, y, w, h = int(primary["x"]), int(primary["y"]), int(primary["width"]), int(primary["height"])
                crop = frame[y : y + h, x : x + w]
        confidence = round(float(primary["confidence"]), 4) if primary else 0.0
        return {
            "detected": bool(boxes),
            "count": len(boxes),
            "boxes": boxes,
            "center_x": center_x,
            "center_y": center_y,
            "confidence": confidence,
            "apparent_gender_estimate": self._uncertain_if_low(estimate_apparent_gender(crop)),
            "estimated_age_range": self._uncertain_if_low(estimate_age_range(crop)),
            "expression": self._uncertain_if_low(estimate_expression(crop)),
        }

    def _build_attention(self, face: dict[str, Any], frame_size: tuple[int, int] | None) -> dict[str, Any]:
        detected = bool(face["detected"])
        cx = face.get("center_x")
        cy = face.get("center_y")
        eyes_visible = detected and float(face.get("confidence", 0.0)) >= 0.65
        eye_contact = "unknown"
        reason = "no face" if not detected else "low confidence"
        if detected and isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
            if cx < 0.38:
                eye_contact = "looking_left"
                reason = "looking away"
            elif cx > 0.62:
                eye_contact = "looking_right"
                reason = "looking away"
            elif cy < 0.32:
                eye_contact = "looking_up"
                reason = "looking away"
            elif cy > 0.72:
                eye_contact = "looking_down"
                reason = "looking away"
            elif eyes_visible:
                eye_contact = "looking_at_camera"
                reason = "none"
        attention_state = "unknown"
        if eye_contact == "looking_at_camera":
            attention_state = "attentive"
            self.state.looking_away_frames = 0
        elif eye_contact in {"looking_left", "looking_right", "looking_down", "looking_up"}:
            attention_state = "distracted"
            self.state.looking_away_frames += 1
        elif not detected:
            self.state.looking_away_frames += 1
        return {
            "eyes_visible": eyes_visible,
            "eye_contact": eye_contact,
            "attention_state": attention_state,
            "blink_detected": False,
            "eye_aspect_ratio": None,
            "distraction_reason": reason,
            "landmarks": self._mock_eye_landmarks(face, frame_size) if eyes_visible else [],
        }

    def _build_body_posture(self, face: dict[str, Any], context: dict[str, Any], frame_size: tuple[int, int] | None) -> dict[str, Any]:
        detected = bool(face["detected"])
        cx = face.get("center_x")
        primary = self._primary_box(face.get("boxes", []))
        body_position = "unknown"
        if isinstance(cx, (int, float)):
            if cx < 0.35:
                body_position = "left"
            elif cx > 0.65:
                body_position = "right"
            else:
                body_position = "centered"
        if primary and frame_size:
            ratio = primary["height"] / max(1, frame_size[1])
            if ratio > 0.55:
                body_position = "too_close"
            elif ratio < 0.12:
                body_position = "too_far"
        sitting_minutes = int(self._coerce_float(context.get("sitting_minutes")) or 0)
        posture = "sitting" if detected else "unknown"
        confidence = 0.55 if detected else 0.0
        warning = "Take a stretch break." if sitting_minutes > 60 else None
        return {
            "person_detected": detected,
            "posture": posture,
            "confidence": confidence,
            "body_position": body_position,
            "sitting_minutes": sitting_minutes,
            "posture_warning": warning,
            "body_box": self._body_box_from_face(primary, frame_size),
            "pose_points": [],
        }

    @staticmethod
    def _build_tracking(face: dict[str, Any], frame_size: tuple[int, int] | None) -> dict[str, Any]:
        _ = frame_size
        if not face["detected"]:
            return {
                "target_x": None,
                "target_y": None,
                "direction": "center",
                "recommended_motor_action": "hold_position",
                "motor_ready": False,
                "tracking_quality": "lost",
            }
        x = face.get("center_x") if isinstance(face.get("center_x"), (int, float)) else 0.5
        y = face.get("center_y") if isinstance(face.get("center_y"), (int, float)) else 0.5
        direction = "center"
        action = "hold_position"
        if x < 0.4:
            direction, action = "left", "move_left"
        elif x > 0.6:
            direction, action = "right", "move_right"
        elif y < 0.35:
            direction, action = "up", "tilt_up"
        elif y > 0.65:
            direction, action = "down", "tilt_down"
        quality = "good" if direction == "center" and face.get("confidence", 0) >= 0.65 else "weak"
        return {
            "target_x": round(float(x), 4),
            "target_y": round(float(y), 4),
            "direction": direction,
            "recommended_motor_action": action,
            "motor_ready": True,
            "tracking_quality": quality,
        }

    @staticmethod
    def _build_sensors(sensor_data: Any) -> dict[str, Any]:
        source = "context" if isinstance(sensor_data, dict) and sensor_data else "mock"
        data = sensor_data if isinstance(sensor_data, dict) else {}
        return {
            "temperature": data.get("temperature", 24.5 if source == "mock" else None),
            "humidity": data.get("humidity", 48 if source == "mock" else None),
            "distance_cm": data.get("distance_cm", None),
            "light_level": data.get("light_level", "normal" if source == "mock" else None),
            "battery_percent": data.get("battery_percent", 86 if source == "mock" else None),
            "motion_detected": data.get("motion_detected", False if source == "mock" else None),
            "edge_device_connected": bool(data.get("edge_device_connected", False)),
            "source": source,
        }

    @staticmethod
    def _build_health_behavior(body: dict[str, Any], attention: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        sitting_warning = int(body.get("sitting_minutes") or 0) > 60
        water_minutes = VisionService._coerce_float(context.get("water_gap_minutes"))
        medicine_status = str(context.get("medicine_status", "ok"))
        fatigue_warning = attention.get("attention_state") == "drowsy"
        drowsiness_warning = bool(attention.get("blink_detected")) or fatigue_warning
        attention_warning = attention.get("attention_state") == "distracted"
        water_status = "overdue" if water_minutes is not None and water_minutes > 120 else "ok"
        summary = "No alert."
        if sitting_warning:
            summary = "Sitting time is high."
        elif attention_warning:
            summary = "User may be distracted."
        elif medicine_status == "overdue":
            summary = "Medicine reminder is overdue."
        return {
            "sitting_time_warning": sitting_warning,
            "water_reminder_status": water_status,
            "medicine_reminder_status": medicine_status,
            "fatigue_warning": fatigue_warning,
            "drowsiness_warning": drowsiness_warning,
            "attention_warning": attention_warning,
            "behavior_summary": summary,
        }

    def _build_decision(
        self,
        face: dict[str, Any],
        attention: dict[str, Any],
        body: dict[str, Any],
        tracking: dict[str, Any],
        health: dict[str, Any],
        sensors: dict[str, Any],
        context: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        _ = sensors, context
        if not face["detected"]:
            no_face_seconds = now - self.state.last_face_seen_at if self.state.last_face_seen_at else self.no_face_alert_seconds
            return {
                "message": "No user visible.",
                "alert_level": "info" if no_face_seconds >= self.no_face_alert_seconds else "normal",
                "recommended_action": "scan_for_user",
                "should_speak": False,
                "should_log_event": no_face_seconds >= self.no_face_alert_seconds,
                "should_move_robot": False,
            }
        if body.get("posture") == "falling_risk":
            return self._decision("Possible fall risk detected.", "critical", "request_help", True, True, False)
        if health.get("sitting_time_warning"):
            return self._decision("Take a stretch break.", "warning", "suggest_stand_break", True, True, False)
        if health.get("medicine_reminder_status") == "overdue":
            return self._decision("Medicine reminder is overdue.", "warning", "medicine_reminder", True, True, False)
        if health.get("water_reminder_status") == "overdue":
            return self._decision("Consider drinking water.", "info", "water_reminder", False, False, False)
        if attention.get("attention_state") == "drowsy":
            return self._decision("User may be drowsy.", "warning", "check_on_user", True, True, False)
        if attention.get("attention_state") == "distracted" and self.state.looking_away_frames >= 3:
            return self._decision("User may be distracted.", "warning", "refocus_prompt", False, True, False)
        should_move = tracking.get("recommended_motor_action") != "hold_position"
        message = "User appears attentive and centered. Posture looks normal. No health alert."
        if should_move:
            message = f"User detected. Robot can adjust {tracking.get('direction')} to center the target."
        return self._decision(message, "normal", tracking.get("recommended_motor_action"), False, False, should_move)

    @staticmethod
    def _decision(message: str, level: str, action: str | None, speak: bool, log: bool, move: bool) -> dict[str, Any]:
        return {
            "message": message,
            "alert_level": level,
            "recommended_action": action,
            "should_speak": speak,
            "should_log_event": log,
            "should_move_robot": move,
        }

    @staticmethod
    def _build_overlays(face: dict[str, Any], body: dict[str, Any], attention: dict[str, Any], tracking: dict[str, Any]) -> dict[str, Any]:
        return {
            "face_boxes": face.get("boxes", []),
            "body_boxes": [body["body_box"]] if body.get("body_box") else [],
            "eye_landmarks": attention.get("landmarks", []),
            "pose_points": body.get("pose_points", []),
            "gaze": {
                "from_x": face.get("center_x"),
                "from_y": face.get("center_y"),
                "direction": tracking.get("direction"),
            },
            "attention_indicator": attention.get("attention_state"),
        }

    @staticmethod
    def _primary_box(boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not boxes:
            return None
        return max(boxes, key=lambda box: int(box["width"]) * int(box["height"]))

    @staticmethod
    def _body_box_from_face(face: dict[str, Any] | None, frame_size: tuple[int, int] | None) -> dict[str, Any] | None:
        if not face or not frame_size:
            return None
        x = max(0, int(face["x"] - face["width"] * 0.65))
        y = int(face["y"])
        width = min(frame_size[0] - x, int(face["width"] * 2.3))
        height = min(frame_size[1] - y, int(face["height"] * 4.2))
        return {"x": x, "y": y, "width": max(1, width), "height": max(1, height), "confidence": 0.45}

    @staticmethod
    def _mock_eye_landmarks(face: dict[str, Any], frame_size: tuple[int, int] | None) -> list[dict[str, float]]:
        primary = VisionService._primary_box(face.get("boxes", []))
        if not primary or not frame_size:
            return []
        fw, fh = frame_size
        y = (primary["y"] + primary["height"] * 0.38) / fh
        return [
            {"x": (primary["x"] + primary["width"] * 0.35) / fw, "y": y},
            {"x": (primary["x"] + primary["width"] * 0.65) / fw, "y": y},
        ]

    @staticmethod
    def _uncertain_if_low(result: dict[str, Any], threshold: float = 0.55) -> dict[str, Any]:
        confidence = float(result.get("confidence", 0.0) or 0.0)
        label = str(result.get("label", "unknown"))
        if confidence < threshold:
            label = "unknown"
        return {"label": label, "confidence": round(confidence, 4)}

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


vision_service = VisionService()
