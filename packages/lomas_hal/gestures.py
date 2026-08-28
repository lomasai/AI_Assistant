from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from lomas_core.errors import LomasError

# The one source of truth for how the robot moves. The firmware table is
# generated from this file, so a gesture edited here and a gesture running on
# the board cannot drift apart - there is a test that fails if they have.
GESTURES_FILE = "gestures.yaml"
SERVOS_FILE = "servos.yaml"
SENSORS_FILE = "sensors.yaml"

JOINTS_KEY = "joints"
GESTURES_KEY = "gestures"
FRAMES_KEY = "frames"
AT_KEY = "at"
POSE_KEY = "pose"
ID_KEY = "id"

NO_TIME = 0.0
FULL_SPEED = 100


@dataclass(frozen=True, slots=True)
class Keyframe:
    at: float  # seconds from the start of the gesture
    pose: dict[str, float]  # joint name to degrees


@dataclass(frozen=True, slots=True)
class Gesture:
    name: str
    id: int
    frames: tuple[Keyframe, ...]
    description: str = ""

    @property
    def duration(self) -> float:
        return max((frame.at for frame in self.frames), default=NO_TIME)

    def joints(self) -> set[str]:
        return {joint for frame in self.frames for joint in frame.pose}

    def at(self, when: float) -> dict[str, float]:
        """Linear interpolation between keyframes.

        Only the simulator calls this. On real hardware the ESP32 does it, in
        an interrupt, because a Python pause here would be a servo glitch.
        """
        if not self.frames:
            return {}
        if when <= self.frames[0].at:
            return dict(self.frames[0].pose)
        if when >= self.frames[-1].at:
            return dict(self.frames[-1].pose)

        for before, after in zip(self.frames, self.frames[1:]):
            if before.at <= when <= after.at:
                span = after.at - before.at
                share = (when - before.at) / span if span else NO_TIME
                return {
                    joint: value + (after.pose.get(joint, value) - value) * share
                    for joint, value in before.pose.items()
                }
        return dict(self.frames[-1].pose)


@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    channel: int
    min_degrees: float
    max_degrees: float
    rest_degrees: float
    servo: str = ""

    def clamp(self, degrees: float) -> float:
        return max(self.min_degrees, min(self.max_degrees, degrees))


class GestureLibrary:
    """Joints, limits and keyframes, read from config.

    No servo angle appears anywhere in this package's code. If a shoulder is
    reaching too far the fix is a number in a YAML file, not a rebuild.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.joints: dict[str, Joint] = {}
        self.gestures: dict[str, Gesture] = {}
        self.limits: dict[str, float] = {}
        self._load()

    def gesture(self, name: str) -> Gesture:
        if name not in self.gestures:
            known = ", ".join(sorted(self.gestures)) or "none"
            raise LomasError(f"no gesture '{name}'. Available: {known}")
        return self.gestures[name]

    def by_id(self, gesture_id: int) -> Gesture | None:
        return next((g for g in self.gestures.values() if g.id == gesture_id), None)

    def joint(self, name: str) -> Joint:
        if name not in self.joints:
            known = ", ".join(sorted(self.joints)) or "none"
            raise LomasError(f"no joint '{name}'. Configured: {known}")
        return self.joints[name]

    def names(self) -> list[str]:
        return sorted(self.gestures)

    def check(self) -> list[str]:
        """Every complaint at once.

        A gesture naming a joint that does not exist, or reaching past a
        limit, is caught here at start-up rather than by a servo buzzing
        against its stop in front of a class.
        """
        problems: list[str] = []
        seen_ids: dict[int, str] = {}

        for gesture in self.gestures.values():
            claimed = seen_ids.get(gesture.id)
            if claimed:
                problems.append(f"gesture id {gesture.id} used by both {claimed} and {gesture.name}")
            seen_ids[gesture.id] = gesture.name

            for frame in gesture.frames:
                for joint_name, degrees in frame.pose.items():
                    joint = self.joints.get(joint_name)
                    if joint is None:
                        problems.append(f"{gesture.name} moves unknown joint '{joint_name}'")
                        continue
                    if not joint.min_degrees <= degrees <= joint.max_degrees:
                        problems.append(
                            f"{gesture.name} sends {joint_name} to {degrees} deg, outside "
                            f"{joint.min_degrees}..{joint.max_degrees}"
                        )
        return problems

    def _load(self) -> None:
        servos = _read(self.root / SERVOS_FILE)
        for name, body in (servos.get(JOINTS_KEY) or {}).items():
            self.joints[name] = Joint(
                name=name,
                channel=int(body["channel"]),
                min_degrees=float(body["min"]),
                max_degrees=float(body["max"]),
                rest_degrees=float(body["rest"]),
                servo=body.get("servo", ""),
            )

        gestures = _read(self.root / GESTURES_FILE)
        for name, body in (gestures.get(GESTURES_KEY) or {}).items():
            frames = tuple(
                Keyframe(at=float(frame[AT_KEY]), pose={k: float(v) for k, v in frame[POSE_KEY].items()})
                for frame in body[FRAMES_KEY]
            )
            self.gestures[name] = Gesture(
                name=name,
                id=int(body[ID_KEY]),
                frames=frames,
                description=body.get("description", ""),
            )

        self.limits = {k: float(v) for k, v in (_read(self.root / SENSORS_FILE).get("limits") or {}).items()}

    # --- the firmware table -------------------------------------------------

    def as_c_header(self) -> str:
        """The same gestures, as a C array for the ESP32.

        Generated rather than hand written, because two hand-maintained
        copies of a movement table is one copy too many. A test regenerates
        this and fails if the checked-in header has drifted.
        """
        channels = {name: joint.channel for name, joint in self.joints.items()}
        lines = [
            "// Generated from config/hardware/gestures.yaml and servos.yaml.",
            "// Do not edit. Run tests/test_hardware.py to regenerate.",
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"#define GESTURE_COUNT {len(self.gestures)}",
            f"#define JOINT_COUNT {len(self.joints)}",
            "",
            "typedef struct { uint8_t channel; int16_t degrees; } lomas_joint_target_t;",
            "typedef struct { uint16_t at_ms; uint8_t count; "
            "const lomas_joint_target_t *targets; } lomas_keyframe_t;",
            "typedef struct { uint8_t id; const char *name; uint8_t frames; "
            "const lomas_keyframe_t *keyframes; } lomas_gesture_t;",
            "",
        ]

        for name in sorted(self.gestures):
            gesture = self.gestures[name]
            for index, frame in enumerate(gesture.frames):
                targets = ", ".join(
                    f"{{{channels[joint]}, {int(round(degrees))}}}"
                    for joint, degrees in sorted(frame.pose.items())
                    if joint in channels
                )
                lines.append(
                    f"static const lomas_joint_target_t {name}_f{index}[] = {{{targets}}};"
                )
            frames = ", ".join(
                f"{{{int(round(frame.at * 1000))}, "
                f"{len([j for j in frame.pose if j in channels])}, {name}_f{index}}}"
                for index, frame in enumerate(gesture.frames)
            )
            lines.append(f"static const lomas_keyframe_t {name}_frames[] = {{{frames}}};")
            lines.append("")

        table = ",\n".join(
            f'    {{{self.gestures[name].id}, "{name}", '
            f"{len(self.gestures[name].frames)}, {name}_frames}}"
            for name in sorted(self.gestures)
        )
        lines.append("static const lomas_gesture_t LOMAS_GESTURES[GESTURE_COUNT] = {")
        lines.append(table)
        lines.append("};")
        return "\n".join(lines) + "\n"


def _read(path: Path) -> dict:
    if not path.exists():
        raise LomasError(f"missing hardware config {path}")
    body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(body, dict):
        raise LomasError(f"{path}: expected a mapping")
    return body
