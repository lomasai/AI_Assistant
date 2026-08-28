"""The body, and the line between the Pi and the board.

The interesting tests here are the boundary ones. Python must not be in any
loop that has to be real time, the two copies of the protocol must not drift,
and no servo angle may appear in code. Each of those is a rule somebody could
break in one line, so each one is a test.
"""
from __future__ import annotations

import ast
import re
import struct
from pathlib import Path

import pytest

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.contracts import (
    SAFETY_CLEARED,
    SAFETY_HALT,
    SESSION_OPENED,
    VISION_TRACKS,
    SafetyHalt,
    TracksSeen,
    TrackView,
)
from lomas_core.errors import LomasError
from lomas_hal import (
    BACKENDS,
    Command,
    Error,
    Event,
    Flag,
    FrameError,
    GestureLibrary,
    Reader,
    Reply,
    crc8,
    decode,
    decode_telemetry,
    encode,
    encode_telemetry,
)
from lomas_hal.backends.esp32 import Esp32
from lomas_hal.protocol import (
    ACK_FORMAT,
    EVENT_FORMAT,
    FRAME_OVERHEAD,
    LOOK_AT_FORMAT,
    START,
    TELEMETRY_SIZE,
    VERSION,
)

from app import container, seed
from app.body import Body
from app.flow.states import SessionState

FIRMWARE = Path("firmware/esp32")
HARDWARE = Path("config/hardware")
HAL = Path("packages/lomas_hal")

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "web.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
    "flow.attendance_wait_seconds=1",
    "flow.answer_wait_seconds=1",
    "flow.tick_seconds=0.1",
    "hardware.enabled=true",
    "hardware.simulate_travel_time=false",
]


def build(*extra: str):
    cfg = load("config", "debug", [*HEADLESS, *extra], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    seed.demo_class(system)
    return system


@pytest.fixture
def system():
    built = build()
    yield built
    built.close()


@pytest.fixture
def library() -> GestureLibrary:
    return GestureLibrary(HARDWARE)


@pytest.fixture
def sim(system):
    system.body.start()
    return system.body.backend


# --- the wire -------------------------------------------------------------


def test_a_frame_survives_the_round_trip() -> None:
    payload = struct.pack(LOOK_AT_FORMAT, 300, -50)
    frame = decode(encode(Command.LOOK_AT, 7, payload))

    assert frame.version == VERSION
    assert frame.seq == 7
    assert frame.command is Command.LOOK_AT
    assert struct.unpack(LOOK_AT_FORMAT, frame.payload) == (300, -50)


def test_a_flipped_bit_is_caught() -> None:
    """Servo cable next to a serial line, on a robot. Noise is normal, and a
    frame that arrives wrong must be refused rather than acted on."""
    raw = bytearray(encode(Command.MOVE, 1, struct.pack("<hh", 500, 0)))
    raw[-2] ^= 0x01

    with pytest.raises(FrameError, match="checksum"):
        decode(bytes(raw))


def test_the_checksum_covers_the_header_not_the_sync_bytes() -> None:
    frame = encode(Command.PING, 3)
    body = frame[len(START) : -1]
    assert crc8(body) == frame[-1]


def test_the_reader_resynchronises_after_rubbish() -> None:
    """One bad frame must cost one frame, not every frame after it."""
    reader = Reader()
    good = encode(Command.PING, 1)

    frames = reader.feed(b"\x00\xff\x12" + good + encode(Command.HALT, 2))

    assert [f.command for f in frames] == [Command.PING, Command.HALT]
    assert reader.dropped == 3


def test_a_frame_split_across_reads_still_arrives() -> None:
    reader = Reader()
    whole = encode(Command.GESTURE, 4, bytes((1, 100)))

    assert reader.feed(whole[:5]) == []
    frames = reader.feed(whole[5:])

    assert [f.command for f in frames] == [Command.GESTURE]


def test_a_lying_length_is_refused_not_trusted() -> None:
    raw = bytearray(encode(Command.PING, 1, b"\x01\x02"))
    raw[5] = 9  # claims nine payload bytes, carries two

    reader = Reader()
    assert reader.feed(bytes(raw)) == []


def test_telemetry_is_a_fixed_block() -> None:
    """The struct on the board and the format here must agree to the byte."""
    payload = encode_telemetry(
        flags=int(Flag.CALIBRATED | Flag.CLIFF),
        ultrasonic_mm=(1200, 900, 2000, 2000, 2000, 800),
        cliff_mm=(60, 61, 400, 59),
        pitch=-3.5, roll=1.2, yaw=0.0,
        current_ma=1800, battery_mv=11800, uptime_ms=120000,
    )
    assert len(payload) == TELEMETRY_SIZE

    reading = decode_telemetry(payload)
    assert reading.ultrasonic_mm[0] == 1200
    assert reading.cliff_mm[2] == 400
    assert reading.pitch == pytest.approx(-3.5)
    assert reading.has(Flag.CLIFF)
    assert not reading.safe


def test_a_short_telemetry_block_is_refused(library) -> None:
    with pytest.raises(FrameError):
        decode_telemetry(b"\x00" * (TELEMETRY_SIZE - 1))


# --- the two copies of the protocol must not drift ------------------------


def firmware_constants() -> dict[str, int]:
    text = (FIRMWARE / "protocol.h").read_text(encoding="utf-8")
    found: dict[str, int] = {}

    for name, value in re.findall(r"#define\s+(LOMAS_\w+)\s+(0x[0-9A-Fa-f]+|\d+)", text):
        found[name] = int(value, 0)
    for name, value in re.findall(r"(LOMAS_\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)", text):
        found[name] = int(value, 0)
    for name, shift in re.findall(r"(LOMAS_FLAG_\w+)\s*=\s*1\s*<<\s*(\d+)", text):
        found[name] = 1 << int(shift)
    return found


@pytest.mark.parametrize(
    "prefix,members",
    [
        ("LOMAS_CMD_", Command),
        ("LOMAS_ERR_", Error),
        ("LOMAS_EVENT_", Event),
        ("LOMAS_FLAG_", Flag),
    ],
)
def test_python_and_firmware_agree(prefix: str, members) -> None:
    """Two files describing one wire. If they disagree the robot moves in ways
    nobody asked for, so it is worth a test rather than a comment."""
    found = firmware_constants()
    for member in members:
        key = f"{prefix}{member.name}"
        assert key in found, f"{key} is missing from firmware/esp32/protocol.h"
        assert found[key] == int(member), f"{key} is {found[key]}, Python says {int(member)}"


def test_the_reply_codes_agree() -> None:
    found = firmware_constants()
    for member in Reply:
        assert found[f"LOMAS_REPLY_{member.name}"] == int(member)


def test_the_frame_shape_agrees() -> None:
    found = firmware_constants()
    assert found["LOMAS_PROTOCOL_VERSION"] == VERSION
    assert found["LOMAS_FRAME_OVERHEAD"] == FRAME_OVERHEAD
    assert found["LOMAS_TELEMETRY_SIZE"] == TELEMETRY_SIZE
    assert found["LOMAS_START_A"] == START[0]
    assert found["LOMAS_START_B"] == START[1]


def test_the_firmware_gesture_table_is_generated_from_the_yaml(library) -> None:
    """One source of truth for how the robot moves. Two hand-maintained
    copies of a movement table is one copy too many."""
    checked_in = (FIRMWARE / "gestures_generated.h").read_text(encoding="utf-8")
    assert checked_in == library.as_c_header(), (
        "firmware/esp32/gestures_generated.h has drifted from "
        "config/hardware/gestures.yaml. Regenerate it."
    )


# --- no servo angle in code -----------------------------------------------


def test_the_gestures_all_stay_inside_their_joint_limits(library) -> None:
    """Past a limit the horn hits the shell and the servo stalls. Caught at
    start-up, not by a buzzing servo in front of a class."""
    assert library.check() == []


def test_a_gesture_past_a_limit_is_caught(tmp_path: Path) -> None:
    for name in ["servos.yaml", "sensors.yaml", "gestures.yaml"]:
        (tmp_path / name).write_text((HARDWARE / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "gestures.yaml").write_text(
        "gestures:\n"
        "  reach:\n"
        "    id: 9\n"
        "    frames:\n"
        "      - {at: 0.0, pose: {shoulder_left: 400}}\n",
        encoding="utf-8",
    )

    problems = GestureLibrary(tmp_path).check()
    assert problems and "outside" in problems[0]


def test_a_gesture_naming_a_joint_that_does_not_exist_is_caught(tmp_path: Path) -> None:
    for name in ["servos.yaml", "sensors.yaml"]:
        (tmp_path / name).write_text((HARDWARE / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "gestures.yaml").write_text(
        "gestures:\n  wave:\n    id: 9\n    frames:\n      - {at: 0.0, pose: {tail: 10}}\n",
        encoding="utf-8",
    )

    assert "unknown joint" in GestureLibrary(tmp_path).check()[0]


def test_changing_the_yaml_changes_what_the_robot_does(tmp_path: Path) -> None:
    """The claim, tested rather than asserted: movement is config."""
    for name in ["servos.yaml", "sensors.yaml"]:
        (tmp_path / name).write_text((HARDWARE / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "gestures.yaml").write_text(
        "gestures:\n"
        "  bow:\n"
        "    id: 9\n"
        "    frames:\n"
        "      - {at: 0.0, pose: {neck_pitch: 0}}\n"
        "      - {at: 3.5, pose: {neck_pitch: 25}}\n",
        encoding="utf-8",
    )

    bow = GestureLibrary(tmp_path).gesture("bow")
    assert bow.duration == 3.5
    assert bow.at(1.75)["neck_pitch"] == pytest.approx(12.5)


def test_no_servo_angle_appears_in_any_python_file() -> None:
    """Angles live in config/hardware/gestures.yaml and nowhere else. A
    number here is a number nobody can tune without a rebuild."""
    offenders: list[str] = []
    joints = set(GestureLibrary(HARDWARE).joints)

    for path in [*HAL.rglob("*.py"), Path("app/body.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                named = isinstance(key, ast.Constant) and key.value in joints
                numeric = isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
                if named and numeric:
                    offenders.append(f"{path}:{node.lineno}")

    assert not offenders, offenders


def test_python_never_compares_a_distance_against_a_threshold() -> None:
    """Cliff, tilt and over-current are the board's job, in an interrupt.
    A cliff sensor read by a garbage-collected language is read late."""
    limits = set(GestureLibrary(HARDWARE).limits)
    assert limits, "sensors.yaml has no limits"

    for path in HAL.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            rendered = ast.unparse(node)
            for limit in limits:
                assert limit not in rendered, f"{path}: Python is deciding {limit}"


# --- the simulator --------------------------------------------------------


def test_the_simulator_sends_the_bytes_the_board_would_get(sim) -> None:
    sim.sent.clear()
    sim.gesture("namaste")

    frame = decode(sim.sent[-1])
    assert frame.command is Command.GESTURE
    assert frame.payload[0] == sim.library.gesture("namaste").id
    assert sim.frames()[-1].startswith("a5 5a")


def test_connecting_uploads_the_limits_to_the_board(sim) -> None:
    """The thresholds are sent once and applied there. Python knows the
    numbers only well enough to hand them over."""
    commands = [decode(frame).command for frame in sim.sent]
    assert Command.SET_LIMITS in commands
    assert Command.SET_TELEMETRY_HZ in commands


def test_look_at_is_clamped_to_the_neck(sim) -> None:
    sim.look_at(200.0, 0.0)

    yaw, _pitch = struct.unpack(LOOK_AT_FORMAT, decode(sim.sent[-1]).payload)
    assert yaw / 10.0 == sim.library.joints["neck_yaw"].max_degrees


def test_a_halted_robot_refuses_a_gesture_rather_than_queueing_it(sim) -> None:
    """A robot that catches up on three gestures the moment it is cleared is
    a robot that hurts someone."""
    sim.halt()
    handle = sim.gesture("celebrate")

    assert handle.cancelled
    assert "celebrate" not in sim.gestures_played


def test_halt_stops_what_is_already_running() -> None:
    system = build("hardware.simulate_travel_time=true")
    try:
        system.body.start()
        sim = system.body.backend
        handle = sim.gesture("namaste")
        assert not handle.done

        sim.halt()
        assert handle.cancelled
    finally:
        system.close()


def test_the_simulator_reports_a_clear_floor(sim) -> None:
    """Inventing a cliff would train everyone to ignore cliff warnings."""
    reading = sim.read_sensors()

    assert not reading.cliff
    assert not reading.tilted
    assert reading.battery_mv > 0


# --- the esp32 backend, without a board -----------------------------------


def esp32_for(cfg) -> Esp32:
    board = Esp32(cfg.hardware, FakeClock())
    board.connected = True
    return board


def test_a_missing_serial_library_says_what_to_do() -> None:
    cfg = load("config", "debug", [*HEADLESS, "hardware.backend=esp32"], use_env=False)
    board = Esp32(cfg.hardware)

    with pytest.raises(LomasError) as refused:
        board.connect()
    assert "simulator" in str(refused.value)


def test_telemetry_from_the_board_becomes_a_reading() -> None:
    cfg = load("config", "debug", HEADLESS, use_env=False)
    board = esp32_for(cfg)

    payload = encode_telemetry(
        flags=int(Flag.CALIBRATED),
        ultrasonic_mm=(1500,) * 6,
        cliff_mm=(58,) * 4,
        battery_mv=11900,
    )
    for frame in Reader().feed(encode(Reply.TELEMETRY, 0, payload)):
        board._handle(frame)

    assert board.read_sensors().battery_mv == 11900
    assert not board.read_sensors().halted


def test_a_board_cut_out_reaches_the_flow_as_a_halt() -> None:
    """The board has already stopped the motors. This is only how the lesson
    finds out - nothing in Python re-decides it."""
    system = build()
    try:
        board = Esp32(system.cfg.hardware, system.clock)
        body = Body(system.cfg, system.bus, system.clock, board)
        system.orchestrator.open_session()

        for frame in Reader().feed(
            encode(Reply.EVENT, 0, struct.pack(EVENT_FORMAT, int(Event.CLIFF_DETECTED), 40))
        ):
            board._handle(frame)

        assert board.halted
        halts = [p for _n, p in system.bus.replay(SAFETY_HALT)]
        assert halts and halts[-1].reason == "cliff_detected"
        assert halts[-1].detail["source"] == "esp32"
        assert system.extras["machine"].state is SessionState.HALTED
        assert body.played >= 0
    finally:
        system.close()


def test_a_refused_frame_is_logged_not_swallowed() -> None:
    cfg = load("config", "debug", HEADLESS, use_env=False)
    board = esp32_for(cfg)

    for frame in Reader().feed(
        encode(Reply.NACK, 0, struct.pack(ACK_FORMAT, 5, int(Error.UNKNOWN_GESTURE)))
    ):
        board._handle(frame)  # must not raise


def test_gesture_done_releases_the_handle() -> None:
    cfg = load("config", "debug", HEADLESS, use_env=False)
    board = esp32_for(cfg)
    handle = board.gesture("nod")

    for frame in Reader().feed(
        encode(Reply.EVENT, 0, struct.pack(EVENT_FORMAT, int(Event.GESTURE_DONE), 2))
    ):
        board._handle(frame)

    assert handle.done and not handle.cancelled


# --- the body, wired to the flow ------------------------------------------


def test_an_event_becomes_a_gesture_through_config(system, sim) -> None:
    sim.gestures_played.clear()
    system.orchestrator.open_session()

    assert sim.gestures_played == [system.cfg.hardware.gestures[SESSION_OPENED]]


def test_adding_a_gesture_is_a_config_line() -> None:
    system = build('hardware.gestures={"student.identified":"nod"}')
    try:
        system.body.start()
        system.bus.publish("student.identified", {"student_id": "s1"})
        assert system.body.backend.gestures_played == ["nod"]
    finally:
        system.close()


def test_the_head_follows_the_nearest_face(system, sim) -> None:
    sim.sent.clear()
    system.bus.publish(VISION_TRACKS, TracksSeen(
        source_id="head", zone="front", at=1.0, width=1280, height=720,
        tracks=(TrackView(track_id=1, x=900, y=200, w=120, h=120, student_id="s1",
                          attention=0.9, yaw=0.0, pitch=0.0, seen_for=3.0),),
    ))

    look = [decode(f) for f in sim.sent if decode(f).command is Command.LOOK_AT]
    assert look, "the head did not move"
    yaw, _ = struct.unpack(LOOK_AT_FORMAT, look[-1].payload)
    assert yaw > 0, "a face on the right should turn the head right"


def test_the_head_does_not_chase_small_movements(system, sim) -> None:
    """A head that snaps between children every tenth of a second is
    unsettling to watch and hard on the servos."""
    def seen(x: int) -> TracksSeen:
        return TracksSeen(source_id="head", zone="front", at=1.0, width=1280, height=720,
                          tracks=(TrackView(track_id=1, x=x, y=200, w=120, h=120,
                                            student_id="s1", attention=0.9, yaw=0.0,
                                            pitch=0.0, seen_for=3.0),))

    system.bus.publish(VISION_TRACKS, seen(900))
    sim.sent.clear()
    system.bus.publish(VISION_TRACKS, seen(905))

    assert not [f for f in sim.sent if decode(f).command is Command.LOOK_AT]


def test_a_safety_halt_reaches_the_servos(system, sim) -> None:
    system.bus.publish(SAFETY_HALT, SafetyHalt(reason="e-stop", at=0.0))
    assert sim.halted

    system.bus.publish(SAFETY_CLEARED, {"reason": "reset"})
    assert not sim.halted


# --- the acceptance -------------------------------------------------------


def test_a_full_lesson_runs_on_the_simulator(system, sim) -> None:
    assert system.orchestrator.run() is SessionState.CLOSED

    assert sim.gestures_played, "the robot never moved"
    assert all(frame.startswith("a5 5a") for frame in sim.frames())
    assert [decode(f).command for f in sim.sent][0] is Command.PING


def test_switching_to_real_hardware_is_one_config_key() -> None:
    assert set(BACKENDS.keys()) == {"simulator", "esp32"}

    cfg = load("config", "debug", [*HEADLESS, "hardware.backend=esp32"], use_env=False)
    board = BACKENDS.create(cfg.hardware.backend, cfg.hardware)
    assert isinstance(board, Esp32)
    assert board.library.names() == GestureLibrary(HARDWARE).names()


def test_a_class_runs_with_no_body_at_all() -> None:
    """Rule four, for the most expensive subsystem in the product."""
    cfg = load("config", "debug", [*HEADLESS, "hardware.enabled=false"], use_env=False)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    try:
        seed.demo_class(system)
        assert system.body is None
        assert system.orchestrator.run() is SessionState.CLOSED
    finally:
        system.close()
