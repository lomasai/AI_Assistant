#!/usr/bin/env python3
"""Is this robot able to do everything it is configured to do?

Installing one optional thing quietly upgrades another - mediapipe pulls its
own OpenCV, which pulls numpy 2, which is compiled against a different ABI
than the camera bindings the system package built against. Nothing says so
until a class is running and the camera is dark.

    python tools/doctor.py --mode pi

Every line is a thing the config asks for, and whether this machine can do
it. Nothing is downloaded, nothing is changed, and a missing optional part
is reported rather than raised.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

from lomas_core.config import load  # noqa: E402

OK = "ok"
MISSING = "missing"
BROKEN = "BROKEN"
NOT_ASKED = "-"
WIDTH = 22
RULE = 60


def version_of(module) -> str:
    return str(getattr(module, "__version__", "") or "installed")


def imports(name: str) -> tuple[str, str]:
    """What happened when this was imported. A module that raises on import
    is the one worth knowing about: it is installed, so nothing reports it
    as absent, and it fails where nobody is looking."""
    try:
        return OK, version_of(__import__(name))
    except ImportError as exc:
        return MISSING, str(exc).split("(")[0].strip()
    except Exception as exc:  # a wheel built against another numpy does this
        return BROKEN, f"{type(exc).__name__}: {exc}"[:120]


SEEN: list[tuple[str, str, str]] = []


def say(what: str, state: str, detail: str = "") -> bool:
    SEEN.append((state, what, detail))
    print(f"  {state:<8} {what:<{WIDTH}} {detail}")
    return state != BROKEN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="pi")
    ap.add_argument("--config-dir", default=str(ROOT / "config"))
    args = ap.parse_args()

    cfg = load(args.config_dir, args.mode, [], use_env=True)
    print(f"python {sys.version.split()[0]}, profile {args.mode}\n")

    well = True
    print("what everything is built on")
    state, numpy_version = imports("numpy")
    well &= say("numpy", state, numpy_version)
    state, detail = imports("cv2")
    well &= say("opencv", state, detail)
    if state == OK:
        import cv2

        well &= say("opencv aruco", OK if hasattr(cv2, "aruco") else MISSING,
                    "printed cards need this")
        _two_opencvs()

    print("\nseeing")
    well &= _camera(cfg)
    for model, why in (("face.detector", cfg.face.detector), ("face.embedder", cfg.face.embedder)):
        print(f"  {'-':<8} {model:<{WIDTH}} {why}")
    well &= _models(cfg)

    print("\nhearing and speaking")
    well &= _speech(cfg)

    print("\nsigns")
    well &= _signs(cfg)

    print("\nthe rest")
    well &= _optional(cfg)

    print("\n" + "-" * RULE)
    return _verdict(well)


def _verdict(well: bool) -> int:
    """Said plainly. A row of `missing` under a line claiming everything is
    fine is worse than no tool at all."""
    broken = [what for state, what, _detail in SEEN if state == BROKEN]
    absent = [what for state, what, _detail in SEEN if state == MISSING]

    if broken:
        print(f"BROKEN: {', '.join(broken)}")
        print("Installed but unusable, which is almost always two wheels built "
              "against different numpy versions:")
        print('  pip install "numpy<2" "opencv-python-headless<5"')
    if absent:
        print(f"missing: {', '.join(absent)}")
        print("Each is something this profile asks for and this machine cannot do.")
    if not broken and not absent:
        print("Everything this profile asks for is here.")
    return 0 if well and not absent else 1


def _two_opencvs() -> None:
    """Two OpenCV distributions in one environment overwrite each other's
    files, and which one wins is whichever pip touched last."""
    from importlib import metadata

    found = [name for name in ("opencv-python", "opencv-python-headless",
                               "opencv-contrib-python", "opencv-contrib-python-headless")
             if _installed(metadata, name)]
    if len(found) > 1:
        say("opencv copies", "WARN", f"{len(found)} installed: {', '.join(found)}")


def _installed(metadata, name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def _camera(cfg) -> bool:
    kinds = {source.kind for source in cfg.sources if source.enabled}
    well = True
    if "picamera2" in kinds:
        state, detail = imports("picamera2")
        well &= say("picamera2", state, detail or "the ribbon camera")
    for kind in sorted(kinds - {"picamera2"}):
        say(f"source {kind}", NOT_ASKED, "no import of its own")
    return well


def _models(cfg) -> bool:
    wanted = [
        (Path(cfg.face.model_path), "finding faces"),
        (Path(cfg.face.embedder_model_path), "putting names on faces"),
    ]
    well = True
    for path, why in wanted:
        here = path if path.is_absolute() else ROOT / path
        well &= say(path.name, OK if here.exists() else MISSING, why)
    return well


def _speech(cfg) -> bool:
    well = True
    if cfg.speech.tts.engine == "piper_python":
        state, detail = imports("piper")
        well &= say("piper-tts", state, detail)
        state, detail = imports("onnxruntime")
        well &= say("onnxruntime", state, detail)
    voice = Path(cfg.speech.tts.model_dir) / f"{cfg.speech.tts.voice.get('en', '')}.onnx"
    say(voice.name or "voice", OK if voice.exists() else MISSING, "the English voice")

    if cfg.speech.stt.engine == "vosk":
        state, detail = imports("vosk")
        well &= say("vosk", state, detail)
    if cfg.speech.stt.api_key_env:
        import os

        held = bool(os.environ.get(cfg.speech.stt.api_key_env))
        say(cfg.speech.stt.api_key_env, OK if held else MISSING,
            f"{cfg.speech.stt.engine} hears nothing without it")

    player = cfg.speech.tts.player
    if player not in ("auto", "none"):
        say(player, OK if shutil.which(player) else MISSING, cfg.speech.tts.player_device)
    recorder = cfg.speech.audio.recorder
    if recorder not in ("auto", "none"):
        say(recorder, OK if shutil.which(recorder) else MISSING, "the microphone")
    if shutil.which("amixer") is None and cfg.speech.tts.volume.control == "alsa":
        say("amixer", MISSING, "the volume slider falls back to software")
    return well


def _signs(cfg) -> bool:
    if not cfg.signs.enabled:
        say("signs", NOT_ASKED, "switched off in this profile")
        return True

    from lomas_signs import CARD_READERS, HAND_READERS

    cards = CARD_READERS.create(cfg.signs.cards.reader, cfg.signs.cards)
    say(f"cards: {cfg.signs.cards.reader}", OK if cards.available else MISSING, cards.describe())

    if cfg.signs.hands.reader == "none":
        say("hands", NOT_ASKED, "switched off in this profile")
        return True

    # Only what this reader actually needs. raised_hand needs nothing, which
    # is why the robot uses it.
    if cfg.signs.hands.reader == "mediapipe":
        state, detail = imports("mediapipe")
        say("mediapipe", state, detail)
        model = Path(cfg.signs.hands.model)
        here = model if model.is_absolute() else ROOT / model
        say(model.name, OK if here.exists() else MISSING,
            "python tools/fetch_models.py --hands")

    hands = HAND_READERS.create(cfg.signs.hands.reader, cfg.signs.hands)
    say(f"hands: {cfg.signs.hands.reader}", OK if hands.available else MISSING, hands.describe())
    hands.close()
    return True


def _optional(cfg) -> bool:
    well = True
    if cfg.display.face_screen.enabled and cfg.display.face_screen.surface == "pygame":
        state, detail = imports("pygame")
        well &= say("pygame", state, detail)
    if cfg.hardware.enabled and cfg.hardware.backend == "esp32":
        state, detail = imports("serial")
        well &= say("pyserial", state, detail)
        say(cfg.hardware.port, OK if Path(cfg.hardware.port).exists() else MISSING, "the body")
    if cfg.sync.enabled and cfg.sync.backend == "git":
        say("git", OK if shutil.which("git") else MISSING, "filing traces")
    return well


if __name__ == "__main__":
    raise SystemExit(main())
