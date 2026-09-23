"""The tools, actually run.

They exist to be run on a robot, in a classroom, by somebody who has walked
across a room with a card in their hand - which is the worst possible place
to discover that a function moved or gained an argument. Twice now a tool
has shipped that could not get past its own first ten lines: once a moved
import, once `load_secrets()` called with no path.

Importing them is not enough. These run them, against the mock camera, so
the startup path is exercised end to end with no hardware.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def tool(name: str):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(module, *argv: str) -> int:
    was, sys.argv = sys.argv, [module.__file__, *argv]
    try:
        return module.main()
    finally:
        sys.argv = was


def test_the_sign_checker_runs_against_a_camera() -> None:
    """The mock source is a real frame through the real bus, so this covers
    config, secrets, the readers and the frame loop."""
    assert run(tool("signs_check"), "--mode", "debug", "--seconds", "0.4") == 0


def test_the_sign_checker_measures_on_the_small_copy() -> None:
    """The robot reads a downscaled frame; a tool that measured the full one
    would report a cost the robot never pays."""
    checker = tool("signs_check")

    assert "downscale" in (TOOLS / "signs_check.py").read_text(encoding="utf-8")
    assert checker.distance_m(0, 1280) == 0.0
    assert checker.distance_m(100, 1280) > checker.distance_m(200, 1280)


def test_the_card_printer_makes_a_sheet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOMAS__storage__path", str(tmp_path / "cards.db"))
    out = tmp_path / "sheet.png"

    assert run(tool("make_cards"), "--mode", "debug", "--spare", "3",
               "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0


def test_a_card_sheet_is_readable_back(tmp_path, monkeypatch) -> None:
    """The one thing that matters about a printed card: the robot can read
    it. Printing something the detector cannot find is a silent failure a
    school would discover with forty children holding paper."""
    import cv2

    from lomas_core.config import load
    from lomas_signs import CARD_READERS

    monkeypatch.setenv("LOMAS__storage__path", str(tmp_path / "cards.db"))
    out = tmp_path / "sheet.png"
    run(tool("make_cards"), "--mode", "debug", "--spare", "6", "--out", str(out))

    cfg = load("config", "debug", [], use_env=False).signs.cards
    found = CARD_READERS.create(cfg.reader, cfg).read(cv2.imread(str(out)))

    assert len(found) == 6, "a printed sheet the robot cannot read is a useless sheet"
    assert all(card.upright for card in found)


def test_the_doctor_reports_on_a_machine_that_has_everything() -> None:
    """debug asks for nothing optional, so a working checkout passes it."""
    assert run(tool("doctor"), "--mode", "debug") == 0


def test_the_doctor_does_not_call_a_machine_well_when_it_is_not() -> None:
    """A row of `missing` under a line saying everything is fine is worse
    than no tool at all. The pi profile wants a camera and a body this
    laptop does not have."""
    assert run(tool("doctor"), "--mode", "pi") == 1


def test_the_doctor_reads_the_same_secrets_file_the_robot_does(tmp_path,
                                                               monkeypatch) -> None:
    """It reported GROQ_API_KEY missing on a robot that has it in
    config/secrets.env, which is how a report teaches people to skim it."""
    import os
    import shutil

    config = tmp_path / "config"
    shutil.copytree(ROOT / "config", config)
    (config / "secrets.env").write_text("GROQ_API_KEY=not-a-real-key", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    run(tool("doctor"), "--mode", "debug", "--config-dir", str(config))

    assert os.environ.get("GROQ_API_KEY") == "not-a-real-key"


def test_the_model_fetcher_reports_without_downloading() -> None:
    assert run(tool("fetch_models"), "--check") in (0, 1)


@pytest.mark.parametrize("name", ["audio_check", "face_check", "trace_report"])
def test_the_other_tools_still_offer_help(name: str) -> None:
    """--help exits before anything is opened, so this is only a check that
    the arguments parse. The two above are the ones with a startup path."""
    module = tool(name)
    with pytest.raises(SystemExit) as left:
        run(module, "--help")

    assert left.value.code == 0
