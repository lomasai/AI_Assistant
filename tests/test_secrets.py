"""Keys live in the environment, never in the config tree.

That separation is the whole point of this file. `--show-config` prints the
resolved config and `/api/debug/config` serves it, so a secret that reaches
Config is a secret on a web page. What config holds is the *name* of a
variable; what the secrets file holds is the value.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lomas_core.clock import FakeClock
from lomas_core.config import load
from lomas_core.secrets import SECRETS_FILE, is_setting, load_secrets

from app import container, seed
from app.web.server import create_app

PRETEND = "gsk_a_key_that_must_never_be_printed"

HEADLESS = [
    "storage.backend=memory",
    "vision.pipeline.enabled=false",
    "hardware.enabled=false",
    "speech.tts.engine=null",
    "speech.stt.engine=keyboard",
    "speech.wake.engine=keyboard",
    "llm.provider=offline",
]


@pytest.fixture
def secrets_file(tmp_path: Path) -> Path:
    path = tmp_path / SECRETS_FILE
    path.write_text(
        "# a comment\n"
        "\n"
        f"GROQ_API_KEY={PRETEND}\n"
        'ANTHROPIC_API_KEY="sk-ant-quoted"\n'
        "export OPENAI_API_KEY=sk-with-export-prefix\n"
        "MALFORMED_LINE_NO_EQUALS\n"
        "EMPTY_VALUE=\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ["GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "EMPTY_VALUE"]:
        monkeypatch.delenv(name, raising=False)


# --- reading the file -----------------------------------------------------


def test_keys_reach_the_environment(secrets_file: Path) -> None:
    loaded = load_secrets(secrets_file)

    assert set(loaded) == {"GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    assert os.environ["GROQ_API_KEY"] == PRETEND


def test_quotes_and_an_export_prefix_are_tolerated(secrets_file: Path) -> None:
    """People paste these files from a shell script and from a dashboard.
    Both spellings have to work or the first thing anybody meets is a bug."""
    load_secrets(secrets_file)

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted"
    assert os.environ["OPENAI_API_KEY"] == "sk-with-export-prefix"


def test_blank_and_malformed_lines_are_skipped(secrets_file: Path) -> None:
    loaded = load_secrets(secrets_file)

    assert "EMPTY_VALUE" not in loaded
    assert "MALFORMED_LINE_NO_EQUALS" not in loaded


def test_an_exported_variable_wins(secrets_file: Path, monkeypatch) -> None:
    """A real deployment passes secrets in properly. The file is for the
    bench, and must not quietly overwrite the real thing."""
    monkeypatch.setenv("GROQ_API_KEY", "the-real-one")
    loaded = load_secrets(secrets_file)

    assert os.environ["GROQ_API_KEY"] == "the-real-one"
    assert "GROQ_API_KEY" not in loaded


def test_override_is_available_but_not_the_default(secrets_file: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "stale")
    load_secrets(secrets_file, override=True)

    assert os.environ["GROQ_API_KEY"] == PRETEND


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """Most people never make one. Running without it is the normal case."""
    assert load_secrets(tmp_path / "nothing-here.env") == []


def test_it_returns_names_and_never_values(secrets_file: Path) -> None:
    """A log line naming a key is fine. One carrying its value has leaked it."""
    loaded = load_secrets(secrets_file)

    assert all(PRETEND not in entry for entry in loaded)
    assert PRETEND not in " ".join(loaded)


# --- the guarantee --------------------------------------------------------


def test_a_key_never_reaches_the_resolved_config(secrets_file: Path) -> None:
    load_secrets(secrets_file)
    cfg = load("config", "debug", HEADLESS, use_env=True)

    assert PRETEND not in json.dumps(cfg.model_dump())
    assert cfg.llm.endpoints["groq"].api_key_env == "GROQ_API_KEY"


def test_a_key_never_reaches_the_diagnostics_page(secrets_file: Path) -> None:
    """The endpoint that would leak it. /api/debug/config serves the whole
    resolved config, which is exactly why secrets are kept out of it."""
    load_secrets(secrets_file)
    cfg = load("config", "debug", HEADLESS, use_env=True)
    system = container.build(cfg, clock=FakeClock(), bus=container.event_bus(cfg))
    try:
        seed.demo_class(system)
        with TestClient(create_app(system)) as client:
            for path in ["/api/debug/config", "/api/debug/metrics", "/api/debug/plugins",
                         "/api/state", "/api/display"]:
                body = client.get(path)
                assert PRETEND not in body.text, f"{path} leaked the key"
    finally:
        system.close()


def test_the_env_prefix_cannot_smuggle_a_key_into_config(monkeypatch) -> None:
    """LOMAS__ variables do become config. A key must not be reachable that
    way either, so nothing in the schema is named to accept one."""
    monkeypatch.setenv("GROQ_API_KEY", PRETEND)
    cfg = load("config", "debug", HEADLESS, use_env=True)

    assert PRETEND not in json.dumps(cfg.model_dump())


def test_the_example_file_carries_no_real_key() -> None:
    """It is committed. Somebody will eventually paste a key into it."""
    example = Path("config/secrets.env.example").read_text(encoding="utf-8")

    for line in example.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        assert not value.strip(), f"{name.strip()} has a value in the committed example"


def test_the_real_secrets_file_is_ignored_by_git() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "config/secrets.env" in ignored


# --- the same file also carries settings ----------------------------------


def test_a_setting_in_the_file_reaches_the_config(tmp_path: Path, monkeypatch) -> None:
    """One file for the whole machine. Provider, model and engine are not
    secrets, but nobody wants to edit two files to change a model."""
    for name in ["LOMAS__llm__provider", "LOMAS__llm__model",
                 "LOMAS__llm__endpoints__groq__model", "LOMAS__speech__tts__engine"]:
        monkeypatch.delenv(name, raising=False)

    path = tmp_path / SECRETS_FILE
    path.write_text(
        "\n".join([
            f"GROQ_API_KEY={PRETEND}",
            "LOMAS__llm__provider=groq",
            "LOMAS__llm__model=llama-3.1-8b-instant",
            "LOMAS__llm__endpoints__groq__model=llama-3.1-8b-instant",
            "LOMAS__speech__tts__engine=gtts",
        ]),
        encoding="utf-8",
    )
    load_secrets(path)
    cfg = load("config", "debug", ["storage.backend=memory"], use_env=True)

    assert cfg.llm.provider == "groq"
    assert cfg.llm.model == "llama-3.1-8b-instant"
    assert cfg.llm.endpoints["groq"].model == "llama-3.1-8b-instant"
    assert cfg.speech.tts.engine == "gtts"
    assert PRETEND not in json.dumps(cfg.model_dump()), "the key came along with them"


def test_a_setting_is_printable_and_a_key_is_not() -> None:
    """A model name on a log line is fine. A key on one has leaked it."""
    assert is_setting("LOMAS__llm__provider")
    assert not is_setting("GROQ_API_KEY")
    assert not is_setting("ANTHROPIC_API_KEY")


def test_every_commented_setting_in_the_example_names_a_real_config_key() -> None:
    """A documented key that does not exist is worse than none: it looks like
    it works and quietly does nothing."""
    cfg = load("config", "debug", ["storage.backend=memory"], use_env=False)
    example = Path("config/secrets.env.example").read_text(encoding="utf-8")

    for line in example.splitlines():
        entry = line.strip().lstrip("#").strip()
        if not entry.startswith("LOMAS__") or "=" not in entry:
            continue

        path = entry.split("=")[0].removeprefix("LOMAS__").split("__")
        node = cfg
        for part in path:
            node = node[part] if isinstance(node, dict) else getattr(node, part, None)
            if isinstance(node, dict) and part in node:
                node = node[part]
            assert node is not None or part in ("en", "hi"), f"{entry} names nothing real"
