"""The four config layers resolve last-wins, and a typo fails at startup."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lomas_core.config import load
from lomas_core.errors import ConfigError


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "profiles").mkdir()
    (tmp_path / "default.yaml").write_text(
        textwrap.dedent(
            """
            runtime:
              mode: user
              log_level: INFO
              locale: en
            tenancy:
              org_id: from-default
              school_id: from-default
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "profiles" / "user.yaml").write_text(
        "runtime:\n  log_level: WARNING\n", encoding="utf-8"
    )
    (tmp_path / "profiles" / "debug.yaml").write_text(
        "runtime:\n  mode: debug\n  log_level: DEBUG\n", encoding="utf-8"
    )
    return tmp_path


def test_default_layer_applies(config_dir: Path) -> None:
    cfg = load(config_dir, "user", use_env=False)
    assert cfg.tenancy.org_id == "from-default"
    assert cfg.runtime.locale == "en"


def test_profile_beats_default(config_dir: Path) -> None:
    assert load(config_dir, "user", use_env=False).runtime.log_level == "WARNING"
    assert load(config_dir, "debug", use_env=False).runtime.mode == "debug"


def test_site_beats_profile(config_dir: Path) -> None:
    (config_dir / "site.yaml").write_text(
        "runtime:\n  log_level: ERROR\ntenancy:\n  org_id: from-site\n", encoding="utf-8"
    )
    cfg = load(config_dir, "user", use_env=False)
    assert cfg.runtime.log_level == "ERROR"
    assert cfg.tenancy.org_id == "from-site"
    assert cfg.tenancy.school_id == "from-default", "unrelated keys survive the overlay"


def test_env_beats_site(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (config_dir / "site.yaml").write_text("runtime:\n  log_level: ERROR\n", encoding="utf-8")
    monkeypatch.setenv("LOMAS__runtime__log_level", "DEBUG")
    assert load(config_dir, "user").runtime.log_level == "DEBUG"


def test_cli_override_beats_everything(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (config_dir / "site.yaml").write_text("runtime:\n  log_level: ERROR\n", encoding="utf-8")
    monkeypatch.setenv("LOMAS__runtime__log_level", "DEBUG")
    cfg = load(config_dir, "user", overrides=["runtime.log_level=WARNING"])
    assert cfg.runtime.log_level == "WARNING"


def test_override_keeps_yaml_types(config_dir: Path) -> None:
    cfg = load(config_dir, "user", ["runtime.event_replay_size=64"], use_env=False)
    assert cfg.runtime.event_replay_size == 64
    assert isinstance(cfg.runtime.event_replay_size, int)


def test_unknown_key_is_refused(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="runtime.colour"):
        load(config_dir, "user", ["runtime.colour=blue"], use_env=False)


def test_bad_value_names_the_key(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="runtime.log_level"):
        load(config_dir, "user", ["runtime.log_level=LOUD"], use_env=False)


def test_unknown_mode_lists_the_options(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="debug, user"):
        load(config_dir, "pi", use_env=False)


def test_debug_writes_to_scratch_tenant(config_dir: Path) -> None:
    live = load(config_dir, "user", use_env=False)
    bench = load(config_dir, "debug", use_env=False)
    assert live.active_org_id == "from-default"
    assert bench.active_org_id == bench.tenancy.scratch_org_id
