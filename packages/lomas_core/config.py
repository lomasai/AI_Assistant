from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import ValidationError

from lomas_core.errors import ConfigError
from lomas_core.schema import Config

ENV_PREFIX = "LOMAS__"
ENV_SEPARATOR = "__"

# The only spellings allowed to become None or a bool. See _coerce.
EXPLICIT_SCALARS = {"true", "false", "~"}

# Layers, lowest precedence first. Anything later wins.
#   1. default.yaml          every key, documented
#   2. profiles/<mode>.yaml  per-mode overrides
#   3. site.yaml             per-school deployment, gitignored
#   4. environment           LOMAS__runtime__log_level=DEBUG
#   5. --set                 runtime.log_level=DEBUG


def load(
    config_dir: str | Path,
    mode: str,
    overrides: Iterable[str] = (),
    use_env: bool = True,
) -> Config:
    root = Path(config_dir)
    tree: dict[str, Any] = {}

    default_file = root / "default.yaml"
    if not default_file.exists():
        raise ConfigError(f"missing {default_file} - it must list every key")
    _merge(tree, _read_yaml(default_file))

    profile_file = root / "profiles" / f"{mode}.yaml"
    if not profile_file.exists():
        available = sorted(p.stem for p in (root / "profiles").glob("*.yaml"))
        raise ConfigError(f"unknown mode '{mode}'. Available: {', '.join(available) or 'none'}")
    _merge(tree, _read_yaml(profile_file))

    site_file = root / "site.yaml"
    if site_file.exists():
        _merge(tree, _read_yaml(site_file))

    if use_env:
        _merge(tree, _from_env())

    for override in overrides:
        _apply_override(tree, override)

    try:
        return Config.model_validate(tree)
    except ValidationError as exc:
        raise ConfigError(_explain(exc)) from None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return loaded


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge(current, value)
        else:
            base[key] = value
    return base


def _from_env() -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split(ENV_SEPARATOR)
        _assign(tree, path, _coerce(raw))
    return tree


def _apply_override(tree: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"--set expects key.path=value, got '{override}'")
    dotted, raw = override.split("=", 1)
    path = [part for part in dotted.strip().split(".") if part]
    if not path:
        raise ConfigError(f"--set has an empty key: '{override}'")
    _assign(tree, path, _coerce(raw))


def _assign(tree: dict[str, Any], path: list[str], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[path[-1]] = value


def _coerce(raw: str) -> Any:
    """Read a shell string as YAML so ints, bools and lists survive.

    With one correction. YAML 1.1 reads `null`, `no`, `yes`, `on` and `off`
    as None or booleans, which is wrong for config values far more often than
    it is right: `null` is the name of a real TTS engine here, and `no` is the
    ISO code for Norwegian. Only the unambiguous spellings coerce; everything
    else stays the string the user typed. Use `~` for an explicit null.
    """
    text = raw.strip()
    if not text:
        return ""

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return raw

    if (value is None or isinstance(value, bool)) and text.lower() not in EXPLICIT_SCALARS:
        return text
    return value


def _explain(exc: ValidationError) -> str:
    lines = ["configuration is invalid:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
