"""Rules 1 and 2: packages never import siblings, and never import the app.

This is what keeps each package droppable into another project. If it fails,
the fix is an event or an argument, never a new import.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
TOOLS = ROOT / "tools"
FOUNDATION = "lomas_core"
APP = "app"


def package_names() -> set[str]:
    return {p.name for p in PACKAGES.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}


def source_files() -> list[Path]:
    return sorted(PACKAGES.rglob("*.py"))


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(PACKAGES)))
def test_package_does_not_reach_sideways(path: Path) -> None:
    owner = path.relative_to(PACKAGES).parts[0]
    siblings = package_names() - {owner, FOUNDATION}

    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    roots = imported_roots(tree)

    offending = roots & siblings
    assert not offending, (
        f"{path.relative_to(ROOT)} imports sibling package(s) {sorted(offending)}. "
        f"Packages may import {FOUNDATION} and third-party libraries only."
    )

    assert APP not in roots, (
        f"{path.relative_to(ROOT)} imports the app. Dependencies point one way: "
        "app -> packages -> core."
    )


def test_foundation_depends_on_nothing_of_ours() -> None:
    others = package_names() - {FOUNDATION}
    for path in (PACKAGES / FOUNDATION).rglob("*.py"):
        roots = imported_roots(ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)))
        leaked = roots & (others | {APP})
        assert not leaked, f"{path.name} pulls in {sorted(leaked)}; core must stay domain-free"


@pytest.mark.parametrize("path", sorted(TOOLS.glob("*.py")),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_every_tool_imports(path: Path) -> None:
    """A tool is only ever run on the robot, which is the worst place to
    find out that one of its imports moved. `signs_check.py` shipped asking
    lomas_vision.source for something that lives in the package root, and
    the first anybody knew was a traceback on the Pi.
    """
    spec = importlib.util.spec_from_file_location(f"tool_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
