"""Rule 5: no literal in a decision position.

Thresholds, timings and limits belong in config so behaviour can be tuned
without a code change. This flags bare constants used in comparisons or as
default arguments. Put the value in schema.py, or add `# tuning-exempt` to
the line if it genuinely is not a tunable.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "packages", ROOT / "app"]

# schema.py is where defaults are supposed to live; contracts carry no logic.
EXEMPT_FILES = {"schema.py", "contracts.py", "__init__.py"}
EXEMPT_LINE = "tuning-exempt"

# Identity, emptiness and off-by-one are structure, not tuning.
ALLOWED_NUMBERS = {0, 1, -1, 0.0, 1.0, 2}
BASE_STRINGS = {"", "utf-8", "*", ".", "/", "=", "__main__"}
SCHEMA_FILE = ROOT / "packages" / "lomas_core" / "schema.py"


def schema_vocabulary() -> set[str]:
    """Strings declared in a Literal[...] in schema.py are config vocabulary.

    Comparing against one is structure, not a hidden tunable, so the scanner
    should not flag it. Anything not declared there still gets caught.
    """
    if not SCHEMA_FILE.exists():
        return set()

    words: set[str] = set()
    tree = ast.parse(SCHEMA_FILE.read_text(encoding="utf-8-sig"), filename=str(SCHEMA_FILE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        base = node.value
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if name != "Literal":
            continue
        members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        words.update(m.value for m in members if isinstance(m, ast.Constant) and isinstance(m.value, str))
    return words


ALLOWED_STRINGS = BASE_STRINGS | schema_vocabulary()


def source_files() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if root.exists():
            found.extend(p for p in root.rglob("*.py") if p.name not in EXEMPT_FILES)
    return sorted(found)


def is_bare_constant(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value not in ALLOWED_NUMBERS
    if isinstance(value, str):
        return value not in ALLOWED_STRINGS and len(value) > 1
    return False


def findings(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    tree = ast.parse("\n".join(lines), filename=str(path))
    hits: list[str] = []

    def exempt(node: ast.AST) -> bool:
        line = getattr(node, "lineno", 0)
        return bool(line) and EXEMPT_LINE in lines[line - 1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and not exempt(node):
            for side in [node.left, *node.comparators]:
                if is_bare_constant(side):
                    hits.append(f"line {side.lineno}: comparison against {side.value!r}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not exempt(node):
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d)]
            for default in defaults:
                if is_bare_constant(default):
                    hits.append(f"line {default.lineno}: default argument {default.value!r}")

    return hits


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_tunable_literals(path: Path) -> None:
    hits = findings(path)
    assert not hits, (
        f"{path.relative_to(ROOT)} holds tunable values in code:\n  "
        + "\n  ".join(hits)
        + "\nMove them to config/schema.py, or mark the line `# tuning-exempt`."
    )
