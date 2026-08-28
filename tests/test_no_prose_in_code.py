"""No prompt text in any .py file.

The moment an English sentence written for a model is inlined into Python,
adding Hindi stops being content work and becomes code work - and the
multilingual promise quietly dies six months later when nobody remembers why.

Error messages are prose too, and they are legitimate, so the rule is a
length limit rather than a ban on words. If a genuinely long string belongs
in code, mark the line `# prose-exempt`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "packages", ROOT / "app"]
EXEMPT_LINE = "prose-exempt"

# Long enough to hold a real error message with a remedy, far shorter than any
# useful system prompt.
MAX_STRING = 240


def source_files() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if root.exists():
            found.extend(root.rglob("*.py"))
    return sorted(found)


def long_strings(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    tree = ast.parse("\n".join(lines), filename=str(path))
    hits: list[str] = []

    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node in docstrings:
            continue  # documentation for us, not instructions for a model
        if len(node.value) <= MAX_STRING:
            continue
        if EXEMPT_LINE in lines[node.lineno - 1]:
            continue
        hits.append(f"line {node.lineno}: {len(node.value)} character string")

    return hits


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_prompt_text_in_code(path: Path) -> None:
    hits = long_strings(path)
    assert not hits, (
        f"{path.relative_to(ROOT)} holds prose that belongs in config/prompts:\n  "
        + "\n  ".join(hits)
        + f"\nPrompts live in config/prompts/<language>/. Strings over {MAX_STRING} "
        "characters in code cannot be translated."
    )


def test_the_prompt_directory_is_where_prose_actually_lives() -> None:
    """A sanity check on the rule: the prompts really are long enough that the
    limit above would have caught them."""
    prompts = ROOT / "config" / "prompts" / "en"
    longest = max(len(p.read_text(encoding="utf-8")) for p in prompts.glob("*.yaml"))
    assert longest > MAX_STRING
