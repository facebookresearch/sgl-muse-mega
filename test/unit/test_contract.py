"""The SGLang names this package reads, pinned on CPU against the installed SGLang.

The package plugs into SGLang from outside, so it reads SGLang by name: its imports, the loaded model's modules and
weights, the KV pools, the DFlash2 worker's state. Every module that reads SGLang lists those names in
``SGLANG_READS`` ({SGLang file pattern: names}); a refactor that renames one fails here, naming the file to repair,
instead of failing at a GPU server's startup. The check is static (AST): a name is provided when the file defines it,
assigns it (``x.name = ...`` too), imports it, declares it as a parameter (``f(p)``) or holds it as a string literal
(``"name"``, for attributes set through setattr-like calls). Run it against each SGLang the package supports.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest

sglang = pytest.importorskip("sglang")

import sgl_muse_mega  # noqa: E402

ROOT = Path(sglang.__file__).resolve().parent
PACKAGE = Path(sgl_muse_mega.__file__).resolve().parent


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


@lru_cache(maxsize=None)
def _provided(path: Path) -> frozenset:
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                names.update(
                    f"{node.name}({p.arg})"
                    for p in a.posonlyargs + a.args + a.kwonlyargs
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            ):
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        names.add(t.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(f'"{node.value}"')
    return frozenset(names)


def _declared(path: Path) -> dict:
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def _sglang_imports(path: Path) -> set:
    return {
        a.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("sglang.")
        for a in node.names
    }


READERS = [
    p for p in sorted(PACKAGE.rglob("*.py")) if _declared(p) or _sglang_imports(p)
]


def _id(path: Path) -> str:
    return str(path.relative_to(PACKAGE.parent))


def test_the_readers_are_found():
    assert {_id(p) for p in READERS} >= {
        "sgl_muse_mega/plugin.py",
        "sgl_muse_mega/muse_spark/runtime/iteration.py",
    }


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_every_sglang_import_is_declared(path):
    declared = {n for names in _declared(path).values() for n in names}
    undeclared = _sglang_imports(path) - declared
    assert not undeclared, f"{_id(path)}: declare {sorted(undeclared)} in SGLANG_READS"


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_sglang_provides_what_the_package_reads(path):
    missing = []
    for pattern, names in _declared(path).items():
        files = sorted(ROOT.glob(pattern))
        assert files, f"{_id(path)}: no SGLang file matches {pattern}"
        provided = frozenset().union(*map(_provided, files))
        missing += [f"{pattern}: {n}" for n in names if n not in provided]
    assert (
        not missing
    ), f"SGLang no longer provides what {_id(path)} reads:\n  " + "\n  ".join(missing)
