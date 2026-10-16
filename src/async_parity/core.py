"""Find sync/async method pairs whose bodies diverge beyond the async plumbing.

Libraries often keep ``run`` and ``run_async`` (or ``chat`` and ``achat``) side by side, and a fix
applied to one of them tends not to reach the other. This module normalizes both bodies (drops
``await``, ``async for``/``async with``, docstrings, and the a/a_/async_/_async/Async naming of
counterparts) and diffs what is left. What remains is usually either a deliberate difference or a
parity bug.
"""

from __future__ import annotations

import ast
import difflib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Names a coroutine function commonly takes relative to its sync twin.
PARTNER_PATTERNS = ("{}_async", "a{}", "a_{}", "async_{}")

# Prefixes/suffixes that mark the async counterpart of an identifier.
# Leading underscores are kept, so `_aclient` maps to `_client`.
_ASYNC_FORMS = (
    (re.compile(r"^(_*)(.+)_async$"), r"\1\2"),
    (re.compile(r"^(_*)async_(.+)$"), r"\1\2"),
    (re.compile(r"^(_*)a_(.+)$"), r"\1\2"),
    (re.compile(r"^(_*)Async([A-Z].*)$"), r"\1\2"),
    (re.compile(r"^(_*)a([a-z_].*)$"), r"\1\2"),
)


@dataclass(frozen=True)
class Divergence:
    path: Path
    cls: str
    sync_name: str
    async_name: str
    diff: list[str]

    @property
    def sync_only(self) -> list[str]:
        """Lines present in the sync body with no counterpart in the async body."""
        return [line[1:] for line in self.diff if line.startswith("-") and not line.startswith("---")]

    @property
    def async_only(self) -> list[str]:
        return [line[1:] for line in self.diff if line.startswith("+") and not line.startswith("+++")]

    def header(self) -> str:
        return f"{self.path}:{self.cls}.{self.sync_name} vs {self.async_name}"


class _StripAsync(ast.NodeTransformer):
    """Turns a coroutine body into the shape its sync twin would have."""

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return self.visit(node.value)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        return self.generic_visit(ast.For(**{f: getattr(node, f) for f in node._fields}))

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        return self.generic_visit(ast.With(**{f: getattr(node, f) for f in node._fields}))

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # docstring or bare string
        return self.generic_visit(node)


class _Rename(ast.NodeTransformer):
    """Maps async identifiers onto the sync names they stand in for."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = self.mapping.get(node.attr, node.attr)
        return node


def _identifiers(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
    return names


def _rename_map(sync_names: set[str], async_names: set[str]) -> dict[str, str]:
    """Pair async identifiers with sync ones, only when the sync name is actually used.

    Requiring the counterpart to appear in the sync body keeps ordinary names that happen to start
    with "a" (``append``, ``args``) from being mangled.
    """
    mapping: dict[str, str] = {}
    for name in async_names - sync_names:
        for pattern, repl in _ASYNC_FORMS:
            candidate = pattern.sub(repl, name)
            if candidate != name and candidate in sync_names:
                mapping[name] = candidate
                break
    return mapping


def _body_lines(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [ast.unparse(stmt) for stmt in fn.body]


def normalized_pair(
    sync_fn: ast.FunctionDef, async_fn: ast.AsyncFunctionDef | ast.FunctionDef
) -> tuple[list[str], list[str]]:
    """Return both bodies as comparable source lines."""
    sync_copy = _StripAsync().visit(ast.parse(ast.unparse(sync_fn)).body[0])
    async_copy = _StripAsync().visit(ast.parse(ast.unparse(async_fn)).body[0])
    mapping = _rename_map(_identifiers(sync_copy), _identifiers(async_copy))
    async_copy = _Rename(mapping).visit(async_copy)
    ast.fix_missing_locations(sync_copy)
    ast.fix_missing_locations(async_copy)

    def lines(fn: ast.AST) -> list[str]:
        out: list[str] = []
        for chunk in _body_lines(fn):  # type: ignore[arg-type]
            out.extend(chunk.splitlines())
        return out

    return lines(sync_copy), lines(async_copy)


def _pairs(cls: ast.ClassDef) -> Iterator[tuple[ast.FunctionDef, ast.AsyncFunctionDef | ast.FunctionDef]]:
    methods = {f.name: f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, fn in methods.items():
        if name.startswith("__") or isinstance(fn, ast.AsyncFunctionDef):
            continue
        for pattern in PARTNER_PATTERNS:
            partner = methods.get(pattern.format(name))
            if partner is not None and partner is not fn:
                yield fn, partner
                break


def scan_source(source: str, path: Path = Path("<string>")) -> Iterator[Divergence]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for sync_fn, async_fn in _pairs(cls):
            a, b = normalized_pair(sync_fn, async_fn)
            diff = list(difflib.unified_diff(a, b, sync_fn.name, async_fn.name, n=1, lineterm=""))
            if diff:
                yield Divergence(path, cls.name, sync_fn.name, async_fn.name, diff)


def scan_path(root: Path, exclude: tuple[str, ...] = ()) -> Iterator[Divergence]:
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in files:
        if any(part in exclude for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        yield from scan_source(source, path)
