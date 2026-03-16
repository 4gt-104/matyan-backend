from __future__ import annotations

import ast
import copy
import re
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any, override

from RestrictedPython import (
    compile_restricted,
    limited_builtins,
    safe_builtins,
    utility_builtins,
)
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.transformer import RestrictingNodeTransformer

if TYPE_CHECKING:
    from collections.abc import Callable


def safe_import(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    if args and args[0] != "time":
        msg = f"{args[0]} package cannot be imported."
        raise ImportError(msg)
    return __import__(*args, **kwargs)


extra_builtins = {
    "datetime": datetime,
    "timedelta": timedelta,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "any": any,
    "all": all,
    "__import__": safe_import,
}

builtins = safe_builtins.copy()
builtins.update(utility_builtins)
builtins.update(limited_builtins)
builtins.update(extra_builtins)


def safer_getattr(
    obj: object,
    name: str,
    default: object = None,
    getattr: Callable[..., object] = getattr,  # noqa: A002
) -> object:
    """Getattr implementation which prevents using format on string objects.

    format() is considered harmful:
    http://lucumr.pocoo.org/2016/12/29/careful-with-str-format/

    """
    if name == "format" and isinstance(obj, str):
        msg = f"Using format() on a {obj.__class__.__name__} is not safe."
        raise NotImplementedError(msg)
    if name[0] == "_":
        msg = f'"{name}" is an invalid attribute name because it starts with "_"'
        raise AttributeError(msg)
    return getattr(obj, name, default)


restricted_globals = {
    "__builtins__": builtins,
    "_getattr_": safer_getattr,
    "_write_": full_write_guard,
    "_getiter_": iter,
    "_getitem_": default_guarded_getitem,
    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    "_unpack_sequence_": guarded_unpack_sequence,
}

from loguru import logger  # noqa: E402

if TYPE_CHECKING:
    from types import CodeType


# ---------------------------------------------------------------------------
# Query string helpers (kept for backward-compat; prefer prepare_query)
# ---------------------------------------------------------------------------


class Query:
    __slots__ = ("__weakref__", "expr")

    def __init__(self, expr: str) -> None:
        self.expr = expr

    @abstractmethod
    def check(self, **params: Any) -> bool: ...  # noqa: ANN401

    def __call__(self, **params: Any) -> bool:  # noqa: ANN401
        return self.check(**params)


@lru_cache(maxsize=100)
def compile_checker(expr: str) -> CodeType:
    source_code = expr
    return compile_restricted(source_code, filename="<inline code>", mode="eval")


def syntax_error_check(expr: str) -> None:
    if not expr:
        return
    expr = strip_query(expr)
    try:
        compile_restricted(expr, filename="<inline code>", mode="eval")
    except SyntaxError:
        compile(expr, filename="<inline code>", mode="eval")


@lru_cache(maxsize=100)
def strip_query(query: str) -> str:
    stripped_query = query.strip()
    if query.lower().startswith("select"):
        try:
            stripped_query = re.split("if", query, maxsplit=1, flags=re.IGNORECASE)[1]
        except IndexError:
            stripped_query = ""

    if stripped_query:
        stripped_query = f"({stripped_query.strip()})"

    return stripped_query


@lru_cache(maxsize=100)
def query_add_default_expr(query: str) -> str:
    default_expression = "run.is_archived == False"
    if not query:
        return default_expression
    if "run.is_archived" not in query and "run.archived" not in query:
        return f"{default_expression} and {query}"
    return query


# ---------------------------------------------------------------------------
# AST-based preprocessing pipeline
# ---------------------------------------------------------------------------

_DEFAULT_EXPR_SRC = "(run.is_archived == False)"
_DEFAULT_AST: ast.Expression = compile(_DEFAULT_EXPR_SRC, "<query>", "eval", ast.PyCF_ONLY_AST)  # type: ignore[assignment]


def _default_expression_ast() -> ast.Expression:
    """Return a cached AST for the default ``run.is_archived == False`` predicate."""
    return _DEFAULT_AST


def _is_archived_predicate(node: ast.AST) -> bool:
    """Return True if *node* is a comparison referencing ``run.archived`` or ``run.is_archived``."""
    if not isinstance(node, ast.Compare):
        return False
    for n in (node.left, *node.comparators):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "run"
            and n.attr in ("archived", "is_archived")
        ):
            return True
    return False


class _DatetimeRewriter(ast.NodeTransformer):
    """Replace safe ``datetime(...)`` calls with ``Constant(utc_timestamp)``.

    Also records whether the tree contains an archived predicate so the
    caller can decide whether to attach the default filter.
    """

    def __init__(self, tz_offset: int) -> None:
        super().__init__()
        self._tz_offset = tz_offset
        self.has_archived = False

    @override
    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        if _is_archived_predicate(node):
            self.has_archived = True
        self.generic_visit(node)
        return node

    @override
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        if not (isinstance(node.func, ast.Name) and node.func.id == "datetime"):
            return node
        if node.keywords:
            return node
        if not all(isinstance(a, ast.Constant) for a in node.args):
            return node

        try:
            args = tuple(a.value for a in node.args)  # type: ignore[union-attr]
            naive_dt = datetime(*args)  # noqa: DTZ001
            user_tz = timezone(timedelta(minutes=self._tz_offset))
            aware_dt = naive_dt.replace(tzinfo=user_tz)
            utc_ts = aware_dt.timestamp()
        except (TypeError, ValueError, OverflowError):
            return node

        return ast.copy_location(ast.Constant(value=utc_ts), node)


class _ChainedCompareRewriter(ast.NodeTransformer):
    """Split chained comparisons into AND of single comparisons.

    ``a <= b < c`` becomes ``(a <= b) and (b < c)``.  This is
    semantics-preserving (Python evaluates chained compares identically)
    and lets the planner handle each half as a single-op Compare node.
    """

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) <= 1:
            return node
        pairs: list[ast.expr] = []
        left = node.left
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            pairs.append(ast.Compare(left=left, ops=[op], comparators=[comparator]))
            left = comparator
        return ast.BoolOp(op=ast.And(), values=pairs)


def _transform_query_ast(tree: ast.Expression, tz_offset: int) -> ast.Expression:
    """Rewrite ``datetime(...)`` to timestamps and attach the default filter if needed."""
    rewriter = _DatetimeRewriter(tz_offset)
    tree = rewriter.visit(tree)
    tree = _ChainedCompareRewriter().visit(tree)

    if not rewriter.has_archived:
        default_body = _default_expression_ast().body
        tree.body = ast.BoolOp(op=ast.And(), values=[default_body, tree.body])

    ast.fix_missing_locations(tree)
    return tree


def prepare_query(raw_query: str, tz_offset: int = 0) -> ast.Expression:
    """Preprocess a MatyanQL query into a ready-to-use AST.

    1. Strip the query string (handle ``SELECT ... IF`` prefix).
    2. If empty, return the default-only AST (fast path).
    3. Parse with ``compile(..., PyCF_ONLY_AST)``.
    4. Single walk: rewrite ``datetime(...)`` to timestamps and maybe add
       the default ``run.is_archived == False`` predicate.

    Returns an ``ast.Expression``.
    Raises ``SyntaxError`` if the query cannot be parsed (callers should
    translate this to an HTTP error for the UI).
    """
    stripped = strip_query(raw_query)

    if not stripped:
        return _default_expression_ast()

    tree: ast.Expression = compile(stripped, "<query>", "eval", ast.PyCF_ONLY_AST)  # type: ignore[assignment]
    return _transform_query_ast(tree, tz_offset)


# ---------------------------------------------------------------------------
# RestrictedPythonQuery
# ---------------------------------------------------------------------------


def _compile_restricted_ast(tree: ast.Expression) -> CodeType:
    """Apply RestrictedPython's security transformer to an AST and compile.

    Equivalent to ``compile_restricted(source, ..., mode="eval")`` but
    accepts an ``ast.Expression`` directly — no string round-trip.

    Deep-copies the tree first because ``RestrictingNodeTransformer``
    mutates in place (e.g. rewrites attribute access to ``_getattr_``
    calls), and the input may share nodes with cached constants like
    ``_DEFAULT_AST``.
    """
    tree = copy.deepcopy(tree)
    errors: list[str] = []
    warnings: list[str] = []
    used_names: dict[str, bool] = {}
    RestrictingNodeTransformer(errors, warnings, used_names).visit(tree)
    if errors:
        raise SyntaxError(errors)
    return compile(tree, "<query>", "eval")


class RestrictedPythonQuery(Query):
    __slots__ = ("_checker", "run_metadata_cache")

    allowed_params = frozenset(
        (
            "run",
            "metric",
            "images",
            "audios",
            "distributions",
            "figures",
            "texts",
        ),
    )

    def __init__(self, query: str = "", *, prepared_ast: ast.Expression | None = None) -> None:
        if prepared_ast is not None:
            expr_str = ast.unparse(prepared_ast)
            super().__init__(expr=expr_str)
            self._checker = _compile_restricted_ast(prepared_ast)
        else:
            stripped_query = strip_query(query)
            expr = query_add_default_expr(stripped_query)
            super().__init__(expr=expr)
            self._checker = compile_checker(expr)
        self.run_metadata_cache = None

    def __bool__(self) -> bool:
        return bool(self.expr)

    def check(self, **params: Any) -> bool:  # noqa: ANN401
        assert set(params.keys()).issubset(self.allowed_params)

        try:
            namespace = dict(**params, **restricted_globals)
            return eval(self._checker, restricted_globals, namespace)  # noqa: S307
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug(
                "MatyanQL query evaluation failed (expr={})",
                self.expr,
            )
            return False
