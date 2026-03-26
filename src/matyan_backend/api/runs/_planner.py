"""AST-based multi-predicate query planner for MatyanQL.

Accepts a *prepared* ``ast.Expression`` (produced by
``_query.prepare_query``) and recursively evaluates index-backed
predicates.

- **AND** (``ast.BoolOp(And)``): intersect candidate sets from each branch.
  When some branches are indexed and others are not, the intersection of
  the indexed branches is returned with ``exact=False`` so the caller
  knows to run ``RestrictedPythonQuery.check()`` on each candidate.
- **OR** (``ast.BoolOp(Or)``): union candidate sets from each branch.
  If any branch is unindexed, the whole OR falls back to ``None``.
- **Compare**: match known predicate shapes and resolve via FDB indexes.

``plan_query`` returns a ``PlanResult(candidates, exact)`` where:

- ``candidates is None`` → full scan (lazy path).
- ``candidates`` is a ``list[str]`` and ``exact is True`` → every hash
  satisfies the full query.
- ``candidates`` is a ``list[str]`` and ``exact is False`` → candidates
  are a superset; the caller must filter with ``check(run=...)``.

Queries referencing unindexed attributes (``run.name``,
``run.description``) in an AND with indexed predicates now return the
indexed subset with ``exact=False`` instead of falling back to full scan.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, NamedTuple

from matyan_backend.storage.indexes import (
    lookup_by_active,
    lookup_by_archived,
    lookup_by_experiment,
    lookup_by_hparam_eq,
    lookup_by_hparam_range,
    lookup_by_tag,
    lookup_by_trace_name,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class PlanResult(NamedTuple):
    """Result of ``plan_query``.

    - ``candidates is None`` → full scan (lazy path).
    - ``candidates`` is a list and ``exact is True`` → exact result.
    - ``candidates`` is a list and ``exact is False`` → superset;
      caller must apply ``RestrictedPythonQuery.check()``.
    - ``trace_names`` is ``None`` when the query has no indexed
      metric-name predicate (stream all traces).  When set, only the
      listed metric names matched the index and should be streamed.
    """

    candidates: list[str] | None
    exact: bool
    trace_names: frozenset[str] | None = None


_NONE_RESULT = PlanResult(candidates=None, exact=True)

_AST_OP_MAP: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_literal(raw: str) -> Any:  # noqa: ANN401
    """Best-effort parse of a Python literal from a regex capture."""
    s = raw.strip()
    if s in ("True", "False"):
        return s == "True"
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _plan_hparam_val(db: Database, name: str, op: str, val: Any) -> list[str] | None:  # noqa: ANN401, PLR0911
    """Core hparam planner that takes an already-resolved Python value."""
    if op == "!=":
        return None
    if op == "==":
        return lookup_by_hparam_eq(db, name, val)
    if op == "<":
        return lookup_by_hparam_range(db, name, lo=None, hi=val)
    if op == "<=":
        results = lookup_by_hparam_range(db, name, lo=None, hi=val)
        exact = lookup_by_hparam_eq(db, name, val)
        return list(dict.fromkeys(results + exact))
    if op == ">":
        results = lookup_by_hparam_range(db, name, lo=val, hi=None)
        exact = lookup_by_hparam_eq(db, name, val)
        return [h for h in results if h not in set(exact)]
    if op == ">=":
        return lookup_by_hparam_range(db, name, lo=val, hi=None)
    return None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


class _EvalResult(NamedTuple):
    """Internal result of AST evaluation.

    ``hashes is None`` means the node could not be resolved from indexes.
    ``exact`` indicates whether the result is exact or a superset.
    ``trace_names`` propagates indexed metric names through AND/OR.
    """

    hashes: list[str] | None
    exact: bool
    trace_names: frozenset[str] | None = None


_EVAL_NONE = _EvalResult(hashes=None, exact=True)


def _exact(hashes: list[str], *, trace_names: frozenset[str] | None = None) -> _EvalResult:
    return _EvalResult(hashes=hashes, exact=True, trace_names=trace_names)


def _parse_ast(expr: str) -> ast.expr | None:
    """Parse a Python expression string and return the root AST node.

    .. deprecated:: Use ``prepare_query`` + ``plan_query`` instead.
    """
    try:
        return ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None


def _is_run_attr(node: ast.expr, attr: str) -> bool:
    """True if *node* is ``run.<attr>``."""  # noqa: D401
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "run"
        and node.attr == attr
    )


def _is_metric_attr(node: ast.expr, attr: str) -> bool:
    """True if *node* is ``metric.<attr>``."""  # noqa: D401
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "metric"
        and node.attr == attr
    )


def _literal_from_node(node: ast.expr) -> tuple[bool, Any]:
    """Extract a Python literal from an AST Constant node.

    Returns ``(ok, value)``.  If the node is not a resolvable constant,
    returns ``(False, None)``.
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    return False, None


def _hparam_name_from_dot(node: ast.expr) -> str | None:
    """Extract hparam name from ``run.hparams.<name>``."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "run"
        and node.value.attr == "hparams"
    ):
        return node.attr
    return None


def _hparam_name_from_bracket(node: ast.expr) -> str | None:
    """Extract hparam name from ``run["hparams"]["<name>"]``."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.slice, ast.Constant)
        and node.value.slice.value == "hparams"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "run"
    ):
        return node.slice.value
    return None


def _eval_compare(db: Database, node: ast.Compare) -> _EvalResult:  # noqa: C901, PLR0911, PLR0912
    """Evaluate a single Compare node against known index patterns."""
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return _EVAL_NONE

    left = node.left
    op = node.ops[0]
    right = node.comparators[0]

    # --- "tag" in run.tags ---
    if isinstance(op, ast.In) and _is_run_attr(right, "tags"):
        ok, tag_name = _literal_from_node(left)
        if ok and isinstance(tag_name, str):
            return _exact(lookup_by_tag(db, tag_name))
        return _EVAL_NONE

    # --- run.hash == "x" ---
    if _is_run_attr(left, "hash") and isinstance(op, ast.Eq):
        ok, val = _literal_from_node(right)
        if ok and isinstance(val, str):
            return _exact([val])
        return _EVAL_NONE

    # --- run.experiment == "x" ---
    if _is_run_attr(left, "experiment") and isinstance(op, ast.Eq):
        ok, val = _literal_from_node(right)
        if ok and isinstance(val, str):
            return _exact(lookup_by_experiment(db, val))
        return _EVAL_NONE

    # --- run.active == True/False ---
    if _is_run_attr(left, "active") and isinstance(op, ast.Eq):
        ok, val = _literal_from_node(right)
        if ok and isinstance(val, bool):
            return _exact(lookup_by_active(db, val))
        return _EVAL_NONE

    # --- run.is_archived == True/False  /  run.archived == True/False ---
    if _is_run_attr(left, "is_archived") or _is_run_attr(left, "archived"):
        if isinstance(op, ast.Eq):
            ok, val = _literal_from_node(right)
            if ok and isinstance(val, bool):
                return _exact(lookup_by_archived(db, val))
        return _EVAL_NONE

    # --- run.hparams.<name> <op> <val> (dot syntax) ---
    hparam_name = _hparam_name_from_dot(left)
    if hparam_name is not None:
        op_str = _AST_OP_MAP.get(type(op))
        if op_str is None:
            return _EVAL_NONE
        ok, val = _literal_from_node(right)
        if not ok:
            return _EVAL_NONE
        result = _plan_hparam_val(db, hparam_name, op_str, val)
        return _exact(result) if result is not None else _EVAL_NONE

    # --- run["hparams"]["<name>"] <op> <val> (bracket syntax) ---
    hparam_name = _hparam_name_from_bracket(left)
    if hparam_name is not None:
        op_str = _AST_OP_MAP.get(type(op))
        if op_str is None:
            return _EVAL_NONE
        ok, val = _literal_from_node(right)
        if not ok:
            return _EVAL_NONE
        result = _plan_hparam_val(db, hparam_name, op_str, val)
        return _exact(result) if result is not None else _EVAL_NONE

    # --- metric.name == "x" (Tier 3 trace-name index) ---
    if _is_metric_attr(left, "name") and isinstance(op, ast.Eq):
        ok, val = _literal_from_node(right)
        if ok and isinstance(val, str):
            return _exact(lookup_by_trace_name(db, val), trace_names=frozenset({val}))
        return _EVAL_NONE

    return _EVAL_NONE


def _merge_trace_names(
    a: frozenset[str] | None,
    b: frozenset[str] | None,
    *,
    mode: str,
) -> frozenset[str] | None:
    """Combine trace_names from two branches.

    AND: intersect (or keep whichever side is not None).
    OR:  union.
    ``None`` means "all traces" (no filter).
    """
    if a is None and b is None:
        return None
    if mode == "and":
        if a is None:
            return b
        if b is None:
            return a
        return a & b
    # mode == "or"  # noqa: ERA001
    if a is None or b is None:
        return None
    return a | b


def _eval_expr(db: Database, node: ast.expr) -> _EvalResult:  # noqa: C901, PLR0911, PLR0912
    """Recursively evaluate an AST expression into a candidate hash list."""
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            child_results = [_eval_expr(db, child) for child in node.values]
            indexed = [r for r in child_results if r.hashes is not None]

            if not indexed:
                return _EVAL_NONE

            has_unindexed = any(r.hashes is None for r in child_results)
            all_exact = all(r.exact for r in child_results) and not has_unindexed

            result_set: set[str] | None = None
            merged_traces: frozenset[str] | None = None
            for r in indexed:
                child_set = set(r.hashes)  # ty:ignore[invalid-argument-type]
                if result_set is None:
                    result_set = child_set
                else:
                    result_set &= child_set
                merged_traces = _merge_trace_names(merged_traces, r.trace_names, mode="and")

            hashes = list(result_set) if result_set is not None else []
            return _EvalResult(hashes=hashes, exact=all_exact, trace_names=merged_traces)

        if isinstance(node.op, ast.Or):
            seen: set[str] = set()
            result_list: list[str] = []
            merged_traces: frozenset[str] | None = None
            first = True
            for child in node.values:
                child_result = _eval_expr(db, child)
                if child_result.hashes is None:
                    return _EVAL_NONE
                if not child_result.exact:
                    return _EVAL_NONE
                for h in child_result.hashes:
                    if h not in seen:
                        seen.add(h)
                        result_list.append(h)
                if first:
                    merged_traces = child_result.trace_names
                    first = False
                else:
                    merged_traces = _merge_trace_names(merged_traces, child_result.trace_names, mode="or")
            return _EvalResult(hashes=result_list, exact=True, trace_names=merged_traces)

        return _EVAL_NONE

    if isinstance(node, ast.Compare):
        return _eval_compare(db, node)

    return _EVAL_NONE


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _has_unindexed_metric_node(node: ast.AST) -> bool:
    """Walk the AST and return True if any ``metric.*`` usage is not index-backed.

    Collects the set of ``metric.*`` Attribute node ids that live inside
    indexed Compare nodes (``metric.name == "literal"``), then walks the
    full tree — any ``metric.*`` Attribute whose id is not in the
    "safe" set means there is an unindexed metric predicate.
    """
    safe_ids: set[int] = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Compare):
            continue
        if len(n.ops) != 1 or len(n.comparators) != 1:
            continue
        if _is_metric_attr(n.left, "name") and isinstance(n.ops[0], ast.Eq):
            ok, val = _literal_from_node(n.comparators[0])
            if ok and isinstance(val, str):
                safe_ids.add(id(n.left))

    for n in ast.walk(node):
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "metric"
            and id(n) not in safe_ids
        ):
            return True
    return False


def query_has_sequence_level_predicate(prepared_ast: ast.Expression) -> bool:
    """Return True if the prepared AST references any ``metric.*`` field."""
    return "metric." in ast.unparse(prepared_ast)


def query_has_unindexed_sequence_predicate(prepared_ast: ast.Expression) -> bool:
    """Return True if the prepared AST has sequence-level predicates the index cannot resolve.

    Currently the only indexed metric predicate is ``metric.name == "..."``.
    Anything else (``metric.context``, ``metric.name.startswith``, etc.)
    requires the lazy path.
    """
    if "metric." not in ast.unparse(prepared_ast):
        return False
    return _has_unindexed_metric_node(prepared_ast.body)


def plan_query(db: Database, prepared_ast: ast.Expression) -> PlanResult:
    """Return candidate run hashes from index, or ``None`` for full scan.

    Accepts a prepared ``ast.Expression`` (from ``prepare_query``) and
    recursively resolves index-backed predicates.  AND nodes produce
    intersections, OR nodes produce unions.

    When an AND mixes indexed and unindexed predicates, returns the
    intersection of the indexed parts with ``exact=False`` so the caller
    knows to run ``check(run=...)`` on each candidate.
    """
    root = prepared_ast.body
    result = _eval_expr(db, root)
    return PlanResult(candidates=result.hashes, exact=result.exact, trace_names=result.trace_names)
