"""Tree operations: flatten/unflatten nested Python objects into flat FDB key-value pairs.

Keys use ``fdb.tuple.pack()`` within a Subspace. Values use msgpack via ``encoding.py``.

Scalars are stored with a ``LEAF_SENTINEL`` suffix so they fall within
``subspace.range(path)`` (which only covers keys that extend *path* by at
least one element).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import encoding

if TYPE_CHECKING:
    from matyan_backend.fdb_types import DirectorySubspace, Transaction

_EMPTY_DICT_SENTINEL = "__empty_dict__"
_EMPTY_LIST_SENTINEL = "__empty_list__"
LEAF_SENTINEL = "__leaf__"


# ---------------------------------------------------------------------------
# Flatten (nested Python object -> flat (tuple-path, bytes-value) pairs)
# ---------------------------------------------------------------------------


def _flatten(path: tuple, obj: Any) -> list[tuple[tuple, Any]]:  # noqa: ANN401
    if isinstance(obj, dict):
        if not obj:
            return [((*path, _EMPTY_DICT_SENTINEL), True)]
        pairs: list[tuple[tuple, Any]] = []
        for k, v in obj.items():
            pairs.extend(_flatten((*path, k), v))
        return pairs

    if isinstance(obj, list):
        if not obj:
            return [((*path, _EMPTY_LIST_SENTINEL), True)]
        pairs = []
        for i, v in enumerate(obj):
            pairs.extend(_flatten((*path, i), v))
        return pairs

    return [((*path, LEAF_SENTINEL), obj)]


# ---------------------------------------------------------------------------
# Unflatten (sorted flat (relative-tuple, raw-value) pairs -> nested object)
# ---------------------------------------------------------------------------


def _unflatten(items: list[tuple[tuple, Any]]) -> Any:  # noqa: ANN401, C901
    """Reconstruct a nested dict/list from sorted ``(path_tuple, decoded_value)`` pairs."""
    if not items:
        return None

    # Single leaf sentinel -> scalar value
    if len(items) == 1:
        key = items[0][0]
        if key == (LEAF_SENTINEL,):
            return items[0][1]
        if key == (_EMPTY_DICT_SENTINEL,):
            return {}
        if key == (_EMPTY_LIST_SENTINEL,):
            return []

    # Group by first key element
    groups: dict[Any, list[tuple[tuple, Any]]] = {}
    order: list[Any] = []
    for path, val in items:
        head = path[0]
        tail = path[1:]
        if head not in groups:
            groups[head] = []
            order.append(head)
        groups[head].append((tail, val))

    # Determine if this level is a list (all keys are sequential ints) or dict
    is_list = all(isinstance(k, int) for k in order)

    if is_list:
        result_list: list[Any] = [None] * (max(order) + 1) if order else []
        for key in order:
            result_list[key] = _unflatten(groups[key])
        return result_list

    result_dict: dict[str, Any] = {}
    for key in order:
        result_dict[key] = _unflatten(groups[key])
    return result_dict


# ---------------------------------------------------------------------------
# Public API -- operate on an FDB transaction + subspace
# ---------------------------------------------------------------------------


def tree_set(tr: Transaction, subspace: DirectorySubspace, path: tuple, value: Any) -> None:  # noqa: ANN401
    """Write *value* (possibly nested dict/list) under *path* in *subspace*.

    Clears the existing subtree first, then writes all leaf key-value pairs.
    """
    r = subspace.range(path)
    del tr[r.start : r.stop]

    for leaf_path, leaf_val in _flatten(path, value):
        tr[subspace.pack(leaf_path)] = encoding.encode_value(leaf_val)


def tree_get(tr: Transaction, subspace: DirectorySubspace, path: tuple) -> Any:  # noqa: ANN401
    """Read and reconstruct the value stored under *path*.

    Returns ``None`` if no keys exist under *path*.
    """
    r = subspace.range(path)
    kvs = list(tr.get_range(r.start, r.stop))
    if not kvs:
        return None

    prefix_len = len(path)
    items: list[tuple[tuple, Any]] = []
    for kv in kvs:
        full_key = subspace.unpack(kv.key)
        relative = full_key[prefix_len:]
        items.append((relative, encoding.decode_value(kv.value)))

    return _unflatten(items)


def tree_delete(tr: Transaction, subspace: DirectorySubspace, path: tuple) -> None:
    """Delete the entire subtree under *path*."""
    r = subspace.range(path)
    del tr[r.start : r.stop]


def tree_keys(tr: Transaction, subspace: DirectorySubspace, path: tuple, *, level: int = 1) -> list[str | int]:
    """Return distinct key elements at *level* positions below *path*.

    With ``level=1`` (default), returns the immediate children keys.

    Uses FDB prefix-skip scanning: after finding a key with prefix P,
    jumps directly past *all* keys sharing that prefix.  This makes the
    cost O(distinct_children) instead of O(total_keys_under_path).
    """
    r = subspace.range(path)
    begin = r.start
    end = r.stop
    depth = len(path)

    result: list[str | int] = []

    while True:
        kvs = list(tr.get_range(begin, end, limit=1))
        if not kvs:
            break
        full_key = subspace.unpack(kvs[0].key)
        if len(full_key) <= depth + level - 1:
            begin = kvs[0].key + b"\x00"
            continue
        element = full_key[depth] if level == 1 else full_key[depth : depth + level]
        result.append(element)
        child_prefix = (*path, full_key[depth]) if level == 1 else (*path, *full_key[depth : depth + level])
        skip_range = subspace.range(child_prefix)
        begin = skip_range.stop

    return result
