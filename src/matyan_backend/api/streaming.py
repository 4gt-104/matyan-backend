r"""Binary streaming codec compatible with the Aim UI.

The UI decodes responses using ``streamEncoding.ts`` which expects:

- **Path encoding** - keys joined by ``\\xfe`` sentinel.
  String keys → UTF-8 bytes.  Integer keys → ``\\xfe`` + 8-byte BE int64.
  Each key (string or integer block) is terminated by ``\\xfe``.
- **Value encoding** - 1-byte type-id prefix + payload.
  0=None  1=Bool  2=Int(LE i64)  3=Float(LE f64)
  4=String(UTF-8)  5=Bytes  6=ArrayFlag  7=ObjectFlag
- **Stream chunks** - ``[4B LE key_len][key][4B LE val_len][val]…``

Pure-Python port of ``aim/storage/treeutils.pyx`` and
``aim/storage/encoding/encoding.pyx`` (no Cython).
"""

from __future__ import annotations

import struct as _struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Sentinel / flags
# ---------------------------------------------------------------------------

_PATH_SENTINEL = b"\xfe"
_PATH_SENTINEL_CODE = 0xFE

_NONE = 0
_BOOL = 1
_INT = 2
_FLOAT = 3
_STRING = 4
_BYTES = 5
_ARRAY = 6
_OBJECT = 7

PROGRESS_REPORT_INTERVAL = 0.5

_PACK_LE_I = _struct.Struct("<I")
_PACK_LE_Q = _struct.Struct("<q")
_PACK_LE_D = _struct.Struct("<d")
_PACK_BE_Q = _struct.Struct(">q")


class _ArrayFlag:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<ArrayFlag>"


class _ObjectFlag:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<ObjectFlag>"


ArrayFlag = _ArrayFlag()
ObjectFlag = _ObjectFlag()


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------


def encode_value(value: Any) -> bytes:  # noqa: ANN401, PLR0911
    """Encode a primitive Python value into the Aim binary format."""
    if value is None:
        return bytes([_NONE])
    if isinstance(value, bool):
        return bytes([_BOOL, int(value)])
    if isinstance(value, int):
        return bytes([_INT]) + _PACK_LE_Q.pack(value)
    if isinstance(value, float):
        return bytes([_FLOAT]) + _PACK_LE_D.pack(value)
    if isinstance(value, str):
        return bytes([_STRING]) + value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes([_BYTES]) + bytes(value)
    if isinstance(value, _ArrayFlag):
        return bytes([_ARRAY])
    if isinstance(value, _ObjectFlag):
        return bytes([_OBJECT])
    msg = f"Unsupported value type: {type(value)}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------


def _encode_key(key: str | int) -> bytes:
    """Encode a single path component."""
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, int):
        return _PATH_SENTINEL + _PACK_BE_Q.pack(key)
    msg = f"Unsupported key type: {type(key)}"
    raise TypeError(msg)


def encode_path(path: tuple) -> bytes:
    r"""Encode a tuple path into ``\\xfe``-delimited bytes."""
    return b"".join(_encode_key(k) + _PATH_SENTINEL for k in path)


# ---------------------------------------------------------------------------
# Tree unfolding
# ---------------------------------------------------------------------------


_NON_ITERABLE = (str, bytes, bytearray, memoryview)


def _is_iterable_sequence(obj: object) -> bool:
    """Check whether *obj* is an iterable sequence (list, tuple, generator, etc.)."""
    return hasattr(obj, "__iter__") and not isinstance(obj, (*_NON_ITERABLE, dict))


def unfold_tree(
    obj: Any,  # noqa: ANN401
    *,
    path: tuple = (),
    unfold_array: bool = True,
    depth: int | None = None,
) -> Iterator[tuple[tuple, Any]]:
    """Flatten a nested Python object into ``(path, value)`` pairs.

    Sequences (lists, tuples, generators, and any other non-dict iterable)
    are lazily consumed via ``enumerate()``, so a generator of trace dicts
    will be unfolded one element at a time without materializing the full list.
    """
    if depth == 0:
        yield path, obj
        return
    next_depth = None if depth is None else depth - 1

    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        yield path, obj
    elif isinstance(obj, dict):
        if not obj:
            yield path, ObjectFlag
        for key, val in obj.items():
            yield from unfold_tree(val, path=(*path, key), unfold_array=unfold_array, depth=next_depth)
    elif _is_iterable_sequence(obj):
        if not unfold_array:
            yield path, obj
        else:
            yield path, ArrayFlag
            for idx, val in enumerate(obj):
                yield from unfold_tree(val, path=(*path, idx), unfold_array=unfold_array, depth=next_depth)
    else:
        yield path, repr(obj)


# ---------------------------------------------------------------------------
# Encode tree  →  binary stream data
# ---------------------------------------------------------------------------


def encode_paths_vals(
    paths_vals: Iterator[tuple[tuple, Any]],
) -> Iterator[tuple[bytes, bytes]]:
    """Encode ``(path, value)`` pairs into ``(encoded_path, encoded_val)``."""
    for path, val in paths_vals:
        yield encode_path(path), encode_value(val)


def encode_tree(obj: Any) -> Iterator[tuple[bytes, bytes]]:  # noqa: ANN401
    """Convenience: unfold + encode in one shot."""  # noqa: D401
    return encode_paths_vals(unfold_tree(obj))


def collect_streamable_data(encoded_tree: Iterator[tuple[bytes, bytes]]) -> bytes:
    """Pack encoded key-value pairs into a single binary chunk.

    Format per pair: ``[4-byte LE key_len][key][4-byte LE val_len][val]``
    """
    parts: list[bytes] = []
    pack = _PACK_LE_I.pack
    for key, val in encoded_tree:
        parts.append(pack(len(key)))
        parts.append(key)
        parts.append(pack(len(val)))
        parts.append(val)
    return b"".join(parts)


def stream_tree_data(
    encoded_tree: Iterator[tuple[bytes, bytes]],
    flush_every: int = 50,
) -> Iterator[bytes]:
    """Yield binary chunks incrementally instead of materializing everything.

    Semantically equivalent to ``collect_streamable_data`` but yields a
    ``bytes`` chunk every *flush_every* encoded pairs, keeping memory bounded
    when the source ``encoded_tree`` is itself a lazy generator (e.g. backed
    by chunked FDB reads).
    """
    parts: list[bytes] = []
    pack = _PACK_LE_I.pack
    count = 0
    for key, val in encoded_tree:
        parts.append(pack(len(key)))
        parts.append(key)
        parts.append(pack(len(val)))
        parts.append(val)
        count += 1
        if count >= flush_every:
            yield b"".join(parts)
            parts = []
            count = 0
    if parts:
        yield b"".join(parts)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def make_progress_key(counter: int) -> str:
    return f"progress_{counter}"
