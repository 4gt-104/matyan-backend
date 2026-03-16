from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import msgpack

_EXT_TYPE_DATETIME = 1


def _encode_ext(obj: object) -> bytes:
    if isinstance(obj, datetime):
        ts = obj.timestamp()
        return msgpack.ExtType(_EXT_TYPE_DATETIME, msgpack.packb(ts))  # type: ignore[return-value]
    msg = f"Cannot msgpack-encode object of type {type(obj)}"
    raise TypeError(msg)


def _decode_ext(code: int, data: bytes) -> object:
    if code == _EXT_TYPE_DATETIME:
        ts: float = msgpack.unpackb(data)
        return datetime.fromtimestamp(ts, tz=UTC)
    return msgpack.ExtType(code, data)


def encode_value(obj: Any) -> bytes:  # noqa: ANN401
    return msgpack.packb(obj, default=_encode_ext, use_bin_type=True)


def decode_value(data: bytes | bytearray) -> Any:  # noqa: ANN401
    if type(data) not in (bytes, bytearray):
        data = bytes(data)
    return msgpack.unpackb(data, ext_hook=_decode_ext, raw=False)
