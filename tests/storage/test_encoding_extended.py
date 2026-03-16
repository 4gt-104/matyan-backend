"""Tests for storage/encoding.py — msgpack serialization with datetime extension."""

from __future__ import annotations

from datetime import UTC, datetime

import msgpack
import pytest

from matyan_backend.storage.encoding import _decode_ext, _encode_ext, decode_value, encode_value


class TestDatetimeRoundTrip:
    def test_datetime_encode_decode(self) -> None:
        dt = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)
        encoded = encode_value(dt)
        decoded = decode_value(encoded)
        assert isinstance(decoded, datetime)
        assert abs(decoded.timestamp() - dt.timestamp()) < 0.001

    def test_nested_datetime(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        data = {"ts": dt, "name": "test"}
        encoded = encode_value(data)
        decoded = decode_value(encoded)
        assert decoded["name"] == "test"
        assert isinstance(decoded["ts"], datetime)


class TestEncodeExt:
    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot msgpack-encode"):
            _encode_ext(object())


class TestDecodeExt:
    def test_unknown_code_returns_ext_type(self) -> None:

        result = _decode_ext(99, b"\x00")
        assert isinstance(result, msgpack.ExtType)
        assert result.code == 99


class TestPrimitiveRoundTrips:
    def test_int(self) -> None:
        assert decode_value(encode_value(42)) == 42

    def test_float(self) -> None:
        assert decode_value(encode_value(3.14)) == pytest.approx(3.14)

    def test_string(self) -> None:
        assert decode_value(encode_value("hello")) == "hello"

    def test_none(self) -> None:
        assert decode_value(encode_value(None)) is None

    def test_bool(self) -> None:
        assert decode_value(encode_value(True)) is True
        assert decode_value(encode_value(False)) is False

    def test_list(self) -> None:
        assert decode_value(encode_value([1, 2, 3])) == [1, 2, 3]

    def test_dict(self) -> None:
        assert decode_value(encode_value({"a": 1})) == {"a": 1}

    def test_bytes(self) -> None:
        assert decode_value(encode_value(b"\x01\x02")) == b"\x01\x02"
