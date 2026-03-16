"""Tests for api/streaming.py — binary streaming codec."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from matyan_backend.api.streaming import (
    ArrayFlag,
    ObjectFlag,
    _ArrayFlag,
    _encode_key,
    _ObjectFlag,
    collect_streamable_data,
    encode_path,
    encode_tree,
    encode_value,
    make_progress_key,
    stream_tree_data,
    unfold_tree,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class TestEncodeValue:
    def test_none(self) -> None:
        assert encode_value(None) == bytes([0])

    def test_bool_true(self) -> None:
        assert encode_value(True) == bytes([1, 1])

    def test_bool_false(self) -> None:
        assert encode_value(False) == bytes([1, 0])

    def test_int(self) -> None:
        result = encode_value(42)
        assert result[0] == 2
        assert struct.unpack("<q", result[1:])[0] == 42

    def test_float(self) -> None:
        result = encode_value(3.14)
        assert result[0] == 3
        assert abs(struct.unpack("<d", result[1:])[0] - 3.14) < 1e-10

    def test_string(self) -> None:
        result = encode_value("hello")
        assert result[0] == 4
        assert result[1:] == b"hello"

    def test_bytes(self) -> None:
        result = encode_value(b"\x01\x02")
        assert result[0] == 5
        assert result[1:] == b"\x01\x02"

    def test_bytearray(self) -> None:
        result = encode_value(bytearray(b"\x03"))
        assert result[0] == 5

    def test_array_flag(self) -> None:
        result = encode_value(ArrayFlag)
        assert result == bytes([6])

    def test_object_flag(self) -> None:
        result = encode_value(ObjectFlag)
        assert result == bytes([7])

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported value type"):
            encode_value(object())


class TestEncodePath:
    def test_string_keys(self) -> None:
        result = encode_path(("a", "b"))
        assert b"\xfe" in result

    def test_int_key(self) -> None:
        result = encode_path((0,))
        assert b"\xfe" in result
        assert len(result) > 1

    def test_mixed_keys(self) -> None:
        result = encode_path(("key", 42))
        assert isinstance(result, bytes)

    def test_empty_path(self) -> None:
        assert encode_path(()) == b""


class TestUnfoldTree:
    def test_primitive(self) -> None:
        result = list(unfold_tree(42))
        assert result == [((), 42)]

    def test_none(self) -> None:
        result = list(unfold_tree(None))
        assert result == [((), None)]

    def test_string(self) -> None:
        result = list(unfold_tree("hello"))
        assert result == [((), "hello")]

    def test_dict(self) -> None:
        result = list(unfold_tree({"a": 1, "b": 2}))
        assert (("a",), 1) in result
        assert (("b",), 2) in result

    def test_empty_dict(self) -> None:
        result = list(unfold_tree({}))
        assert result == [((), ObjectFlag)]

    def test_list(self) -> None:
        result = list(unfold_tree([10, 20]))
        assert ((), ArrayFlag) in result
        assert ((0,), 10) in result
        assert ((1,), 20) in result

    def test_list_unfold_false(self) -> None:
        result = list(unfold_tree([1, 2], unfold_array=False))
        assert result == [((), [1, 2])]

    def test_nested_dict(self) -> None:
        result = list(unfold_tree({"a": {"b": 1}}))
        assert (("a", "b"), 1) in result

    def test_depth_limit(self) -> None:
        result = list(unfold_tree({"a": {"b": 1}}, depth=1))
        paths = [p for p, _ in result]
        assert ("a",) in paths
        assert ("a", "b") not in paths

    def test_bytes_value(self) -> None:
        result = list(unfold_tree(b"\x01"))
        assert result == [((), b"\x01")]

    def test_unsupported_type_uses_repr(self) -> None:
        class Custom:
            pass

        result = list(unfold_tree(Custom()))
        assert len(result) == 1
        assert "Custom" in result[0][1]


class TestCollectStreamableData:
    def test_basic(self) -> None:
        data = collect_streamable_data(encode_tree({"key": "val"}))
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_empty(self) -> None:
        data = collect_streamable_data(iter([]))
        assert data == b""

    def test_structure(self) -> None:
        encoded = list(encode_tree({"x": 1}))
        data = collect_streamable_data(iter(encoded))
        offset = 0
        for key_bytes, val_bytes in encoded:
            key_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            assert data[offset : offset + key_len] == key_bytes
            offset += key_len
            val_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            assert data[offset : offset + val_len] == val_bytes
            offset += val_len


class TestMakeProgressKey:
    def test_format(self) -> None:
        assert make_progress_key(0) == "progress_0"
        assert make_progress_key(42) == "progress_42"


class TestFlagReprs:
    def test_array_flag_repr(self) -> None:
        assert repr(_ArrayFlag()) == "<ArrayFlag>"

    def test_object_flag_repr(self) -> None:
        assert repr(_ObjectFlag()) == "<ObjectFlag>"


class TestEncodePathEdge:
    def test_unsupported_key_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported key type"):
            _encode_key(3.14)  # ty:ignore[invalid-argument-type]


class TestUnfoldTreeTuple:
    def test_tuple(self) -> None:
        result = list(unfold_tree((1, 2)))
        assert ((), ArrayFlag) in result
        assert ((0,), 1) in result
        assert ((1,), 2) in result

    def test_memoryview(self) -> None:
        result = list(unfold_tree(memoryview(b"\x01")))
        assert len(result) == 1


class TestUnfoldTreeGenerator:
    """unfold_tree should lazily consume generators and other iterables."""

    def test_generator(self) -> None:
        def gen() -> Generator[int]:
            yield 10
            yield 20

        result = list(unfold_tree(gen()))
        assert ((), ArrayFlag) in result
        assert ((0,), 10) in result
        assert ((1,), 20) in result

    def test_generator_nested_in_dict(self) -> None:
        def gen() -> Generator[dict]:
            yield {"a": 1}
            yield {"a": 2}

        result = list(unfold_tree({"items": gen()}))
        paths = [p for p, _ in result]
        assert ("items", 0, "a") in paths
        assert ("items", 1, "a") in paths

    def test_range_iterable(self) -> None:
        result = list(unfold_tree(range(3)))
        assert ((), ArrayFlag) in result
        assert ((0,), 0) in result
        assert ((2,), 2) in result

    def test_set_iterable(self) -> None:
        result = list(unfold_tree({42}))
        assert ((), ArrayFlag) in result
        assert len(result) == 2

    def test_generator_unfold_false(self) -> None:
        def gen() -> Generator[int]:
            yield 1

        result = list(unfold_tree(gen(), unfold_array=False))
        assert len(result) == 1
        assert result[0][0] == ()

    def test_encode_tree_with_generator(self) -> None:
        """encode_tree should produce identical output for list vs generator."""
        items = [{"x": 1}, {"x": 2}]
        from_list = collect_streamable_data(encode_tree({"data": items}))
        from_gen = collect_streamable_data(encode_tree({"data": iter(items)}))
        assert from_list == from_gen


class TestStreamTreeData:
    """stream_tree_data yields incremental byte chunks."""

    def test_concatenation_matches_collect(self) -> None:
        obj = {"run1": {"params": {"lr": 0.01}, "traces": [{"name": f"m{i}"} for i in range(20)]}}
        expected = collect_streamable_data(encode_tree(obj))
        chunks = list(stream_tree_data(encode_tree(obj), flush_every=5))
        assert b"".join(chunks) == expected

    def test_yields_multiple_chunks(self) -> None:
        obj = {"data": list(range(100))}
        chunks = list(stream_tree_data(encode_tree(obj), flush_every=10))
        assert len(chunks) > 1

    def test_single_pair_yields_one_chunk(self) -> None:
        obj = {"key": "val"}
        chunks = list(stream_tree_data(encode_tree(obj), flush_every=100))
        assert len(chunks) == 1

    def test_empty_tree(self) -> None:
        chunks = list(stream_tree_data(iter([]), flush_every=10))
        assert chunks == []

    def test_with_lazy_generator(self) -> None:
        """Lazy generator + stream_tree_data should produce valid output."""

        def gen() -> Generator[dict]:
            for i in range(10):
                yield {"name": f"metric_{i}", "value": float(i)}

        obj = {"run": {"traces": gen()}}
        chunks = list(stream_tree_data(encode_tree(obj), flush_every=5))
        combined = b"".join(chunks)
        assert len(combined) > 0

        obj_list = {"run": {"traces": [{"name": f"metric_{i}", "value": float(i)} for i in range(10)]}}
        expected = collect_streamable_data(encode_tree(obj_list))
        assert combined == expected
