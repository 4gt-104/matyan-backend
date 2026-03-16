"""Tests for storage/context_utils.py — deterministic context hashing."""

from matyan_api_models.context import context_to_id


class TestContextToId:
    def test_none_returns_zero(self) -> None:
        assert context_to_id(None) == 0

    def test_empty_dict_returns_zero(self) -> None:
        assert context_to_id({}) == 0

    def test_deterministic(self) -> None:
        ctx = {"subset": "train", "augmented": True}
        id1 = context_to_id(ctx)
        id2 = context_to_id(ctx)
        assert id1 == id2

    def test_key_order_irrelevant(self) -> None:
        ctx1 = {"a": 1, "b": 2}
        ctx2 = {"b": 2, "a": 1}
        assert context_to_id(ctx1) == context_to_id(ctx2)

    def test_different_dicts_differ(self) -> None:
        id1 = context_to_id({"x": 1})
        id2 = context_to_id({"x": 2})
        assert id1 != id2

    def test_returns_positive_int(self) -> None:
        result = context_to_id({"key": "value"})
        assert isinstance(result, int)
        assert result > 0
