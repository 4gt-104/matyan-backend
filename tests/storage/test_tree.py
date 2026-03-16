"""Tests for tree.py — flatten/unflatten and tree_set/tree_get round-trips."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fdb

from matyan_backend.storage.fdb_client import get_directories
from matyan_backend.storage.tree import (
    LEAF_SENTINEL,
    _flatten,
    _unflatten,
    tree_delete,
    tree_get,
    tree_keys,
    tree_set,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database, Transaction


class TestFlattenUnflatten:
    def test_scalar(self) -> None:
        pairs = _flatten(("a",), 42)
        assert pairs == [(("a", LEAF_SENTINEL), 42)]
        assert _unflatten([((LEAF_SENTINEL,), 42)]) == 42

    def test_dict(self) -> None:
        obj = {"x": 1, "y": 2}
        pairs = _flatten((), obj)
        expected = {(("x", LEAF_SENTINEL), 1), (("y", LEAF_SENTINEL), 2)}
        assert set(pairs) == expected
        assert _unflatten(pairs) == obj

    def test_nested_dict(self) -> None:
        obj = {"a": {"b": 10, "c": 20}, "d": 30}
        pairs = _flatten((), obj)
        reconstructed = _unflatten(pairs)
        assert reconstructed == obj

    def test_list(self) -> None:
        obj = [10, 20, 30]
        pairs = _flatten((), obj)
        assert _unflatten(pairs) == obj

    def test_empty_dict(self) -> None:
        pairs = _flatten((), {})
        assert len(pairs) == 1
        assert _unflatten(pairs) == {}

    def test_empty_list(self) -> None:
        pairs = _flatten((), [])
        assert len(pairs) == 1
        assert _unflatten(pairs) == []

    def test_none_value(self) -> None:
        pairs = _flatten(("k",), None)
        assert pairs == [(("k", LEAF_SENTINEL), None)]

    def test_mixed_nested(self) -> None:
        obj = {"layers": [{"units": 64}, {"units": 128}], "lr": 0.01}
        pairs = _flatten((), obj)
        assert _unflatten(pairs) == obj

    def test_unflatten_empty(self) -> None:
        assert _unflatten([]) is None


class TestTreeSetGet:
    def test_scalar_round_trip(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            tree_set(tr, rd, ("test", "val"), 42)
            result = tree_get(tr, rd, ("test", "val"))
            assert result == 42

        go(db)

    def test_dict_round_trip(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            data = {"lr": 0.01, "batch_size": 32, "optimizer": "adam"}
            tree_set(tr, rd, ("run1", "hparams"), data)
            result = tree_get(tr, rd, ("run1", "hparams"))
            assert result == data

        go(db)

    def test_nested_dict_round_trip(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            data = {"model": {"layers": [64, 128, 256], "dropout": 0.5}, "seed": 42}
            tree_set(tr, rd, ("run1", "attrs"), data)
            result = tree_get(tr, rd, ("run1", "attrs"))
            assert result == data

        go(db)

    def test_overwrite(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            tree_set(tr, rd, ("k",), {"a": 1, "b": 2})
            tree_set(tr, rd, ("k",), {"c": 3})
            result = tree_get(tr, rd, ("k",))
            assert result == {"c": 3}

        go(db)

    def test_get_nonexistent(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            assert tree_get(tr, rd, ("does", "not", "exist")) is None

        go(db)

    def test_delete(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            tree_set(tr, rd, ("d",), {"x": 1})
            tree_delete(tr, rd, ("d",))
            assert tree_get(tr, rd, ("d",)) is None

        go(db)

    def test_keys(self, db: Database) -> None:
        rd = get_directories().runs

        @fdb.transactional
        def go(tr: Transaction) -> None:
            tree_set(tr, rd, ("r1", "meta"), {"name": "run1"})
            tree_set(tr, rd, ("r2", "meta"), {"name": "run2"})
            keys = tree_keys(tr, rd, ())
            assert "r1" in keys
            assert "r2" in keys

        go(db)
