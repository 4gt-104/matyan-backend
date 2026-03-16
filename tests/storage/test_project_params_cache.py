"""Tests for the bounded TTL project params cache."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from matyan_backend.storage import project as project_mod
from matyan_backend.storage import runs
from matyan_backend.storage.project import (
    get_project_params_cached,
    init_params_cache,
    invalidate_project_params_cache,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


# ruff: noqa: SLF001


class TestInitParamsCache:
    def test_sets_maxsize_and_ttl(self) -> None:
        init_params_cache(maxsize=10, ttl=60)
        assert project_mod._params_cache.maxsize == 10
        assert project_mod._params_cache.ttl == 60

    def test_clamps_to_minimum_1(self) -> None:
        init_params_cache(maxsize=0, ttl=0)
        assert project_mod._params_cache.maxsize == 1
        assert project_mod._params_cache.ttl == 1


class TestInvalidateCache:
    def test_clears_entries(self, db: Database) -> None:
        init_params_cache(maxsize=32, ttl=300)
        get_project_params_cached(db)
        assert len(project_mod._params_cache) > 0
        invalidate_project_params_cache()
        assert len(project_mod._params_cache) == 0


class TestCacheHit:
    def test_returns_cached_result(self, db: Database) -> None:
        init_params_cache(maxsize=32, ttl=300)
        invalidate_project_params_cache()

        runs.create_run(db, "cache_hit_1")
        runs.set_run_attrs(db, "cache_hit_1", (), {"lr": 0.01})

        first = get_project_params_cached(db)
        assert "lr" in first["params"]

        runs.set_run_attrs(db, "cache_hit_1", (), {"lr": 0.01, "new_key": 42})
        second = get_project_params_cached(db)
        assert second is first

    def test_cache_miss_after_invalidation(self, db: Database) -> None:
        init_params_cache(maxsize=32, ttl=300)

        runs.create_run(db, "cache_inv_1")
        runs.set_run_attrs(db, "cache_inv_1", (), {"lr": 0.01})
        first = get_project_params_cached(db)

        runs.set_run_attrs(db, "cache_inv_1", (), {"lr": 0.01, "extra": 99})
        invalidate_project_params_cache()

        second = get_project_params_cached(db)
        assert second is not first
        assert "extra" in second["params"]


class TestTTLExpiry:
    def test_entry_expires_after_ttl(self, db: Database) -> None:
        init_params_cache(maxsize=32, ttl=1)
        invalidate_project_params_cache()

        runs.create_run(db, "ttl_exp_1")
        runs.set_run_attrs(db, "ttl_exp_1", (), {"lr": 0.01})
        first = get_project_params_cached(db)

        runs.set_run_attrs(db, "ttl_exp_1", (), {"lr": 0.01, "expired_key": True})
        time.sleep(1.1)

        second = get_project_params_cached(db)
        assert second is not first
        assert "expired_key" in second["params"]


class TestMaxsizeBound:
    def test_evicts_when_full(self, db: Database) -> None:
        init_params_cache(maxsize=2, ttl=300)
        invalidate_project_params_cache()

        runs.create_run(db, "max_1")
        runs.set_run_attrs(db, "max_1", (), {"a": 1})
        runs.set_context(db, "max_1", 0, {})
        runs.set_trace_info(db, "max_1", 0, "loss", dtype="float", last=0.1)

        get_project_params_cached(db, sequence_types=("metric",))
        get_project_params_cached(db, sequence_types=("images",))
        get_project_params_cached(db, sequence_types=("texts",))

        assert len(project_mod._params_cache) <= 2


class TestDifferentCacheKeys:
    def test_exclude_params_produces_separate_entry(self, db: Database) -> None:
        init_params_cache(maxsize=32, ttl=300)
        invalidate_project_params_cache()

        runs.create_run(db, "key_1")
        runs.set_run_attrs(db, "key_1", (), {"lr": 0.01})

        with_params = get_project_params_cached(db, exclude_params=False)
        without_params = get_project_params_cached(db, exclude_params=True)

        assert "lr" in with_params["params"]
        assert without_params["params"] == {}
        assert len(project_mod._params_cache) == 2
