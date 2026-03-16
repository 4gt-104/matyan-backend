"""Tests for storage/indexes.py — secondary index CRUD and lookup."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import fdb

from matyan_backend.storage import entities, indexes, runs
from matyan_backend.storage.fdb_client import ensure_directories, get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

# ruff: noqa: SLF001


def _has_rev_entries(db: Database, run_hash: str) -> bool:
    """Return True if at least one reverse index entry exists for *run_hash*."""
    idx = get_directories().indexes
    r = idx.range(("_rev", run_hash))
    return any(True for _ in db.create_transaction().get_range(r.start, r.stop))


def _count_rev_entries(db: Database, run_hash: str) -> int:
    idx = get_directories().indexes
    r = idx.range(("_rev", run_hash))
    return sum(1 for _ in db.create_transaction().get_range(r.start, r.stop))


class TestIndexRun:
    def test_index_and_lookup_archived(self, db: Database) -> None:
        indexes.index_run(db, "r1", is_archived=False, active=True, created_at=100.0)
        indexes.index_run(db, "r2", is_archived=True, active=False, created_at=200.0)

        assert indexes.lookup_by_archived(db, False) == ["r1"]
        assert indexes.lookup_by_archived(db, True) == ["r2"]

    def test_index_and_lookup_active(self, db: Database) -> None:
        indexes.index_run(db, "r1", is_archived=False, active=True, created_at=100.0)
        indexes.index_run(db, "r2", is_archived=False, active=False, created_at=200.0)

        assert indexes.lookup_by_active(db, True) == ["r1"]
        assert indexes.lookup_by_active(db, False) == ["r2"]

    def test_index_and_lookup_experiment(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0, experiment_name="baseline")
        indexes.index_run(db, "r2", created_at=200.0, experiment_name="v2")
        indexes.index_run(db, "r3", created_at=300.0, experiment_name="baseline")

        assert sorted(indexes.lookup_by_experiment(db, "baseline")) == ["r1", "r3"]
        assert indexes.lookup_by_experiment(db, "v2") == ["r2"]
        assert indexes.lookup_by_experiment(db, "nonexistent") == []

    def test_index_and_lookup_tag(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0, tag_names=["best", "prod"])
        indexes.index_run(db, "r2", created_at=200.0, tag_names=["best"])

        assert sorted(indexes.lookup_by_tag(db, "best")) == ["r1", "r2"]
        assert indexes.lookup_by_tag(db, "prod") == ["r1"]
        assert indexes.lookup_by_tag(db, "nope") == []

    def test_index_and_lookup_created_at_range(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0)
        indexes.index_run(db, "r2", created_at=200.0)
        indexes.index_run(db, "r3", created_at=300.0)

        result = indexes.lookup_by_created_at(db, start=150.0, end=250.0)
        assert result == ["r2"]

        result = indexes.lookup_by_created_at(db, start=100.0)
        assert sorted(result) == ["r1", "r2", "r3"]

        result = indexes.lookup_by_created_at(db, end=200.0)
        assert result == ["r1"]

    def test_lookup_all(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0)
        indexes.index_run(db, "r2", created_at=200.0)
        assert indexes.lookup_all_run_hashes(db) == ["r1", "r2"]


class TestDeindexRun:
    def test_deindex_removes_all_entries(self, db: Database) -> None:
        indexes.index_run(
            db,
            "r1",
            is_archived=False,
            active=True,
            created_at=100.0,
            experiment_name="exp",
            tag_names=["t1"],
        )
        assert indexes.lookup_by_archived(db, False) == ["r1"]
        assert indexes.lookup_by_experiment(db, "exp") == ["r1"]
        assert indexes.lookup_by_tag(db, "t1") == ["r1"]

        indexes.deindex_run(db, "r1")

        assert indexes.lookup_by_archived(db, False) == []
        assert indexes.lookup_by_active(db, True) == []
        assert indexes.lookup_by_experiment(db, "exp") == []
        assert indexes.lookup_by_tag(db, "t1") == []
        assert indexes.lookup_all_run_hashes(db) == []


class TestUpdateIndexField:
    def test_swap_archived(self, db: Database) -> None:
        indexes.index_run(db, "r1", is_archived=False, created_at=100.0)
        assert indexes.lookup_by_archived(db, False) == ["r1"]
        assert indexes.lookup_by_archived(db, True) == []

        indexes.update_index_field(db, "r1", "archived", False, True)

        assert indexes.lookup_by_archived(db, False) == []
        assert indexes.lookup_by_archived(db, True) == ["r1"]

    def test_swap_to_none_removes_only(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0, experiment_name="exp1")
        assert indexes.lookup_by_experiment(db, "exp1") == ["r1"]

        indexes.update_index_field(db, "r1", "experiment", "exp1", None)

        assert indexes.lookup_by_experiment(db, "exp1") == []


class TestTagIndexHelpers:
    def test_add_and_remove_tag_index(self, db: Database) -> None:
        indexes.add_tag_index(db, "r1", "my-tag")
        assert indexes.lookup_by_tag(db, "my-tag") == ["r1"]

        indexes.remove_tag_index(db, "r1", "my-tag")
        assert indexes.lookup_by_tag(db, "my-tag") == []

    def test_remove_all_tag_indexes_for_tag(self, db: Database) -> None:
        indexes.add_tag_index(db, "r1", "shared")
        indexes.add_tag_index(db, "r2", "shared")
        indexes.add_tag_index(db, "r1", "other")
        assert sorted(indexes.lookup_by_tag(db, "shared")) == ["r1", "r2"]

        indexes.remove_all_tag_indexes_for_tag(db, "shared")

        assert indexes.lookup_by_tag(db, "shared") == []
        assert indexes.lookup_by_tag(db, "other") == ["r1"]


class TestRenameExperimentIndex:
    def test_rename(self, db: Database) -> None:
        indexes.index_run(db, "r1", created_at=100.0, experiment_name="old-name")
        indexes.index_run(db, "r2", created_at=200.0, experiment_name="old-name")
        assert sorted(indexes.lookup_by_experiment(db, "old-name")) == ["r1", "r2"]

        indexes.rename_experiment_index(db, "old-name", "new-name")

        assert indexes.lookup_by_experiment(db, "old-name") == []
        assert sorted(indexes.lookup_by_experiment(db, "new-name")) == ["r1", "r2"]


class TestRebuildIndexes:
    def test_rebuild_from_scratch(self, db: Database) -> None:
        runs.create_run(db, "rb1", name="Run 1")
        runs.create_run(db, "rb2", name="Run 2")
        runs.update_run_meta(db, "rb2", is_archived=True)

        indexes._clear_all_indexes(db)
        assert indexes.lookup_all_run_hashes(db) == []

        count, ghost_count = indexes.rebuild_indexes(db)
        assert count == 2
        assert ghost_count == 0

        assert "rb1" in indexes.lookup_by_archived(db, False)
        assert "rb2" in indexes.lookup_by_archived(db, True)

    def test_rebuild_includes_experiment_and_tags(self, db: Database) -> None:
        exp = entities.create_experiment(db, "test-exp")
        tag = entities.create_tag(db, "test-tag")
        runs.create_run(db, "rb3", experiment_id=exp["id"])
        entities.set_run_experiment(db, "rb3", exp["id"])
        entities.add_tag_to_run(db, "rb3", tag["id"])

        indexes._clear_all_indexes(db)
        indexes.rebuild_indexes(db)

        assert "rb3" in indexes.lookup_by_experiment(db, "test-exp")
        assert "rb3" in indexes.lookup_by_tag(db, "test-tag")


class TestIndexIntegrationWithRuns:
    """Verify that runs.create_run / update_run_meta / delete_run maintain indexes."""

    def test_create_run_indexes(self, db: Database) -> None:
        runs.create_run(db, "int1", name="Integrated")
        assert "int1" in indexes.lookup_by_archived(db, False)
        assert "int1" in indexes.lookup_by_active(db, True)
        assert "int1" in indexes.lookup_all_run_hashes(db)

    def test_update_archived_updates_index(self, db: Database) -> None:
        runs.create_run(db, "int2")
        assert "int2" in indexes.lookup_by_archived(db, False)

        runs.update_run_meta(db, "int2", is_archived=True)
        assert "int2" not in indexes.lookup_by_archived(db, False)
        assert "int2" in indexes.lookup_by_archived(db, True)

    def test_update_active_updates_index(self, db: Database) -> None:
        runs.create_run(db, "int3")
        assert "int3" in indexes.lookup_by_active(db, True)

        runs.update_run_meta(db, "int3", active=False)
        assert "int3" not in indexes.lookup_by_active(db, True)
        assert "int3" in indexes.lookup_by_active(db, False)

    def test_delete_run_deindexes(self, db: Database) -> None:
        runs.create_run(db, "int4")
        assert "int4" in indexes.lookup_all_run_hashes(db)

        runs.delete_run(db, "int4")
        assert "int4" not in indexes.lookup_all_run_hashes(db)
        assert "int4" not in indexes.lookup_by_archived(db, False)


class TestIndexIntegrationWithEntities:
    """Verify that entity functions maintain indexes."""

    def test_set_run_experiment_indexes(self, db: Database) -> None:
        runs.create_run(db, "ent1")
        exp = entities.create_experiment(db, "my-exp")
        entities.set_run_experiment(db, "ent1", exp["id"])

        assert "ent1" in indexes.lookup_by_experiment(db, "my-exp")

    def test_change_experiment_updates_index(self, db: Database) -> None:
        runs.create_run(db, "ent2")
        exp1 = entities.create_experiment(db, "exp-a")
        exp2 = entities.create_experiment(db, "exp-b")

        entities.set_run_experiment(db, "ent2", exp1["id"])
        assert "ent2" in indexes.lookup_by_experiment(db, "exp-a")

        entities.set_run_experiment(db, "ent2", exp2["id"])
        assert "ent2" not in indexes.lookup_by_experiment(db, "exp-a")
        assert "ent2" in indexes.lookup_by_experiment(db, "exp-b")

    def test_add_remove_tag_indexes(self, db: Database) -> None:
        runs.create_run(db, "ent3")
        tag = entities.create_tag(db, "idx-tag")

        entities.add_tag_to_run(db, "ent3", tag["id"])
        assert "ent3" in indexes.lookup_by_tag(db, "idx-tag")

        entities.remove_tag_from_run(db, "ent3", tag["id"])
        assert "ent3" not in indexes.lookup_by_tag(db, "idx-tag")

    def test_delete_tag_cleans_index(self, db: Database) -> None:
        runs.create_run(db, "ent4")
        tag = entities.create_tag(db, "doomed-tag")
        entities.add_tag_to_run(db, "ent4", tag["id"])
        assert "ent4" in indexes.lookup_by_tag(db, "doomed-tag")

        entities.delete_tag(db, tag["id"])
        assert indexes.lookup_by_tag(db, "doomed-tag") == []

    def test_rename_experiment_updates_index(self, db: Database) -> None:
        runs.create_run(db, "ent5")
        exp = entities.create_experiment(db, "old-exp")
        entities.set_run_experiment(db, "ent5", exp["id"])
        assert "ent5" in indexes.lookup_by_experiment(db, "old-exp")

        entities.update_experiment(db, exp["id"], name="new-exp")
        assert indexes.lookup_by_experiment(db, "old-exp") == []
        assert "ent5" in indexes.lookup_by_experiment(db, "new-exp")


# ---------------------------------------------------------------------------
# Tier 2 — hparam indexes
# ---------------------------------------------------------------------------


class TestIndexHparams:
    def test_index_hparams_scalars(self, db: Database) -> None:
        hparams = {
            "lr": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "use_bn": True,
            "nested": {"a": 1},
            "tags_list": [1, 2, 3],
            "empty": None,
        }
        indexes.index_hparams(db, "hp1", hparams)

        assert indexes.lookup_by_hparam_eq(db, "lr", 0.001) == ["hp1"]
        assert indexes.lookup_by_hparam_eq(db, "batch_size", 32) == ["hp1"]
        assert indexes.lookup_by_hparam_eq(db, "optimizer", "adam") == ["hp1"]
        assert indexes.lookup_by_hparam_eq(db, "use_bn", True) == ["hp1"]

    def test_index_hparams_skips_non_scalars(self, db: Database) -> None:
        """Non-scalar values (dicts, lists, None) should not produce index entries."""
        indexes.index_hparams(db, "hp1b", {"nested": {"a": 1}, "tags": [1, 2], "empty": None})
        all_hp = indexes.lookup_by_hparam_range(db, "nested")
        assert "hp1b" not in all_hp
        all_hp = indexes.lookup_by_hparam_range(db, "tags")
        assert "hp1b" not in all_hp
        all_hp = indexes.lookup_by_hparam_range(db, "empty")
        assert "hp1b" not in all_hp

    def test_deindex_hparams(self, db: Database) -> None:
        indexes.index_hparams(db, "hp2", {"lr": 0.01, "epochs": 100})
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.01) == ["hp2"]

        indexes.deindex_hparams(db, "hp2")
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.01) == []
        assert indexes.lookup_by_hparam_eq(db, "epochs", 100) == []

    def test_lookup_by_hparam_eq(self, db: Database) -> None:
        indexes.index_hparams(db, "eq1", {"lr": 0.001})
        indexes.index_hparams(db, "eq2", {"lr": 0.01})
        indexes.index_hparams(db, "eq3", {"lr": 0.001})

        assert sorted(indexes.lookup_by_hparam_eq(db, "lr", 0.001)) == ["eq1", "eq3"]
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.01) == ["eq2"]
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.1) == []

    def test_lookup_by_hparam_range(self, db: Database) -> None:
        indexes.index_hparams(db, "rng1", {"batch_size": 16})
        indexes.index_hparams(db, "rng2", {"batch_size": 32})
        indexes.index_hparams(db, "rng3", {"batch_size": 64})
        indexes.index_hparams(db, "rng4", {"batch_size": 128})

        result = indexes.lookup_by_hparam_range(db, "batch_size", lo=32, hi=128)
        assert sorted(result) == ["rng2", "rng3"]

        result = indexes.lookup_by_hparam_range(db, "batch_size", lo=None, hi=64)
        assert sorted(result) == ["rng1", "rng2"]

        result = indexes.lookup_by_hparam_range(db, "batch_size", lo=64, hi=None)
        assert sorted(result) == ["rng3", "rng4"]

    def test_deindex_run_also_removes_hparams(self, db: Database) -> None:
        indexes.index_run(db, "hpdr1", created_at=1.0)
        indexes.index_hparams(db, "hpdr1", {"lr": 0.01})
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.01) == ["hpdr1"]

        indexes.deindex_run(db, "hpdr1")
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.01) == []


class TestRebuildWithPendingDeletion:
    def test_rebuild_skips_pending_deletion_runs(self, db: Database) -> None:
        runs.create_run(db, "rbpd1")
        runs.create_run(db, "rbpd2")
        runs.mark_pending_deletion(db, "rbpd2")

        indexes._clear_all_indexes(db)
        count, ghost_count = indexes.rebuild_indexes(db)

        assert "rbpd1" in indexes.lookup_all_run_hashes(db)
        assert "rbpd2" not in indexes.lookup_all_run_hashes(db)
        assert count == 1
        assert ghost_count == 1


class TestTombstones:
    def test_mark_and_check_tombstone(self, db: Database) -> None:
        indexes.mark_run_deleted(db, "tomb1")
        assert indexes.is_run_deleted(db, "tomb1") is True

    def test_clear_tombstone(self, db: Database) -> None:
        indexes.mark_run_deleted(db, "tomb2")
        indexes.clear_run_tombstone(db, "tomb2")
        assert indexes.is_run_deleted(db, "tomb2") is False

    def test_nonexistent_tombstone(self, db: Database) -> None:
        assert indexes.is_run_deleted(db, "nonexistent") is False


class TestListTombstones:
    def test_no_tombstones(self, db: Database) -> None:
        result = indexes.list_tombstones(db)
        assert result == []

    def test_single_tombstone(self, db: Database) -> None:
        indexes.mark_run_deleted(db, "lt1")
        result = indexes.list_tombstones(db)
        assert len(result) == 1
        run_hash, ts = result[0]
        assert run_hash == "lt1"
        assert isinstance(ts, float)
        assert ts > 0

    def test_multiple_tombstones(self, db: Database) -> None:
        indexes.mark_run_deleted(db, "lt_a")
        indexes.mark_run_deleted(db, "lt_b")
        indexes.mark_run_deleted(db, "lt_c")
        result = indexes.list_tombstones(db)
        hashes = {rh for rh, _ in result}
        assert hashes == {"lt_a", "lt_b", "lt_c"}
        for _, ts in result:
            assert isinstance(ts, float)
            assert ts > 0

    def test_correct_timestamp_decoding(self, db: Database) -> None:
        before = time.time()
        indexes.mark_run_deleted(db, "lt_ts")
        after = time.time()
        result = indexes.list_tombstones(db)
        ts_map = dict(result)
        assert before <= ts_map["lt_ts"] <= after

    def test_corrupt_value_does_not_crash(self, db: Database) -> None:

        idx = get_directories().indexes
        db.clear_range(b"\x00", b"\xff")

        ensure_directories(db)
        idx = get_directories().indexes

        @fdb.transactional
        def write_corrupt(tr: object) -> None:
            tr[idx.pack(("_deleted", "corrupt_run"))] = b"\xff\xff\xff"  # type: ignore[index]

        write_corrupt(db)
        result = indexes.list_tombstones(db)
        ts_map = dict(result)
        assert "corrupt_run" in ts_map
        assert ts_map["corrupt_run"] == 0.0


class TestRebuildIndexesWithHparams:
    def test_rebuild_includes_hparams(self, db: Database) -> None:
        runs.create_run(db, "rbhp1")
        runs.set_run_attrs(db, "rbhp1", ("hparams",), {"lr": 0.001, "batch_size": 64})

        indexes._clear_all_indexes(db)
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.001) == []

        indexes.rebuild_indexes(db)
        assert "rbhp1" in indexes.lookup_by_hparam_eq(db, "lr", 0.001)
        assert "rbhp1" in indexes.lookup_by_hparam_eq(db, "batch_size", 64)


class TestHparamWriteTimeMaintenance:
    def test_set_run_attrs_hparams_indexes(self, db: Database) -> None:
        runs.create_run(db, "wm1")
        runs.set_run_attrs(db, "wm1", ("hparams",), {"lr": 0.001, "epochs": 50})

        assert "wm1" in indexes.lookup_by_hparam_eq(db, "lr", 0.001)
        assert "wm1" in indexes.lookup_by_hparam_eq(db, "epochs", 50)

    def test_set_run_attrs_hparams_replaces_old_indexes(self, db: Database) -> None:
        runs.create_run(db, "wm2")
        runs.set_run_attrs(db, "wm2", ("hparams",), {"lr": 0.001})
        assert "wm2" in indexes.lookup_by_hparam_eq(db, "lr", 0.001)

        runs.set_run_attrs(db, "wm2", ("hparams",), {"lr": 0.01})
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.001) == []
        assert "wm2" in indexes.lookup_by_hparam_eq(db, "lr", 0.01)

    def test_set_run_attrs_non_hparams_no_index(self, db: Database) -> None:
        runs.create_run(db, "wm3")
        runs.set_run_attrs(db, "wm3", ("other",), {"x": 42})
        assert indexes.lookup_by_hparam_eq(db, "x", 42) == []


# ---------------------------------------------------------------------------
# Reverse index tests
# ---------------------------------------------------------------------------


class TestReverseIndexEntries:
    def test_index_run_writes_reverse_entries(self, db: Database) -> None:
        indexes.index_run(
            db,
            "rev1",
            is_archived=False,
            active=True,
            created_at=100.0,
            experiment_name="exp",
            tag_names=["t1"],
        )
        assert _has_rev_entries(db, "rev1")
        assert _count_rev_entries(db, "rev1") == 5

    def test_index_hparams_writes_reverse_entries(self, db: Database) -> None:
        indexes.index_hparams(db, "revhp1", {"lr": 0.001, "bs": 32})
        assert _count_rev_entries(db, "revhp1") == 2

    def test_deindex_run_clears_reverse_entries(self, db: Database) -> None:
        indexes.index_run(db, "rev2", created_at=100.0, tag_names=["t"])
        indexes.index_hparams(db, "rev2", {"lr": 0.01})
        assert _has_rev_entries(db, "rev2")

        indexes.deindex_run(db, "rev2")
        assert not _has_rev_entries(db, "rev2")

    def test_deindex_only_affects_target_run(self, db: Database) -> None:
        """Deindexing one run must not affect another run's entries."""
        indexes.index_run(db, "keep1", created_at=100.0, experiment_name="shared")
        indexes.index_run(db, "drop1", created_at=200.0, experiment_name="shared")
        indexes.index_hparams(db, "keep1", {"lr": 0.1})
        indexes.index_hparams(db, "drop1", {"lr": 0.2})

        indexes.deindex_run(db, "drop1")

        assert indexes.lookup_by_experiment(db, "shared") == ["keep1"]
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.1) == ["keep1"]
        assert indexes.lookup_by_hparam_eq(db, "lr", 0.2) == []
        assert _has_rev_entries(db, "keep1")
        assert not _has_rev_entries(db, "drop1")

    def test_update_index_field_maintains_reverse(self, db: Database) -> None:
        indexes.index_run(db, "uf1", is_archived=False, created_at=100.0)
        rev_before = _count_rev_entries(db, "uf1")

        indexes.update_index_field(db, "uf1", "archived", False, True)
        rev_after = _count_rev_entries(db, "uf1")
        assert rev_after == rev_before

    def test_rename_experiment_updates_reverse(self, db: Database) -> None:
        indexes.index_run(db, "re1", created_at=100.0, experiment_name="old")
        indexes.rename_experiment_index(db, "old", "new")

        assert _has_rev_entries(db, "re1")
        assert indexes.lookup_by_experiment(db, "new") == ["re1"]

    def test_add_remove_tag_index_reverse(self, db: Database) -> None:
        indexes.add_tag_index(db, "trev1", "my-tag")
        assert _has_rev_entries(db, "trev1")

        indexes.remove_tag_index(db, "trev1", "my-tag")
        assert not _has_rev_entries(db, "trev1")

    def test_remove_all_tag_indexes_cleans_reverse(self, db: Database) -> None:
        indexes.add_tag_index(db, "bulk1", "bulk-tag")
        indexes.add_tag_index(db, "bulk2", "bulk-tag")
        assert _has_rev_entries(db, "bulk1")

        indexes.remove_all_tag_indexes_for_tag(db, "bulk-tag")
        assert not _has_rev_entries(db, "bulk1")
        assert not _has_rev_entries(db, "bulk2")

    def test_rebuild_creates_reverse_entries(self, db: Database) -> None:
        runs.create_run(db, "rbre1")
        indexes._clear_all_indexes(db)
        assert not _has_rev_entries(db, "rbre1")

        indexes.rebuild_indexes(db)
        assert _has_rev_entries(db, "rbre1")


# ---------------------------------------------------------------------------
# Tier 3 — metric trace name indexes
# ---------------------------------------------------------------------------


class TestTraceIndex:
    def test_index_trace_and_lookup(self, db: Database) -> None:
        indexes.index_trace(db, "tr1", "loss")
        indexes.index_trace(db, "tr2", "loss")
        indexes.index_trace(db, "tr2", "accuracy")

        assert sorted(indexes.lookup_by_trace_name(db, "loss")) == ["tr1", "tr2"]
        assert indexes.lookup_by_trace_name(db, "accuracy") == ["tr2"]
        assert indexes.lookup_by_trace_name(db, "nonexistent") == []

    def test_deindex_traces(self, db: Database) -> None:
        indexes.index_trace(db, "dtr1", "loss")
        indexes.index_trace(db, "dtr1", "acc")
        assert indexes.lookup_by_trace_name(db, "loss") == ["dtr1"]

        indexes.deindex_traces(db, "dtr1")
        assert indexes.lookup_by_trace_name(db, "loss") == []
        assert indexes.lookup_by_trace_name(db, "acc") == []

    def test_deindex_run_also_removes_traces(self, db: Database) -> None:
        indexes.index_run(db, "dtrt1", created_at=1.0)
        indexes.index_trace(db, "dtrt1", "loss")
        assert indexes.lookup_by_trace_name(db, "loss") == ["dtrt1"]

        indexes.deindex_run(db, "dtrt1")
        assert indexes.lookup_by_trace_name(db, "loss") == []

    def test_rebuild_includes_traces(self, db: Database) -> None:
        runs.create_run(db, "rbtr1")
        runs.set_context(db, "rbtr1", 0, {})
        runs.set_trace_info(db, "rbtr1", 0, "loss", dtype="float", last=0.5)
        runs.set_trace_info(db, "rbtr1", 0, "acc", dtype="float", last=0.9)

        indexes._clear_all_indexes(db)
        assert indexes.lookup_by_trace_name(db, "loss") == []

        indexes.rebuild_indexes(db)
        assert "rbtr1" in indexes.lookup_by_trace_name(db, "loss")
        assert "rbtr1" in indexes.lookup_by_trace_name(db, "acc")

    def test_reverse_entries_for_traces(self, db: Database) -> None:
        indexes.index_trace(db, "revtr1", "loss")
        indexes.index_trace(db, "revtr1", "acc")
        assert _count_rev_entries(db, "revtr1") == 2

        indexes.deindex_traces(db, "revtr1")
        assert not _has_rev_entries(db, "revtr1")

    def test_set_trace_info_creates_index(self, db: Database) -> None:
        """set_trace_info in runs.py should automatically index the trace name."""
        runs.create_run(db, "sti1")
        runs.set_context(db, "sti1", 0, {})
        runs.set_trace_info(db, "sti1", 0, "my_metric", dtype="float", last=1.0)

        assert "sti1" in indexes.lookup_by_trace_name(db, "my_metric")

    def test_deindex_only_affects_target_run(self, db: Database) -> None:
        indexes.index_trace(db, "keep_tr", "loss")
        indexes.index_trace(db, "drop_tr", "loss")

        indexes.deindex_traces(db, "drop_tr")

        assert indexes.lookup_by_trace_name(db, "loss") == ["keep_tr"]
