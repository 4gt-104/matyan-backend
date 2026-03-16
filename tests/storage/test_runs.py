"""Tests for runs.py — run CRUD, attributes, traces, contexts, tags."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestRunCRUD:
    def test_create_and_get(self, db: Database) -> None:
        result = runs.create_run(db, "abc123", name="My Run")
        assert result["hash"] == "abc123"
        assert result["name"] == "My Run"
        assert result["active"] is True
        assert result["is_archived"] is False

        fetched = runs.get_run(db, "abc123")
        assert fetched is not None
        assert fetched["hash"] == "abc123"
        assert fetched["name"] == "My Run"

    def test_get_nonexistent(self, db: Database) -> None:
        assert runs.get_run(db, "nonexistent") is None

    def test_create_default_name(self, db: Database) -> None:
        result = runs.create_run(db, "xyz789")
        assert result["name"] == "Run: xyz789"

    def test_update_meta(self, db: Database) -> None:
        runs.create_run(db, "upd1")
        runs.update_run_meta(db, "upd1", name="Updated", is_archived=True)
        fetched = runs.get_run(db, "upd1")
        assert fetched is not None
        assert fetched["name"] == "Updated"
        assert fetched["is_archived"] is True

    def test_delete(self, db: Database) -> None:
        runs.create_run(db, "del1")
        assert runs.get_run(db, "del1") is not None
        runs.delete_run(db, "del1")
        assert runs.get_run(db, "del1") is None

    def test_list_run_hashes(self, db: Database) -> None:
        runs.create_run(db, "list1")
        runs.create_run(db, "list2")
        hashes = runs.list_run_hashes(db)
        assert "list1" in hashes
        assert "list2" in hashes


class TestRunAttrs:
    def test_set_and_get_attrs(self, db: Database) -> None:
        runs.create_run(db, "attr1")
        hparams = {"lr": 0.01, "batch_size": 32, "layers": [64, 128]}
        runs.set_run_attrs(db, "attr1", ("hparams",), hparams)

        result = runs.get_run_attrs(db, "attr1", ("hparams",))
        assert result == hparams

    def test_get_nested_attr(self, db: Database) -> None:
        runs.create_run(db, "attr2")
        runs.set_run_attrs(db, "attr2", (), {"model": {"name": "resnet", "depth": 50}})

        assert runs.get_run_attrs(db, "attr2", ("model", "name")) == "resnet"

    def test_get_all_attrs(self, db: Database) -> None:
        runs.create_run(db, "attr3")
        runs.set_run_attrs(db, "attr3", (), {"a": 1, "b": 2})
        result = runs.get_run_attrs(db, "attr3")
        assert result == {"a": 1, "b": 2}

    def test_get_attrs_nonexistent(self, db: Database) -> None:
        runs.create_run(db, "attr4")
        assert runs.get_run_attrs(db, "attr4") is None


class TestTraceInfo:
    def test_set_and_get_traces(self, db: Database) -> None:
        runs.create_run(db, "trace1")
        runs.set_trace_info(db, "trace1", 0, "loss", dtype="float", last=0.05, last_step=100)
        runs.set_trace_info(db, "trace1", 0, "acc", dtype="float", last=0.95, last_step=100)

        traces = runs.get_run_traces_info(db, "trace1")
        assert len(traces) == 2
        names = {t["name"] for t in traces}
        assert names == {"loss", "acc"}

    def test_empty_traces(self, db: Database) -> None:
        runs.create_run(db, "trace2")
        assert runs.get_run_traces_info(db, "trace2") == []


class TestContexts:
    def test_set_and_get_context(self, db: Database) -> None:
        runs.create_run(db, "ctx1")
        ctx = {"subset": "train", "augmented": True}
        runs.set_context(db, "ctx1", 42, ctx)

        result = runs.get_context(db, "ctx1", 42)
        assert result == ctx

    def test_get_nonexistent_context(self, db: Database) -> None:
        runs.create_run(db, "ctx2")
        assert runs.get_context(db, "ctx2", 999) is None

    def test_get_all_contexts(self, db: Database) -> None:
        runs.create_run(db, "ctx3")
        runs.set_context(db, "ctx3", 1, {"a": 1})
        runs.set_context(db, "ctx3", 2, {"b": 2})
        all_ctx = runs.get_all_contexts(db, "ctx3")
        assert len(all_ctx) == 2
        assert all_ctx[1] == {"a": 1}
        assert all_ctx[2] == {"b": 2}


class TestRunTags:
    def test_add_and_get_tags(self, db: Database) -> None:
        runs.create_run(db, "tag1")
        runs.add_tag_to_run(db, "tag1", "uuid-a")
        runs.add_tag_to_run(db, "tag1", "uuid-b")

        uuids = runs.get_run_tag_uuids(db, "tag1")
        assert set(uuids) == {"uuid-a", "uuid-b"}

    def test_remove_tag(self, db: Database) -> None:
        runs.create_run(db, "tag2")
        runs.add_tag_to_run(db, "tag2", "uuid-x")
        runs.remove_tag_from_run(db, "tag2", "uuid-x")

        assert runs.get_run_tag_uuids(db, "tag2") == []

    def test_set_experiment(self, db: Database) -> None:
        runs.create_run(db, "exp1")
        runs.set_run_experiment(db, "exp1", "exp-uuid-1")
        meta = runs.get_run_meta(db, "exp1")
        assert meta["experiment_id"] == "exp-uuid-1"


class TestPendingDeletion:
    def test_mark_and_check(self, db: Database) -> None:
        runs.create_run(db, "pd1")
        assert runs.is_pending_deletion(db, "pd1") is False
        runs.mark_pending_deletion(db, "pd1")
        assert runs.is_pending_deletion(db, "pd1") is True

    def test_is_pending_deletion_nonexistent(self, db: Database) -> None:
        assert runs.is_pending_deletion(db, "nonexistent") is False

    def test_pending_deletion_in_meta(self, db: Database) -> None:
        """The flag is stored under meta so get_run_meta picks it up."""
        runs.create_run(db, "pd2")
        runs.mark_pending_deletion(db, "pd2")
        meta = runs.get_run_meta(db, "pd2")
        assert meta.get("pending_deletion") is True

    def test_delete_run_clears_pending(self, db: Database) -> None:
        """After full deletion the pending flag is gone along with all data."""
        runs.create_run(db, "pd3")
        runs.mark_pending_deletion(db, "pd3")
        runs.delete_run(db, "pd3")
        assert runs.get_run(db, "pd3") is None
        assert runs.is_pending_deletion(db, "pd3") is False


class TestGetRunBundle:
    def test_returns_all_data(self, db: Database) -> None:
        exp = entities.create_experiment(db, "bundle-exp")
        runs.create_run(db, "bun1", experiment_id=exp["id"])
        runs.set_run_attrs(db, "bun1", (), {"hparams": {"lr": 0.01}})
        runs.set_context(db, "bun1", 0, {"subset": "train"})
        runs.set_trace_info(db, "bun1", 0, "loss", dtype="float", last=0.5, last_step=9)
        tag = entities.create_tag(db, "bundle-tag")
        entities.add_tag_to_run(db, "bun1", tag["id"])

        bundle = runs.get_run_bundle(db, "bun1")
        assert bundle is not None
        assert bundle["meta"]["name"] == "Run: bun1"
        assert bundle["attrs"] == {"hparams": {"lr": 0.01}}
        assert len(bundle["traces"]) == 1
        assert bundle["traces"][0]["name"] == "loss"
        assert bundle["contexts"][0] == {"subset": "train"}
        assert len(bundle["tags"]) == 1
        assert bundle["tags"][0]["name"] == "bundle-tag"
        assert bundle["experiment"] is not None
        assert bundle["experiment"]["name"] == "bundle-exp"

    def test_nonexistent_run(self, db: Database) -> None:
        assert runs.get_run_bundle(db, "nonexistent") is None

    def test_pending_deletion(self, db: Database) -> None:
        runs.create_run(db, "bun2")
        runs.mark_pending_deletion(db, "bun2")
        assert runs.get_run_bundle(db, "bun2") is None

    def test_exclude_attrs(self, db: Database) -> None:
        runs.create_run(db, "bun3")
        runs.set_run_attrs(db, "bun3", (), {"x": 1})
        bundle = runs.get_run_bundle(db, "bun3", include_attrs=False)
        assert bundle is not None
        assert bundle["attrs"] is None

    def test_exclude_traces(self, db: Database) -> None:
        runs.create_run(db, "bun4")
        runs.set_context(db, "bun4", 0, {})
        runs.set_trace_info(db, "bun4", 0, "loss", dtype="float")
        bundle = runs.get_run_bundle(db, "bun4", include_traces=False)
        assert bundle is not None
        assert bundle["traces"] == []
        assert bundle["contexts"] == {}

    def test_no_experiment(self, db: Database) -> None:
        runs.create_run(db, "bun5")
        bundle = runs.get_run_bundle(db, "bun5")
        assert bundle is not None
        assert bundle["experiment"] is None
        assert bundle["tags"] == []


class TestGetRunBundles:
    """Tests for the batched multi-run bundle function."""

    def test_returns_matching_bundles(self, db: Database) -> None:
        runs.create_run(db, "buns1")
        runs.create_run(db, "buns2")
        results = runs.get_run_bundles(db, ["buns1", "buns2"])
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None
        assert results[0]["meta"]["name"] == "Run: buns1"
        assert results[1]["meta"]["name"] == "Run: buns2"

    def test_none_for_missing_run(self, db: Database) -> None:
        runs.create_run(db, "buns3")
        results = runs.get_run_bundles(db, ["buns3", "nonexistent"])
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    def test_none_for_deleted_run(self, db: Database) -> None:
        runs.create_run(db, "buns4")
        runs.create_run(db, "buns5")
        runs.mark_pending_deletion(db, "buns5")
        results = runs.get_run_bundles(db, ["buns4", "buns5"])
        assert results[0] is not None
        assert results[1] is None

    def test_matches_single_bundle(self, db: Database) -> None:
        """Batch result should be identical to single get_run_bundle."""
        exp = entities.create_experiment(db, "buns-exp")
        runs.create_run(db, "buns6", experiment_id=exp["id"])
        runs.set_run_attrs(db, "buns6", (), {"lr": 0.1})
        runs.set_context(db, "buns6", 0, {"split": "train"})
        runs.set_trace_info(db, "buns6", 0, "loss", dtype="float", last=0.3)
        tag = entities.create_tag(db, "buns-tag")
        entities.add_tag_to_run(db, "buns6", tag["id"])

        single = runs.get_run_bundle(db, "buns6")
        batched = runs.get_run_bundles(db, ["buns6"])
        assert len(batched) == 1
        assert batched[0] is not None
        assert single is not None
        assert batched[0]["meta"] == single["meta"]
        assert batched[0]["attrs"] == single["attrs"]
        assert batched[0]["traces"] == single["traces"]
        assert batched[0]["contexts"] == single["contexts"]
        assert len(batched[0]["tags"]) == len(single["tags"])
        assert batched[0]["experiment"]["name"] == single["experiment"]["name"]

    def test_empty_list(self, db: Database) -> None:
        results = runs.get_run_bundles(db, [])
        assert results == []

    def test_exclude_flags(self, db: Database) -> None:
        runs.create_run(db, "buns7")
        runs.set_run_attrs(db, "buns7", (), {"x": 1})
        runs.set_context(db, "buns7", 0, {})
        runs.set_trace_info(db, "buns7", 0, "loss", dtype="float")

        results = runs.get_run_bundles(
            db,
            ["buns7"],
            include_attrs=False,
            include_traces=False,
        )
        assert results[0] is not None
        assert results[0]["attrs"] is None
        assert results[0]["traces"] == []
        assert results[0]["contexts"] == {}


class TestGetMetricSearchBundle:
    """Tests for the metric-search-specific single-run bundle helper."""

    def test_matches_general_bundle_shape(self, db: Database) -> None:
        exp = entities.create_experiment(db, "metric-bundle-exp")
        runs.create_run(db, "metricbun1", experiment_id=exp["id"])
        runs.set_run_attrs(db, "metricbun1", (), {"lr": 0.1})
        runs.set_context(db, "metricbun1", 0, {"split": "train"})
        runs.set_trace_info(db, "metricbun1", 0, "loss", dtype="float", last=0.3)
        tag = entities.create_tag(db, "metric-bundle-tag")
        entities.add_tag_to_run(db, "metricbun1", tag["id"])

        general = runs.get_run_bundle(db, "metricbun1")
        metric = runs.get_metric_search_bundle(db, "metricbun1")

        assert general is not None
        assert metric is not None
        assert metric["meta"] == general["meta"]
        assert metric["attrs"] == general["attrs"]
        assert metric["traces"] == general["traces"]
        assert metric["contexts"] == general["contexts"]
        assert len(metric["tags"]) == len(general["tags"])
        assert metric["experiment"]["name"] == general["experiment"]["name"]


class TestDeleteRunClearsAssociations:
    def test_delete_removes_experiment_association(self, db: Database) -> None:
        exp = entities.create_experiment(db, "assoc-exp")
        runs.create_run(db, "assoc1", experiment_id=exp["id"])
        entities.set_run_experiment(db, "assoc1", exp["id"])
        assert "assoc1" in entities.get_runs_for_experiment(db, exp["id"])

        runs.delete_run(db, "assoc1")
        assert "assoc1" not in entities.get_runs_for_experiment(db, exp["id"])

    def test_delete_removes_tag_association(self, db: Database) -> None:
        runs.create_run(db, "assoc2")
        tag = entities.create_tag(db, "assoc-tag")
        entities.add_tag_to_run(db, "assoc2", tag["id"])
        runs.add_tag_to_run(db, "assoc2", tag["id"])
        assert "assoc2" in entities.get_runs_for_tag(db, tag["id"])

        runs.delete_run(db, "assoc2")
        assert "assoc2" not in entities.get_runs_for_tag(db, tag["id"])
        assert entities.get_tags_for_run(db, "assoc2") == []
