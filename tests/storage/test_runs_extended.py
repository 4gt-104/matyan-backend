"""Extended tests for storage/runs.py — resume, artifacts, list_runs_meta."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs
from matyan_backend.storage.indexes import lookup_by_hparam_eq

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestResumeRun:
    def test_resume_existing(self, db: Database) -> None:
        runs.create_run(db, "resume1")
        runs.update_run_meta(db, "resume1", active=False, finalized_at=1000.0)

        result = runs.resume_run(db, "resume1")
        assert result is not None
        assert result["active"] is True
        assert result["finalized_at"] is None

    def test_resume_nonexistent_returns_none(self, db: Database) -> None:
        assert runs.resume_run(db, "nonexistent-resume") is None


class TestGetRunArtifacts:
    def test_no_blobs(self, db: Database) -> None:
        runs.create_run(db, "art1")
        assert runs.get_run_artifacts(db, "art1") == []

    def test_with_file_artifacts(self, db: Database) -> None:
        runs.create_run(db, "art2")
        runs.set_run_attrs(
            db,
            "art2",
            ("__blobs__", "data/model.pt"),
            {
                "s3_key": "art2/model.pt",
                "content_type": "application/octet-stream",
            },
        )
        runs.set_run_attrs(
            db,
            "art2",
            ("__blobs__", "config.yaml"),
            {
                "s3_key": "art2/config.yaml",
                "content_type": "text/yaml",
            },
        )

        artifacts = runs.get_run_artifacts(db, "art2")
        assert len(artifacts) == 2
        names = {a["name"] for a in artifacts}
        assert "model.pt" in names
        assert "config.yaml" in names

    def test_seq_prefix_excluded(self, db: Database) -> None:
        runs.create_run(db, "art3")
        runs.set_run_attrs(
            db,
            "art3",
            ("__blobs__", "seq/images/0"),
            {
                "s3_key": "art3/images/0",
            },
        )
        runs.set_run_attrs(
            db,
            "art3",
            ("__blobs__", "weights.bin"),
            {
                "s3_key": "art3/weights.bin",
            },
        )
        artifacts = runs.get_run_artifacts(db, "art3")
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "weights.bin"

    def test_non_dict_meta_skipped(self, db: Database) -> None:
        runs.create_run(db, "art4")
        runs.set_run_attrs(db, "art4", ("__blobs__",), {"bad_entry": "not_a_dict"})
        artifacts = runs.get_run_artifacts(db, "art4")
        assert artifacts == []


class TestListRunsMeta:
    def test_returns_all_with_hash(self, db: Database) -> None:
        runs.create_run(db, "lrm1", name="Run A")
        runs.create_run(db, "lrm2", name="Run B")

        result = runs.list_runs_meta(db)
        hashes = {r["hash"] for r in result}
        assert "lrm1" in hashes
        assert "lrm2" in hashes
        for r in result:
            assert "name" in r
            assert "hash" in r


class TestCreateRunWithExperiment:
    def test_create_with_experiment_id(self, db: Database) -> None:
        exp = entities.create_experiment(db, "baseline-exp")
        result = runs.create_run(db, "crwe1", experiment_id=exp["id"])
        assert result["experiment_id"] == exp["id"]


class TestSetRunAttrsHparamIndexing:
    def test_set_hparams_top_level_indexes(self, db: Database) -> None:
        runs.create_run(db, "hpi1")
        runs.set_run_attrs(db, "hpi1", ("hparams",), {"lr": 0.01, "batch_size": 32})

        found = lookup_by_hparam_eq(db, "lr", 0.01)
        assert "hpi1" in found

    def test_set_nested_hparam_reindexes(self, db: Database) -> None:
        runs.create_run(db, "hpi2")
        runs.set_run_attrs(db, "hpi2", ("hparams",), {"lr": 0.1})
        assert "hpi2" in lookup_by_hparam_eq(db, "lr", 0.1)

        runs.set_run_attrs(db, "hpi2", ("hparams", "lr"), 0.01)
        found_new = lookup_by_hparam_eq(db, "lr", 0.01)
        assert "hpi2" in found_new
        found_old = lookup_by_hparam_eq(db, "lr", 0.1)
        assert "hpi2" not in found_old

    def test_non_hparams_path_no_index(self, db: Database) -> None:
        runs.create_run(db, "hpi3")
        runs.set_run_attrs(db, "hpi3", ("model_config",), {"layers": 3})
        found = lookup_by_hparam_eq(db, "layers", 3)
        assert "hpi3" not in found
