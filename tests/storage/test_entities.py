"""Tests for entities.py — experiments, tags, dashboards, apps, reports, notes + associations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matyan_backend.fdb_types import transactional
from matyan_backend.storage import encoding, entities
from matyan_backend.storage.fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database, Transaction


@transactional
def _read_run_experiment_reverse(tr: Transaction, run_hash: str) -> str | None:
    """Read the (run_experiment, run_hash) reverse key and return the exp uuid."""
    sd = get_directories().system
    key = sd.pack(("run_experiment", run_hash))
    raw = tr[key]
    if raw.present():
        return encoding.decode_value(raw)
    return None


class TestExperiments:
    def test_create_and_get(self, db: Database) -> None:
        exp = entities.create_experiment(db, "baseline", description="first try")
        assert exp["name"] == "baseline"
        assert exp["run_count"] == 0
        assert "id" in exp

        fetched = entities.get_experiment(db, exp["id"])
        assert fetched is not None
        assert fetched["name"] == "baseline"
        assert fetched["description"] == "first try"
        assert fetched["run_count"] == 0

    def test_get_by_name(self, db: Database) -> None:
        exp = entities.create_experiment(db, "byname")
        fetched = entities.get_experiment_by_name(db, "byname")
        assert fetched is not None
        assert fetched["id"] == exp["id"]

    def test_duplicate_name_raises(self, db: Database) -> None:
        entities.create_experiment(db, "dup")
        with pytest.raises(ValueError, match="already exists"):
            entities.create_experiment(db, "dup")

    def test_list(self, db: Database) -> None:
        entities.create_experiment(db, "exp_a")
        entities.create_experiment(db, "exp_b")
        exps = entities.list_experiments(db)
        names = {e["name"] for e in exps}
        assert "exp_a" in names
        assert "exp_b" in names

    def test_update(self, db: Database) -> None:
        exp = entities.create_experiment(db, "to_update")
        entities.update_experiment(db, exp["id"], description="updated desc")
        fetched = entities.get_experiment(db, exp["id"])
        assert fetched is not None
        assert fetched["description"] == "updated desc"

    def test_rename(self, db: Database) -> None:
        exp = entities.create_experiment(db, "old_name")
        entities.update_experiment(db, exp["id"], name="new_name")

        assert entities.get_experiment_by_name(db, "old_name") is None
        assert entities.get_experiment_by_name(db, "new_name") is not None

    def test_delete(self, db: Database) -> None:
        exp = entities.create_experiment(db, "to_delete")
        entities.delete_experiment(db, exp["id"])
        assert entities.get_experiment(db, exp["id"]) is None
        assert entities.get_experiment_by_name(db, "to_delete") is None

    def test_get_nonexistent(self, db: Database) -> None:
        assert entities.get_experiment(db, "nonexistent-uuid") is None
        assert entities.get_experiment_by_name(db, "nonexistent") is None


class TestTags:
    def test_create_and_get(self, db: Database) -> None:
        tag = entities.create_tag(db, "important", color="#ff0000")
        assert tag["name"] == "important"
        assert tag["color"] == "#ff0000"
        assert tag["run_count"] == 0

        fetched = entities.get_tag(db, tag["id"])
        assert fetched is not None
        assert fetched["name"] == "important"
        assert fetched["run_count"] == 0

    def test_duplicate_name_raises(self, db: Database) -> None:
        entities.create_tag(db, "dup_tag")
        with pytest.raises(ValueError, match="already exists"):
            entities.create_tag(db, "dup_tag")

    def test_list(self, db: Database) -> None:
        entities.create_tag(db, "tag_x")
        entities.create_tag(db, "tag_y")
        tags = entities.list_tags(db)
        names = {t["name"] for t in tags}
        assert "tag_x" in names
        assert "tag_y" in names

    def test_delete_cleans_associations(self, db: Database) -> None:
        tag = entities.create_tag(db, "cleanup_tag")
        entities.add_tag_to_run(db, "run_hash_1", tag["id"])

        entities.delete_tag(db, tag["id"])

        assert entities.get_tag(db, tag["id"]) is None
        assert entities.get_tags_for_run(db, "run_hash_1") == []


class TestRunTagAssociations:
    def test_add_and_get_tags_for_run(self, db: Database) -> None:
        t1 = entities.create_tag(db, "assoc_a")
        t2 = entities.create_tag(db, "assoc_b")

        entities.add_tag_to_run(db, "run1", t1["id"])
        entities.add_tag_to_run(db, "run1", t2["id"])

        tags = entities.get_tags_for_run(db, "run1")
        tag_names = {t["name"] for t in tags}
        assert tag_names == {"assoc_a", "assoc_b"}

    def test_get_runs_for_tag(self, db: Database) -> None:
        tag = entities.create_tag(db, "multi_run")
        entities.add_tag_to_run(db, "r1", tag["id"])
        entities.add_tag_to_run(db, "r2", tag["id"])

        run_hashes = entities.get_runs_for_tag(db, tag["id"])
        assert set(run_hashes) == {"r1", "r2"}

    def test_remove_tag_from_run(self, db: Database) -> None:
        tag = entities.create_tag(db, "removable")
        entities.add_tag_to_run(db, "r3", tag["id"])
        entities.remove_tag_from_run(db, "r3", tag["id"])

        assert entities.get_tags_for_run(db, "r3") == []
        assert entities.get_runs_for_tag(db, tag["id"]) == []


class TestExperimentRunAssociations:
    def test_set_and_get_runs_for_experiment(self, db: Database) -> None:
        exp = entities.create_experiment(db, "assoc_exp")
        entities.set_run_experiment(db, "ra", exp["id"])
        entities.set_run_experiment(db, "rb", exp["id"])

        run_hashes = entities.get_runs_for_experiment(db, exp["id"])
        assert set(run_hashes) == {"ra", "rb"}

    def test_change_experiment(self, db: Database) -> None:
        exp1 = entities.create_experiment(db, "from_exp")
        exp2 = entities.create_experiment(db, "to_exp")

        entities.set_run_experiment(db, "rc", exp1["id"])
        entities.set_run_experiment(db, "rc", exp2["id"])

        assert "rc" not in entities.get_runs_for_experiment(db, exp1["id"])
        assert "rc" in entities.get_runs_for_experiment(db, exp2["id"])

    def test_remove_experiment(self, db: Database) -> None:
        exp = entities.create_experiment(db, "remove_exp")
        entities.set_run_experiment(db, "rd", exp["id"])
        entities.set_run_experiment(db, "rd", None)

        assert entities.get_runs_for_experiment(db, exp["id"]) == []


class TestDashboards:
    def test_crud(self, db: Database) -> None:
        dash = entities.create_dashboard(db, "my dash", description="desc")
        assert dash["name"] == "my dash"

        fetched = entities.get_dashboard(db, dash["id"])
        assert fetched is not None
        assert fetched["name"] == "my dash"

        entities.update_dashboard(db, dash["id"], name="renamed")
        updated = entities.get_dashboard(db, dash["id"])
        assert updated is not None
        assert updated["name"] == "renamed"

        entities.delete_dashboard(db, dash["id"])
        assert entities.get_dashboard(db, dash["id"]) is None

    def test_list(self, db: Database) -> None:
        entities.create_dashboard(db, "d1")
        entities.create_dashboard(db, "d2")
        names = {d["name"] for d in entities.list_dashboards(db)}
        assert "d1" in names
        assert "d2" in names


class TestDashboardApps:
    def test_crud(self, db: Database) -> None:
        app = entities.create_dashboard_app(db, "metrics_explorer", {"view": "table"})
        assert app["type"] == "metrics_explorer"
        assert app["state"] == {"view": "table"}

        fetched = entities.get_dashboard_app(db, app["id"])
        assert fetched is not None
        assert fetched["type"] == "metrics_explorer"

        entities.update_dashboard_app(db, app["id"], state={"view": "chart"})

        entities.delete_dashboard_app(db, app["id"])
        assert entities.get_dashboard_app(db, app["id"]) is None


class TestReports:
    def test_crud(self, db: Database) -> None:
        report = entities.create_report(db, "weekly", code="print('hi')")
        assert report["name"] == "weekly"

        fetched = entities.get_report(db, report["id"])
        assert fetched is not None
        assert fetched["code"] == "print('hi')"

        entities.update_report(db, report["id"], code="print('updated')")
        updated = entities.get_report(db, report["id"])
        assert updated is not None
        assert updated["code"] == "print('updated')"

        entities.delete_report(db, report["id"])
        assert entities.get_report(db, report["id"]) is None


class TestNotes:
    def test_crud(self, db: Database) -> None:
        note = entities.create_note(db, "my note", run_hash="run1")
        assert note["content"] == "my note"
        assert note["run_hash"] == "run1"

        fetched = entities.get_note(db, note["id"])
        assert fetched is not None
        assert fetched["content"] == "my note"

        entities.update_note(db, note["id"], content="updated")
        updated = entities.get_note(db, note["id"])
        assert updated is not None
        assert updated["content"] == "updated"

        entities.delete_note(db, note["id"])
        assert entities.get_note(db, note["id"]) is None

    def test_list(self, db: Database) -> None:
        entities.create_note(db, "n1")
        entities.create_note(db, "n2")
        notes = entities.list_notes(db)
        contents = {n["content"] for n in notes}
        assert "n1" in contents
        assert "n2" in contents


# ---------------------------------------------------------------------------
# run_count maintenance
# ---------------------------------------------------------------------------


class TestExperimentRunCount:
    def test_run_count_incremented_on_set_experiment(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rc_exp")
        entities.set_run_experiment(db, "rc1", exp["id"])

        fetched = entities.get_experiment(db, exp["id"])
        assert fetched is not None
        assert fetched["run_count"] == 1

    def test_run_count_decremented_on_change(self, db: Database) -> None:
        exp1 = entities.create_experiment(db, "rc_from")
        exp2 = entities.create_experiment(db, "rc_to")
        entities.set_run_experiment(db, "rc2", exp1["id"])
        entities.set_run_experiment(db, "rc2", exp2["id"])

        fetched1 = entities.get_experiment(db, exp1["id"])
        assert fetched1 is not None
        assert fetched1["run_count"] == 0
        fetched2 = entities.get_experiment(db, exp2["id"])
        assert fetched2 is not None
        assert fetched2["run_count"] == 1

    def test_run_count_decremented_on_remove(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rc_rem")
        entities.set_run_experiment(db, "rc3", exp["id"])
        entities.set_run_experiment(db, "rc3", None)

        fetched = entities.get_experiment(db, exp["id"])
        assert fetched is not None
        assert fetched["run_count"] == 0

    def test_run_count_decremented_on_remove_associations(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rc_assoc")
        entities.set_run_experiment(db, "rc4", exp["id"])
        entities.remove_run_associations(db, "rc4")

        fetched = entities.get_experiment(db, exp["id"])
        assert fetched is not None
        assert fetched["run_count"] == 0


class TestTagRunCount:
    def test_run_count_incremented_on_add(self, db: Database) -> None:
        tag = entities.create_tag(db, "tc_tag")
        entities.add_tag_to_run(db, "tc1", tag["id"])

        fetched = entities.get_tag(db, tag["id"])
        assert fetched is not None
        assert fetched["run_count"] == 1

    def test_run_count_decremented_on_remove(self, db: Database) -> None:
        tag = entities.create_tag(db, "tc_rem")
        entities.add_tag_to_run(db, "tc2", tag["id"])
        entities.remove_tag_from_run(db, "tc2", tag["id"])

        fetched = entities.get_tag(db, tag["id"])
        assert fetched is not None
        assert fetched["run_count"] == 0

    def test_run_count_decremented_on_remove_associations(self, db: Database) -> None:
        tag = entities.create_tag(db, "tc_assoc")
        entities.add_tag_to_run(db, "tc3", tag["id"])
        entities.remove_run_associations(db, "tc3")

        fetched = entities.get_tag(db, tag["id"])
        assert fetched is not None
        assert fetched["run_count"] == 0

    def test_run_count_never_negative(self, db: Database) -> None:
        tag = entities.create_tag(db, "tc_neg")
        entities.remove_tag_from_run(db, "tc_x", tag["id"])

        fetched = entities.get_tag(db, tag["id"])
        assert fetched is not None
        assert fetched["run_count"] == 0


# ---------------------------------------------------------------------------
# run_experiment reverse key
# ---------------------------------------------------------------------------


class TestRunExperimentReverseKey:
    def test_reverse_key_written_on_set(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rev_exp")
        entities.set_run_experiment(db, "rev1", exp["id"])

        assert _read_run_experiment_reverse(db, "rev1") == exp["id"]

    def test_reverse_key_updated_on_change(self, db: Database) -> None:
        exp1 = entities.create_experiment(db, "rev_from")
        exp2 = entities.create_experiment(db, "rev_to")
        entities.set_run_experiment(db, "rev2", exp1["id"])
        entities.set_run_experiment(db, "rev2", exp2["id"])

        assert _read_run_experiment_reverse(db, "rev2") == exp2["id"]

    def test_reverse_key_cleared_on_remove(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rev_rem")
        entities.set_run_experiment(db, "rev3", exp["id"])
        entities.set_run_experiment(db, "rev3", None)

        assert _read_run_experiment_reverse(db, "rev3") is None

    def test_reverse_key_cleared_on_remove_associations(self, db: Database) -> None:
        exp = entities.create_experiment(db, "rev_assoc")
        entities.set_run_experiment(db, "rev4", exp["id"])
        entities.remove_run_associations(db, "rev4")

        assert _read_run_experiment_reverse(db, "rev4") is None
