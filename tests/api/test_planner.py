"""Tests for the MatyanQL query planner and preprocessing pipeline."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from matyan_backend.api.runs._planner import (
    PlanResult,
    plan_query,
    query_has_sequence_level_predicate,
    query_has_unindexed_sequence_predicate,
)
from matyan_backend.api.runs._query import prepare_query
from matyan_backend.storage import indexes, runs

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


def _plan(db: Database, raw: str, tz_offset: int = 0) -> PlanResult:
    """Shorthand: prepare_query + plan_query."""
    return plan_query(db, prepare_query(raw, tz_offset))


# ---------------------------------------------------------------------------
# prepare_query tests
# ---------------------------------------------------------------------------


class TestPrepareQuery:
    """Tests for the AST preprocessing pipeline."""

    def test_empty_query_returns_default_ast(self) -> None:
        tree = prepare_query("", 0)
        assert isinstance(tree, ast.Expression)
        src = ast.unparse(tree)
        assert "is_archived" in src
        assert "False" in src

    def test_whitespace_only_returns_default(self) -> None:
        tree = prepare_query("   ", 0)
        src = ast.unparse(tree)
        assert "is_archived" in src

    def test_select_if_stripped(self) -> None:
        tree = prepare_query("SELECT run.hash IF run.active == True", 0)
        src = ast.unparse(tree)
        assert "active" in src

    def test_syntax_error_raised(self) -> None:
        with pytest.raises(SyntaxError):
            prepare_query("invalid ( syntax", 0)

    def test_default_predicate_added_when_missing(self) -> None:
        tree = prepare_query("run.active == True", 0)
        src = ast.unparse(tree)
        assert "is_archived" in src
        assert "active" in src

    def test_default_predicate_not_added_when_archived_present(self) -> None:
        tree = prepare_query("run.is_archived == True", 0)
        src = ast.unparse(tree)
        assert "is_archived" in src
        assert src.count("is_archived") == 1

    def test_default_predicate_not_added_when_archived_alias(self) -> None:
        tree = prepare_query("run.archived == False", 0)
        src = ast.unparse(tree)
        assert "archived" in src

    def test_datetime_rewritten_to_float(self) -> None:
        tree = prepare_query("datetime(2026, 3, 10) <= run.created_at", tz_offset=0)
        src = ast.unparse(tree)
        assert "datetime" not in src
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                assert node.value > 0
                break
        else:
            pytest.fail("Expected a float Constant from datetime rewrite")

    def test_datetime_with_tz_offset(self) -> None:
        tree_utc = prepare_query("datetime(2026, 3, 10) <= run.created_at", tz_offset=0)
        tree_plus3 = prepare_query("datetime(2026, 3, 10) <= run.created_at", tz_offset=180)
        src_utc = ast.unparse(tree_utc)
        src_plus3 = ast.unparse(tree_plus3)
        assert src_utc != src_plus3

    def test_chained_datetime_range(self) -> None:
        tree = prepare_query(
            "datetime(2026, 3, 10) <= run.created_at < datetime(2026, 3, 11)",
            tz_offset=0,
        )
        src = ast.unparse(tree)
        assert "datetime" not in src
        assert " and " in src

    def test_chained_compare_split_into_and(self) -> None:
        tree = prepare_query("1 <= run.created_at < 2", tz_offset=0)
        compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
        assert len(compares) >= 2
        for cmp in compares:
            assert len(cmp.ops) == 1

    def test_single_compare_untouched(self) -> None:
        tree = prepare_query("run.created_at >= datetime(2026, 3, 10)", tz_offset=0)
        compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
        for cmp in compares:
            assert len(cmp.ops) == 1

    def test_three_way_chain(self) -> None:
        tree = prepare_query("1 < run.hparams.lr < 10 < 100", tz_offset=0)
        src = ast.unparse(tree)
        assert src.count(" and ") >= 2

    def test_non_datetime_chained_compare(self) -> None:
        tree = prepare_query("1 < run.hparams.lr <= 10", tz_offset=0)
        compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
        assert len(compares) >= 2
        for cmp in compares:
            assert len(cmp.ops) == 1


# ---------------------------------------------------------------------------
# query_has_sequence_level_predicate / query_has_unindexed_sequence_predicate
# ---------------------------------------------------------------------------


class TestQueryHasSequenceLevelPredicate:
    """query_has_sequence_level_predicate determines if metric search should use lazy path."""

    def test_metric_name_returns_true(self) -> None:
        assert query_has_sequence_level_predicate(prepare_query('metric.name == "loss"')) is True
        assert (
            query_has_sequence_level_predicate(
                prepare_query(
                    '(metric.name == "before_grad_2.0_norm_total") or (metric.name == "after_grad_2.0_norm_total")',
                ),
            )
            is True
        )

    def test_run_only_returns_false(self) -> None:
        assert query_has_sequence_level_predicate(prepare_query("")) is False
        assert query_has_sequence_level_predicate(prepare_query('run.experiment == "x"')) is False
        assert query_has_sequence_level_predicate(prepare_query("run.is_archived == False")) is False


class TestQueryHasUnindexedSequencePredicate:
    def test_metric_name_eq_is_indexed(self) -> None:
        assert query_has_unindexed_sequence_predicate(prepare_query('metric.name == "loss"')) is False

    def test_metric_name_startswith_is_unindexed(self) -> None:
        assert query_has_unindexed_sequence_predicate(prepare_query('metric.name.startswith("lo")')) is True

    def test_metric_context_is_unindexed(self) -> None:
        assert query_has_unindexed_sequence_predicate(prepare_query("metric.context.foo == 1")) is True

    def test_no_metric_returns_false(self) -> None:
        assert query_has_unindexed_sequence_predicate(prepare_query("")) is False
        assert query_has_unindexed_sequence_predicate(prepare_query('run.experiment == "x"')) is False

    def test_metric_name_eq_combined_with_run_predicate(self) -> None:
        assert (
            query_has_unindexed_sequence_predicate(
                prepare_query('metric.name == "loss" and run.experiment == "x"'),
            )
            is False
        )

    def test_metric_name_eq_or(self) -> None:
        assert (
            query_has_unindexed_sequence_predicate(
                prepare_query('metric.name == "loss" or metric.name == "acc"'),
            )
            is False
        )


# ---------------------------------------------------------------------------
# PlanResult
# ---------------------------------------------------------------------------


class TestPlanResult:
    """Smoke tests for PlanResult return type."""

    def test_none_candidates(self) -> None:
        r = PlanResult(candidates=None, exact=True)
        assert r.candidates is None
        assert r.exact is True

    def test_exact_candidates(self) -> None:
        r = PlanResult(candidates=["a"], exact=True)
        assert r.candidates == ["a"]
        assert r.exact is True

    def test_superset_candidates(self) -> None:
        r = PlanResult(candidates=["a", "b"], exact=False)
        assert r.candidates == ["a", "b"]
        assert r.exact is False


# ---------------------------------------------------------------------------
# plan_query (via _plan helper)
# ---------------------------------------------------------------------------


class TestPlanQueryRunHash:
    """run.hash intersected with the default archived predicate."""

    def test_double_quotes(self, db: Database) -> None:
        indexes.index_run(db, "abc123", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.hash == "abc123"')
        assert result.candidates == ["abc123"]
        assert result.exact is True

    def test_single_quotes(self, db: Database) -> None:
        indexes.index_run(db, "0cf28fd9b5bf2002", is_archived=False, created_at=1.0)
        result = _plan(db, "run.hash == '0cf28fd9b5bf2002'")
        assert result.candidates == ["0cf28fd9b5bf2002"]
        assert result.exact is True

    def test_hash_and_experiment_intersection(self, db: Database) -> None:
        """AND of run.hash and experiment returns intersection (empty if disjoint)."""
        indexes.index_run(db, "pri_h1", is_archived=False, created_at=1.0, experiment_name="exp-z")
        result = _plan(db, 'run.hash == "single" and run.experiment == "exp-z"')
        assert result.candidates == []
        assert result.exact is True

    def test_hash_and_archived(self, db: Database) -> None:
        """run.hash intersected with the default is_archived == False."""
        indexes.index_run(db, "some_run", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.hash == "some_run"')
        assert result.candidates == ["some_run"]
        assert result.exact is True

    def test_archived_hash_excluded(self, db: Database) -> None:
        """Archived run is excluded even when queried by hash (default filter)."""
        indexes.index_run(db, "arc_run", is_archived=True, created_at=1.0)
        result = _plan(db, 'run.hash == "arc_run"')
        assert result.candidates == []
        assert result.exact is True


class TestPlanQueryUnindexedAttrs:
    """Queries with unindexed attrs (run.name, run.description) in an AND
    with indexed predicates return a superset with exact=False.
    """

    def test_run_name_returns_superset(self, db: Database) -> None:
        indexes.index_run(db, "un1", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.name == "my run"')
        assert result.candidates is not None
        assert "un1" in result.candidates
        assert result.exact is False

    def test_run_description_returns_superset(self, db: Database) -> None:
        indexes.index_run(db, "un2", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.description == "some desc"')
        assert result.candidates is not None
        assert "un2" in result.candidates
        assert result.exact is False

    def test_run_name_with_archived(self, db: Database) -> None:
        """run.name with archived returns superset from archived index."""
        indexes.index_run(db, "un3", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.name == "x" and run.is_archived == False')
        assert result.candidates is not None
        assert "un3" in result.candidates
        assert result.exact is False


class TestPlanQueryPatterns:
    """Verify pattern extraction without FDB (use pre-seeded indexes)."""

    def test_experiment_pattern(self, db: Database) -> None:
        indexes.index_run(db, "x1", created_at=1.0, experiment_name="baseline")
        indexes.index_run(db, "x2", created_at=2.0, experiment_name="v2")

        result = _plan(db, 'run.experiment == "baseline"')
        assert result.candidates == ["x1"]
        assert result.exact is True

    def test_experiment_single_quotes(self, db: Database) -> None:
        indexes.index_run(db, "sq1", created_at=1.0, experiment_name="my-exp")
        result = _plan(db, "run.experiment == 'my-exp'")
        assert result.candidates == ["sq1"]
        assert result.exact is True

    def test_tag_in_pattern(self, db: Database) -> None:
        indexes.index_run(db, "t1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "t2", is_archived=False, created_at=2.0)
        indexes.add_tag_index(db, "t1", "best")
        indexes.add_tag_index(db, "t2", "best")

        result = _plan(db, '"best" in run.tags')
        assert result.candidates is not None
        assert sorted(result.candidates) == ["t1", "t2"]
        assert result.exact is True

    def test_tag_single_quotes(self, db: Database) -> None:
        indexes.index_run(db, "ts1", is_archived=False, created_at=1.0)
        indexes.add_tag_index(db, "ts1", "prod")
        result = _plan(db, "'prod' in run.tags")
        assert result.candidates == ["ts1"]
        assert result.exact is True

    def test_active_true(self, db: Database) -> None:
        indexes.index_run(db, "a1", active=True, created_at=1.0)
        indexes.index_run(db, "a2", active=False, created_at=2.0)

        result = _plan(db, "run.active == True")
        assert result.candidates == ["a1"]
        assert result.exact is True

    def test_active_false(self, db: Database) -> None:
        indexes.index_run(db, "af1", active=True, created_at=1.0)
        indexes.index_run(db, "af2", active=False, created_at=2.0)

        result = _plan(db, "run.active == False")
        assert result.candidates == ["af2"]
        assert result.exact is True

    def test_archived_true(self, db: Database) -> None:
        indexes.index_run(db, "ar1", is_archived=True, created_at=1.0)
        indexes.index_run(db, "ar2", is_archived=False, created_at=2.0)

        result = _plan(db, "run.is_archived == True")
        assert result.candidates == ["ar1"]
        assert result.exact is True

    def test_archived_alias(self, db: Database) -> None:
        indexes.index_run(db, "al1", is_archived=False, created_at=1.0)
        result = _plan(db, "run.archived == False")
        assert result.candidates == ["al1"]
        assert result.exact is True

    def test_default_empty_query_uses_archived_index(self, db: Database) -> None:
        indexes.index_run(db, "d1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "d2", is_archived=True, created_at=2.0)

        result = _plan(db, "")
        assert result.candidates == ["d1"]
        assert result.exact is True

    def test_unrecognized_query_returns_none(self, db: Database) -> None:
        result = _plan(db, "run.is_archived in [True, False]")
        assert result.candidates is None

    def test_priority_experiment_over_archived(self, db: Database) -> None:
        """Experiment is more selective than archived; planner should prefer it."""
        indexes.index_run(db, "p1", is_archived=False, created_at=1.0, experiment_name="exp-a")
        indexes.index_run(db, "p2", is_archived=False, created_at=2.0, experiment_name="exp-b")

        result = _plan(db, 'run.experiment == "exp-a"')
        assert result.candidates == ["p1"]
        assert result.exact is True


class TestPlanQueryHparamPatterns:
    """Test Tier 2 hparam index integration in the query planner."""

    def test_hparam_equality_dot_syntax(self, db: Database) -> None:
        indexes.index_hparams(db, "hq1", {"lr": 0.001})
        indexes.index_hparams(db, "hq2", {"lr": 0.01})
        indexes.index_run(db, "hq1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hq2", is_archived=False, created_at=2.0)

        result = _plan(db, "run.hparams.lr == 0.001")
        assert result.candidates == ["hq1"]
        assert result.exact is True

    def test_hparam_equality_bracket_syntax(self, db: Database) -> None:
        indexes.index_hparams(db, "hqb1", {"lr": 0.001})
        indexes.index_run(db, "hqb1", is_archived=False, created_at=1.0)

        result = _plan(db, 'run["hparams"]["lr"] == 0.001')
        assert result.candidates == ["hqb1"]
        assert result.exact is True

    def test_hparam_comparison_lt(self, db: Database) -> None:
        indexes.index_hparams(db, "hlt1", {"lr": 0.001})
        indexes.index_hparams(db, "hlt2", {"lr": 0.01})
        indexes.index_hparams(db, "hlt3", {"lr": 0.1})
        indexes.index_run(db, "hlt1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hlt2", is_archived=False, created_at=2.0)
        indexes.index_run(db, "hlt3", is_archived=False, created_at=3.0)

        result = _plan(db, "run.hparams.lr < 0.01")
        assert result.candidates == ["hlt1"]
        assert result.exact is True

    def test_hparam_comparison_gt(self, db: Database) -> None:
        indexes.index_hparams(db, "hgt1", {"batch_size": 16})
        indexes.index_hparams(db, "hgt2", {"batch_size": 32})
        indexes.index_hparams(db, "hgt3", {"batch_size": 64})
        indexes.index_run(db, "hgt1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hgt2", is_archived=False, created_at=2.0)
        indexes.index_run(db, "hgt3", is_archived=False, created_at=3.0)

        result = _plan(db, "run.hparams.batch_size > 32")
        assert result.candidates == ["hgt3"]
        assert result.exact is True

    def test_hparam_string_equality(self, db: Database) -> None:
        indexes.index_hparams(db, "hstr1", {"optimizer": "adam"})
        indexes.index_hparams(db, "hstr2", {"optimizer": "sgd"})
        indexes.index_run(db, "hstr1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hstr2", is_archived=False, created_at=2.0)

        result = _plan(db, 'run.hparams.optimizer == "adam"')
        assert result.candidates == ["hstr1"]
        assert result.exact is True

    def test_hparam_negation_returns_superset(self, db: Database) -> None:
        """!= is not index-backed; AND with archived returns superset."""
        indexes.index_hparams(db, "hne1", {"lr": 0.01})
        indexes.index_run(db, "hne1", is_archived=False, created_at=1.0)

        result = _plan(db, "run.hparams.lr != 0.01")
        assert result.candidates is not None
        assert result.exact is False

    def test_hparam_gte(self, db: Database) -> None:
        indexes.index_hparams(db, "hge1", {"lr": 0.001})
        indexes.index_hparams(db, "hge2", {"lr": 0.01})
        indexes.index_hparams(db, "hge3", {"lr": 0.1})
        indexes.index_run(db, "hge1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hge2", is_archived=False, created_at=2.0)
        indexes.index_run(db, "hge3", is_archived=False, created_at=3.0)

        result = _plan(db, "run.hparams.lr >= 0.01")
        assert result.candidates is not None
        assert sorted(result.candidates) == ["hge2", "hge3"]
        assert result.exact is True

    def test_hparam_lte(self, db: Database) -> None:
        indexes.index_hparams(db, "hle1", {"lr": 0.001})
        indexes.index_hparams(db, "hle2", {"lr": 0.01})
        indexes.index_hparams(db, "hle3", {"lr": 0.1})
        indexes.index_run(db, "hle1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "hle2", is_archived=False, created_at=2.0)
        indexes.index_run(db, "hle3", is_archived=False, created_at=3.0)

        result = _plan(db, "run.hparams.lr <= 0.01")
        assert result.candidates is not None
        assert sorted(result.candidates) == ["hle1", "hle2"]
        assert result.exact is True

    def test_hparam_and_experiment_intersection(self, db: Database) -> None:
        """AND of experiment and hparam returns intersection."""
        indexes.index_run(db, "pri1", is_archived=False, created_at=1.0, experiment_name="exp-x")
        indexes.index_hparams(db, "pri1", {"lr": 0.001})

        result = _plan(db, 'run.experiment == "exp-x" and run.hparams.lr == 0.001')
        assert result.candidates == ["pri1"]
        assert result.exact is True


class TestPlanQueryMultiPredicate:
    """Test AND (intersection) and OR (union) of multiple index-backed predicates."""

    # ---- OR (union) ----

    def test_or_of_run_hashes(self, db: Database) -> None:
        indexes.index_run(db, "0cf28fd9b5bf2002", is_archived=False, created_at=1.0)
        indexes.index_run(db, "397a494499063719", is_archived=False, created_at=2.0)
        result = _plan(db, 'run.hash == "0cf28fd9b5bf2002" or run.hash == "397a494499063719"')
        assert result.candidates is not None
        assert set(result.candidates) == {"0cf28fd9b5bf2002", "397a494499063719"}
        assert result.exact is True

    def test_or_of_experiments(self, db: Database) -> None:
        indexes.index_run(db, "or_e1", is_archived=False, created_at=1.0, experiment_name="exp-a")
        indexes.index_run(db, "or_e2", is_archived=False, created_at=2.0, experiment_name="exp-b")
        indexes.index_run(db, "or_e3", is_archived=False, created_at=3.0, experiment_name="exp-c")

        result = _plan(db, 'run.experiment == "exp-a" or run.experiment == "exp-b"')
        assert result.candidates is not None
        assert set(result.candidates) == {"or_e1", "or_e2"}
        assert result.exact is True

    def test_or_of_tags(self, db: Database) -> None:
        indexes.index_run(db, "or_t1", is_archived=False, created_at=1.0)
        indexes.index_run(db, "or_t2", is_archived=False, created_at=2.0)
        indexes.add_tag_index(db, "or_t1", "alpha")
        indexes.add_tag_index(db, "or_t2", "beta")

        result = _plan(db, '"alpha" in run.tags or "beta" in run.tags')
        assert result.candidates is not None
        assert set(result.candidates) == {"or_t1", "or_t2"}
        assert result.exact is True

    def test_or_deduplicates(self, db: Database) -> None:
        indexes.index_run(db, "or_d1", is_archived=False, created_at=1.0, experiment_name="same")
        indexes.index_run(db, "or_d2", is_archived=False, created_at=2.0, experiment_name="same")

        result = _plan(db, 'run.experiment == "same" or run.experiment == "same"')
        assert result.candidates is not None
        assert sorted(result.candidates) == ["or_d1", "or_d2"]
        assert result.exact is True

    # ---- AND (intersection) ----

    def test_and_experiment_and_tag(self, db: Database) -> None:
        indexes.index_run(db, "and_et1", is_archived=False, created_at=1.0, experiment_name="exp")
        indexes.index_run(db, "and_et2", is_archived=False, created_at=2.0, experiment_name="exp")
        indexes.index_run(db, "and_et3", is_archived=False, created_at=3.0, experiment_name="other")
        indexes.add_tag_index(db, "and_et1", "t")
        indexes.add_tag_index(db, "and_et3", "t")

        result = _plan(db, 'run.experiment == "exp" and "t" in run.tags')
        assert result.candidates is not None
        assert set(result.candidates) == {"and_et1"}
        assert result.exact is True

    def test_and_hash_and_experiment_match(self, db: Database) -> None:
        indexes.index_run(db, "and_he1", is_archived=False, created_at=1.0, experiment_name="e1")

        result = _plan(db, 'run.hash == "and_he1" and run.experiment == "e1"')
        assert result.candidates is not None
        assert set(result.candidates) == {"and_he1"}
        assert result.exact is True

    def test_and_hash_and_experiment_disjoint(self, db: Database) -> None:
        indexes.index_run(db, "and_he2", is_archived=False, created_at=1.0, experiment_name="e2")

        result = _plan(db, 'run.hash == "nonexistent" and run.experiment == "e2"')
        assert result.candidates == []
        assert result.exact is True

    def test_and_active_and_experiment(self, db: Database) -> None:
        indexes.index_run(db, "and_ae1", is_archived=False, active=True, created_at=1.0, experiment_name="x")
        indexes.index_run(db, "and_ae2", is_archived=False, active=False, created_at=2.0, experiment_name="x")

        result = _plan(db, 'run.active == True and run.experiment == "x"')
        assert result.candidates is not None
        assert set(result.candidates) == {"and_ae1"}
        assert result.exact is True

    def test_empty_intersection(self, db: Database) -> None:
        """AND of two disjoint hash conditions returns empty list (not None)."""
        result = _plan(db, 'run.hash == "aaa" and run.hash == "bbb"')
        assert result.candidates == []
        assert result.exact is True

    # ---- Unindexed in OR ----

    def test_unindexed_in_or_returns_none(self, db: Database) -> None:
        """Pure OR with unindexed branch returns None (cannot form superset for OR)."""
        result = _plan(db, 'run.is_archived == False or run.name == "x"')
        assert result.candidates is None

    # ---- AND with indexed + unindexed → superset ----

    def test_and_indexed_and_unindexed_returns_superset(self, db: Database) -> None:
        """AND of indexed and unindexed returns superset with exact=False."""
        indexes.index_run(db, "unand1", is_archived=False, created_at=1.0)
        result = _plan(db, 'run.name == "x" and run.is_archived == False')
        assert result.candidates is not None
        assert "unand1" in result.candidates
        assert result.exact is False

    # ---- Mixed OR inside AND ----

    def test_and_with_nested_or(self, db: Database) -> None:
        """(run.hash == "a" or run.hash == "b") and run.is_archived == False."""
        indexes.index_run(db, "nest_a", is_archived=False, created_at=1.0)
        indexes.index_run(db, "nest_b", is_archived=False, created_at=2.0)
        indexes.index_run(db, "nest_c", is_archived=False, created_at=3.0)

        result = _plan(
            db,
            '(run.hash == "nest_a" or run.hash == "nest_b") and run.is_archived == False',
        )
        assert result.candidates is not None
        assert set(result.candidates) == {"nest_a", "nest_b"}
        assert result.exact is True


class TestPlanQueryMetricName:
    """Test Tier 3 metric trace-name index integration in the query planner."""

    def test_metric_name_eq_returns_candidates(self, db: Database) -> None:
        runs.create_run(db, "mn1")
        runs.set_context(db, "mn1", 0, {})
        runs.set_trace_info(db, "mn1", 0, "loss", dtype="float", last=0.5)

        runs.create_run(db, "mn2")
        runs.set_context(db, "mn2", 0, {})
        runs.set_trace_info(db, "mn2", 0, "acc", dtype="float", last=0.9)

        result = _plan(db, 'metric.name == "loss"')
        assert result.candidates is not None
        assert "mn1" in result.candidates
        assert "mn2" not in result.candidates
        assert result.trace_names == frozenset({"loss"})
        assert result.exact is True

    def test_metric_name_eq_and_experiment(self, db: Database) -> None:
        """AND of metric.name and experiment returns intersection."""
        runs.create_run(db, "mne1", experiment_id=None)
        runs.set_context(db, "mne1", 0, {})
        runs.set_trace_info(db, "mne1", 0, "loss", dtype="float", last=0.5)
        indexes.index_run(db, "mne1", is_archived=False, created_at=1.0, experiment_name="exp-a")

        runs.create_run(db, "mne2", experiment_id=None)
        runs.set_context(db, "mne2", 0, {})
        runs.set_trace_info(db, "mne2", 0, "loss", dtype="float", last=0.3)
        indexes.index_run(db, "mne2", is_archived=False, created_at=2.0, experiment_name="exp-b")

        result = _plan(db, 'metric.name == "loss" and run.experiment == "exp-a"')
        assert result.candidates is not None
        assert set(result.candidates) == {"mne1"}
        assert result.trace_names == frozenset({"loss"})
        assert result.exact is True

    def test_metric_name_or(self, db: Database) -> None:
        """OR of two metric names unions run hashes and trace names."""
        runs.create_run(db, "mno1")
        runs.set_context(db, "mno1", 0, {})
        runs.set_trace_info(db, "mno1", 0, "loss", dtype="float")

        runs.create_run(db, "mno2")
        runs.set_context(db, "mno2", 0, {})
        runs.set_trace_info(db, "mno2", 0, "acc", dtype="float")

        result = _plan(db, 'metric.name == "loss" or metric.name == "acc"')
        assert result.candidates is not None
        assert set(result.candidates) == {"mno1", "mno2"}
        assert result.trace_names == frozenset({"loss", "acc"})
        assert result.exact is True

    def test_metric_name_startswith_has_no_trace_names(self, db: Database) -> None:
        """metric.name.startswith(...) is not indexed — trace_names is None.

        The planner may still return candidates from run-level indexes
        (e.g. the default ``run.active`` predicate), but the streaming
        endpoint will fall back to the lazy path via
        ``query_has_unindexed_sequence_predicate``.
        """
        runs.create_run(db, "mns1")
        runs.set_context(db, "mns1", 0, {})
        runs.set_trace_info(db, "mns1", 0, "loss", dtype="float")

        result = _plan(db, 'metric.name.startswith("lo")')
        assert result.trace_names is None

    def test_metric_context_has_no_trace_names(self, db: Database) -> None:
        """metric.context references are not indexed — trace_names is None.

        Same logic as startswith: planner may return run-level candidates
        but ``query_has_unindexed_sequence_predicate`` forces lazy path.
        """
        runs.create_run(db, "mnc1")
        runs.set_context(db, "mnc1", 0, {"subset": "train"})
        runs.set_trace_info(db, "mnc1", 0, "loss", dtype="float")

        result = _plan(db, 'metric.context.subset == "train"')
        assert result.trace_names is None

    def test_run_only_query_has_no_trace_names(self, db: Database) -> None:
        """Queries without metric predicates have trace_names=None."""
        indexes.index_run(db, "mnn1", is_archived=False, created_at=1.0)
        result = _plan(db, "")
        assert result.trace_names is None
