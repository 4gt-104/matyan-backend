"""Extended tests for api/runs/_planner.py — hparam range operators and bracket syntax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.api.runs._planner import PlanResult, _parse_literal, _plan_hparam_val, plan_query
from matyan_backend.api.runs._query import prepare_query
from matyan_backend.storage import runs

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


def _plan(db: Database, raw: str, tz_offset: int = 0) -> PlanResult:
    """Shorthand: prepare_query + plan_query."""
    return plan_query(db, prepare_query(raw, tz_offset))


class TestParseLiteral:
    def test_true(self) -> None:
        assert _parse_literal("True") is True

    def test_false(self) -> None:
        assert _parse_literal("False") is False

    def test_double_quoted_string(self) -> None:
        assert _parse_literal('"hello"') == "hello"

    def test_single_quoted_string(self) -> None:
        assert _parse_literal("'world'") == "world"

    def test_int(self) -> None:
        assert _parse_literal("42") == 42

    def test_float(self) -> None:
        assert _parse_literal("3.14") == 3.14

    def test_unparseable(self) -> None:
        assert _parse_literal("unparseable") == "unparseable"


class TestHparamPlannerOperators:
    def test_not_equal_falls_back(self, db: Database) -> None:
        result = _plan(db, "run.hparams.lr != 0.01")
        assert result.candidates is not None
        assert result.exact is False

    def test_less_than(self, db: Database) -> None:
        runs.create_run(db, "pllt1")
        runs.set_run_attrs(db, "pllt1", ("hparams",), {"lr": 0.01})
        runs.create_run(db, "pllt2")
        runs.set_run_attrs(db, "pllt2", ("hparams",), {"lr": 0.1})

        result = _plan(db, "run.hparams.lr < 0.1")
        assert result.candidates is not None
        assert "pllt1" in result.candidates
        assert "pllt2" not in result.candidates

    def test_less_than_equal(self, db: Database) -> None:
        runs.create_run(db, "plle1")
        runs.set_run_attrs(db, "plle1", ("hparams",), {"lr": 0.01})
        runs.create_run(db, "plle2")
        runs.set_run_attrs(db, "plle2", ("hparams",), {"lr": 0.1})

        result = _plan(db, "run.hparams.lr <= 0.1")
        assert result.candidates is not None
        assert "plle1" in result.candidates
        assert "plle2" in result.candidates

    def test_greater_than(self, db: Database) -> None:
        runs.create_run(db, "plgt1")
        runs.set_run_attrs(db, "plgt1", ("hparams",), {"lr": 0.01})
        runs.create_run(db, "plgt2")
        runs.set_run_attrs(db, "plgt2", ("hparams",), {"lr": 0.1})

        result = _plan(db, "run.hparams.lr > 0.01")
        assert result.candidates is not None
        assert "plgt2" in result.candidates
        assert "plgt1" not in result.candidates

    def test_greater_than_equal(self, db: Database) -> None:
        runs.create_run(db, "plge1")
        runs.set_run_attrs(db, "plge1", ("hparams",), {"lr": 0.01})

        result = _plan(db, "run.hparams.lr >= 0.01")
        assert result.candidates is not None
        assert "plge1" in result.candidates

    def test_bracket_syntax(self, db: Database) -> None:
        runs.create_run(db, "plbr1")
        runs.set_run_attrs(db, "plbr1", ("hparams",), {"lr": 0.01})

        result = _plan(db, 'run["hparams"]["lr"] == 0.01')
        assert result.candidates is not None
        assert "plbr1" in result.candidates

    def test_unknown_op_returns_none(self, db: Database) -> None:
        result = _plan_hparam_val(db, "lr", "~=", 0.01)
        assert result is None
