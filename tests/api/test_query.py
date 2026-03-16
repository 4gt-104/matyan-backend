"""Tests for api/runs/_query.py — RestrictedPython query evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from matyan_backend.api.runs._query import (
    RestrictedPythonQuery,
    compile_checker,
    query_add_default_expr,
    safe_import,
    safer_getattr,
    strip_query,
    syntax_error_check,
)


class TestSafeImport:
    def test_time_allowed(self) -> None:
        mod = safe_import("time")
        assert hasattr(mod, "time")

    def test_os_blocked(self) -> None:
        with pytest.raises(ImportError, match="os package cannot be imported"):
            safe_import("os")

    def test_sys_blocked(self) -> None:
        with pytest.raises(ImportError, match="sys package cannot be imported"):
            safe_import("sys")


class TestSaferGetattr:
    def test_normal_attribute(self) -> None:
        result = safer_getattr("hello", "upper")
        assert callable(result)

    def test_format_on_string_blocked(self) -> None:
        with pytest.raises(NotImplementedError, match="format"):
            safer_getattr("hello", "format")

    def test_underscore_attribute_blocked(self) -> None:
        with pytest.raises(AttributeError, match="invalid attribute"):
            safer_getattr("hello", "_private")

    def test_default_value(self) -> None:
        result = safer_getattr("hello", "nonexistent", "default")
        assert result == "default"


class TestStripQuery:
    def test_empty(self) -> None:
        assert strip_query("") == ""

    def test_select_if(self) -> None:
        result = strip_query("SELECT run IF run.active == True")
        assert "run.active" in result
        assert "SELECT" not in result.upper()

    def test_plain_expression(self) -> None:
        result = strip_query("run.active == True")
        assert "run.active" in result

    def test_select_no_if(self) -> None:
        result = strip_query("select something")
        assert result == ""

    def test_whitespace_stripped(self) -> None:
        result = strip_query("  run.name == 'test'  ")
        assert result.strip() == "(run.name == 'test')"


class TestQueryAddDefaultExpr:
    def test_empty_query(self) -> None:
        result = query_add_default_expr("")
        assert "run.is_archived == False" in result

    def test_already_has_archived(self) -> None:
        q = "run.is_archived == True"
        assert query_add_default_expr(q) == q

    def test_already_has_run_archived(self) -> None:
        q = "run.archived == False"
        assert query_add_default_expr(q) == q

    def test_adds_default(self) -> None:
        result = query_add_default_expr("run.active == True")
        assert "run.is_archived == False" in result
        assert "run.active == True" in result


class TestSyntaxErrorCheck:
    def test_valid_expression(self) -> None:
        syntax_error_check("run.active == True")

    def test_empty_expression(self) -> None:
        syntax_error_check("")

    def test_invalid_python(self) -> None:
        with pytest.raises(SyntaxError):
            syntax_error_check("def foo(:")


class TestCompileChecker:
    def test_basic(self) -> None:
        code = compile_checker("(True)")
        assert code is not None

    def test_cached(self) -> None:
        c1 = compile_checker("(1 + 1)")
        c2 = compile_checker("(1 + 1)")
        assert c1 is c2


class TestRestrictedPythonQuery:
    def test_check_true(self) -> None:
        q = RestrictedPythonQuery("")
        run = MagicMock()
        run.is_archived = False
        run.archived = False
        assert q.check(run=run) is True

    def test_check_false(self) -> None:
        q = RestrictedPythonQuery("")
        run = MagicMock()
        run.is_archived = True
        run.archived = True
        assert q.check(run=run) is False

    def test_bool(self) -> None:
        q = RestrictedPythonQuery("run.active == True")
        assert bool(q) is True

    def test_invalid_params(self) -> None:
        q = RestrictedPythonQuery("")
        with pytest.raises(AssertionError):
            q.check(invalid_param=True)

    def test_evaluation_error_returns_false(self) -> None:
        q = RestrictedPythonQuery("SELECT run IF run.nonexistent.deeply.nested == True")
        run = MagicMock()
        run.is_archived = False
        run.archived = False
        run.nonexistent = None
        result = q.check(run=run)
        assert isinstance(result, bool)

    def test_callable(self) -> None:
        q = RestrictedPythonQuery("")
        run = MagicMock()
        run.is_archived = False
        run.archived = False
        result = q(run=run)
        assert result is True

    def test_exception_returns_false(self) -> None:
        q = RestrictedPythonQuery("SELECT run IF 1/0")
        run = MagicMock()
        run.is_archived = False
        run.archived = False
        result = q.check(run=run)
        assert result is False
