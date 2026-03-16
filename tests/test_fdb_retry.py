"""Tests for FDB retry helpers and the transactional decorator retry behaviour."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
from fdb.impl import Database, FDBError, Transaction

from matyan_backend.fdb_types import (
    RETRYABLE_FDB_CODES,
    is_retryable_fdb_error,
    run_with_retry,
    transactional,
)

# ---------------------------------------------------------------------------
# is_retryable_fdb_error
# ---------------------------------------------------------------------------


class TestIsRetryableFdbError:
    @pytest.mark.parametrize("code", sorted(RETRYABLE_FDB_CODES))
    def test_retryable_codes(self, code: int) -> None:
        assert is_retryable_fdb_error(FDBError(code)) is True

    @pytest.mark.parametrize("code", [2000, 2101, 4000, 4100])
    def test_non_retryable_codes(self, code: int) -> None:
        assert is_retryable_fdb_error(FDBError(code)) is False

    def test_non_fdb_error(self) -> None:
        assert is_retryable_fdb_error(RuntimeError("boom")) is False

    def test_plain_exception(self) -> None:
        assert is_retryable_fdb_error(Exception("generic")) is False


# ---------------------------------------------------------------------------
# run_with_retry
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    def test_success_first_call(self) -> None:
        fn = MagicMock(return_value=42)
        result = run_with_retry(fn, max_attempts=3, initial_delay=0.001, max_delay=0.01)
        assert result == 42
        assert fn.call_count == 1

    @patch("matyan_backend.fdb_types.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[FDBError(1004), FDBError(1031), "ok"])
        result = run_with_retry(fn, max_attempts=5, initial_delay=0.001, max_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("matyan_backend.fdb_types.time.sleep")
    def test_exhausts_max_attempts(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=FDBError(1004))
        with pytest.raises(FDBError) as exc_info:
            run_with_retry(fn, max_attempts=3, initial_delay=0.001, max_delay=0.01)
        assert exc_info.value.code == 1004
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    def test_non_retryable_fdb_error_not_retried(self) -> None:
        fn = MagicMock(side_effect=FDBError(2101))
        with pytest.raises(FDBError) as exc_info:
            run_with_retry(fn, max_attempts=5, initial_delay=0.001, max_delay=0.01)
        assert exc_info.value.code == 2101
        assert fn.call_count == 1

    def test_non_fdb_exception_not_retried(self) -> None:
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            run_with_retry(fn, max_attempts=5, initial_delay=0.001, max_delay=0.01)
        assert fn.call_count == 1

    @patch("matyan_backend.fdb_types.time.sleep")
    def test_backoff_delay_capped(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[FDBError(1004)] * 9 + ["ok"])
        run_with_retry(fn, max_attempts=10, initial_delay=0.5, max_delay=1.0)
        for call in mock_sleep.call_args_list:
            assert call.args[0] <= 1.0 + 0.15

    def test_return_value_preserved(self) -> None:
        fn = MagicMock(return_value={"key": [1, 2, 3]})
        result = run_with_retry(fn, max_attempts=1, initial_delay=0.001, max_delay=0.01)
        assert result == {"key": [1, 2, 3]}

    @patch("matyan_backend.fdb_types.time.sleep")
    def test_on_retry_callback_called_on_each_retry(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[FDBError(1004), FDBError(1020), "ok"])
        callback = MagicMock()
        result = run_with_retry(fn, max_attempts=5, initial_delay=0.001, max_delay=0.01, on_retry=callback)
        assert result == "ok"
        assert callback.call_count == 2
        assert mock_sleep.call_count == 2

    def test_on_retry_not_called_on_success(self) -> None:
        fn = MagicMock(return_value="ok")
        callback = MagicMock()
        run_with_retry(fn, max_attempts=3, initial_delay=0.001, max_delay=0.01, on_retry=callback)
        assert callback.call_count == 0


# ---------------------------------------------------------------------------
# transactional decorator retry
# ---------------------------------------------------------------------------


def _passthrough_transactional(func):  # noqa: ANN001, ANN202
    """Stand-in for fdb.transactional that just calls the function directly."""

    @functools.wraps(func)
    def wrapper(db_or_tr, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN001, ANN401
        if isinstance(db_or_tr, Database):
            tr = db_or_tr.create_transaction()
            return func(tr, *args, **kwargs)
        return func(db_or_tr, *args, **kwargs)

    return wrapper


@pytest.fixture
def _mock_transactional() -> Generator[None]:
    """Patch fdb.transactional with passthrough. Use via @pytest.mark.usefixtures."""
    with patch(
        "matyan_backend.fdb_types._fdb.transactional",
        side_effect=_passthrough_transactional,
    ):
        yield


class TestTransactionalRetry:
    @pytest.mark.usefixtures("_mock_transactional")
    @patch("matyan_backend.fdb_types.time.sleep")
    def test_retries_when_database_passed(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """When a Database is passed, retryable FDBError triggers retry."""
        mock_db = MagicMock(spec=Database)
        mock_db.create_transaction.return_value = MagicMock()

        call_count = 0

        @transactional
        def my_func(tr: Transaction) -> str:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FDBError(1004)
            return "done"

        result = my_func(mock_db)
        assert result == "done"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.usefixtures("_mock_transactional")
    def test_no_retry_when_transaction_passed(self) -> None:
        """When a Transaction is passed, no retry wrapping occurs."""
        mock_tr = MagicMock(spec=Transaction)
        call_count = 0

        @transactional
        def my_func(tr: Transaction) -> str:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FDBError(1004)
            return "done"

        with pytest.raises(FDBError):
            my_func(mock_tr)
        assert call_count == 1

    @pytest.mark.usefixtures("_mock_transactional")
    @patch("matyan_backend.fdb_types.time.sleep")
    def test_non_retryable_error_not_retried(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """Non-retryable FDBError is raised immediately, no retry."""
        mock_db = MagicMock(spec=Database)
        mock_db.create_transaction.return_value = MagicMock()

        @transactional
        def my_func(tr: Transaction) -> str:  # noqa: ARG001
            raise FDBError(2101)

        with pytest.raises(FDBError) as exc_info:
            my_func(mock_db)
        assert exc_info.value.code == 2101
        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# Retryable code coverage
# ---------------------------------------------------------------------------


class TestRetryableCodeSet:
    def test_expected_codes_present(self) -> None:
        expected = {1004, 1007, 1020, 1021, 1031, 1037, 1051, 1213}
        assert expected == RETRYABLE_FDB_CODES

    def test_is_frozenset(self) -> None:
        assert isinstance(RETRYABLE_FDB_CODES, frozenset)
