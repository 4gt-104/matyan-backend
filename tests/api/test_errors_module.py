"""Unit tests for api.errors (ErrorEnvelope, status mapping, detail normalization)."""

from __future__ import annotations

from matyan_backend.api.errors import ErrorEnvelope


class TestErrorEnvelopeFromHttpException:
    def test_404_no_detail(self) -> None:
        env = ErrorEnvelope.from_http_exception(404, None)
        assert env.error.code == "not_found"
        assert env.error.message == "Not found"
        assert env.detail == "Not found"

    def test_400_with_detail(self) -> None:
        env = ErrorEnvelope.from_http_exception(400, "Duplicate name")
        assert env.error.code == "bad_request"
        assert env.error.message == "Duplicate name"
        assert env.detail == "Duplicate name"

    def test_422_list_detail(self) -> None:
        env = ErrorEnvelope.from_http_exception(
            422,
            [{"type": "missing", "loc": ["body", "name"], "msg": "Field required"}],
        )
        assert env.error.code == "validation_error"
        assert "Field required" in env.error.message
        assert env.detail == env.error.message

    def test_500_no_detail(self) -> None:
        env = ErrorEnvelope.from_http_exception(500, None)
        assert env.error.code == "internal_error"
        assert env.error.message == "Internal server error"
        assert env.detail == "Internal server error"

    def test_unknown_status_maps_to_error_code(self) -> None:
        env = ErrorEnvelope.from_http_exception(418, "I'm a teapot")
        assert env.error.code == "error"
        assert env.error.message == "I'm a teapot"
