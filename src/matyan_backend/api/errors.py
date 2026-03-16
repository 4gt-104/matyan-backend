"""Standard error envelope for all 4xx/5xx responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Map HTTP status codes to stable machine-readable codes.
STATUS_TO_CODE: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    500: "internal_error",
}

DEFAULT_MESSAGES: dict[int, str] = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    409: "Conflict",
    422: "Validation error",
    500: "Internal server error",
}


def _status_to_code(status_code: int) -> str:
    return STATUS_TO_CODE.get(status_code, "error")


def _detail_to_message(detail: Any, status_code: int) -> str:  # noqa: ANN401
    """Normalize HTTPException detail to a single message string."""
    if detail is None:
        return DEFAULT_MESSAGES.get(status_code, "Error")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                parts.append(item.get("msg", str(item)))
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else DEFAULT_MESSAGES.get(status_code, "Error")
    return str(detail)


class ErrorBody(BaseModel):
    """Machine-readable error code and human-readable message.

    :param code: Stable code (e.g. ``not_found``, ``validation_error``).
    :param message: Human-readable message for the client.
    """

    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """Standard envelope for all 4xx/5xx error responses.

    :param error: Nested ErrorBody with code and message.
    :param detail: Same as error.message for UI compatibility.
    """

    error: ErrorBody
    detail: str = Field(..., description="Same as error.message for UI compatibility")

    @classmethod
    def from_http_exception(cls, status_code: int, detail: Any) -> ErrorEnvelope:  # noqa: ANN401
        """Build an ErrorEnvelope from an HTTP status code and exception detail."""
        message = _detail_to_message(detail, status_code)
        code = _status_to_code(status_code)
        return cls(error=ErrorBody(code=code, message=message), detail=message)
