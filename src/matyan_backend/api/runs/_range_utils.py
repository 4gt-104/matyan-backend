"""Utilities for parsing range query parameters (``record_range``, ``index_range``)."""

from __future__ import annotations

from typing import NamedTuple


class IndexRange(NamedTuple):
    start: int | None
    stop: int | None


def parse_range(s: str) -> IndexRange:
    """Parse ``'start:stop'`` into an :class:`IndexRange`.

    Empty or missing parts become ``None``.
    """
    if not s or ":" not in s:
        return IndexRange(None, None)
    parts = s.split(":", 1)
    start = int(parts[0]) if parts[0].strip() else None
    stop = int(parts[1]) if parts[1].strip() else None
    return IndexRange(start, stop)
