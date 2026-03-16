"""Prometheus metrics definitions and helpers for the API backend.

All metric names are prefixed with ``matyan_`` so they are easy to filter
in Prometheus/Grafana.
"""

from __future__ import annotations

import re

from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "matyan_http_requests_total",
    "Total HTTP requests",
    ["method", "path_template", "status_class"],
)

HTTP_REQUEST_DURATION = Histogram(
    "matyan_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path_template", "status_class"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PLANNER_PATH_TOTAL = Counter(
    "matyan_planner_path_total",
    "Number of query executions by planner path (fast = index, lazy = full scan)",
    ["path", "endpoint", "reason"],
)

_HEX_ID_RE = re.compile(r"(?<=/)[0-9a-f]{8,40}(?=/|$)")
_UUID_RE = re.compile(
    r"(?<=/)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)",
    re.IGNORECASE,
)


def normalize_path(path: str) -> str:
    """Replace dynamic IDs in a URL path with ``{id}`` to limit label cardinality."""
    path = _UUID_RE.sub("{id}", path)
    return _HEX_ID_RE.sub("{id}", path)
