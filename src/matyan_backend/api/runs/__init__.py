"""Run REST endpoints: CRUD, tags, notes, logs, artifacts, and streaming search.

Exposes :data:`rest_router_runs` (prefix ``/runs``). Non-streaming handlers
are in :mod:`._run`; streaming search is in :mod:`._streaming`.
"""

from ._run import rest_router_runs

__all__ = ["rest_router_runs"]
