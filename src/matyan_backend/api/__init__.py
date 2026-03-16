"""REST API package for matyan-backend.

Exposes :data:`main_router`, the top-level FastAPI router mounted at ``/rest``.
Sub-routers cover runs, experiments, tags, projects, dashboards, apps, and reports.
"""

from ._main import main_router

__all__ = ["main_router"]
