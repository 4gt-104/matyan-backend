"""Version endpoint: returns backend version and FDB API version."""

from __future__ import annotations

import importlib.metadata

from fastapi import APIRouter

from matyan_backend.config import SETTINGS

version_router = APIRouter(tags=["version"])


def _get_backend_version() -> str:
    """Return the installed matyan-backend package version."""
    try:
        return importlib.metadata.version("matyan-backend")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0.dev0"


@version_router.get("/version/")
async def get_version() -> dict:
    """Return backend version and FDB API version.

    :returns: JSON with ``version``, ``component``, and ``fdb_api_version``.
    """
    return {
        "version": _get_backend_version(),
        "component": "backend",
        "fdb_api_version": SETTINGS.fdb_api_version,
    }
