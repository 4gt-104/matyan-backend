"""Main API router: mounts all REST sub-routers under the ``/rest`` prefix."""

from fastapi import APIRouter

from .dashboard_apps.views import dashboard_apps_router
from .dashboards.views import dashboards_router
from .experiments import rest_router_experiments
from .projects import rest_router_projects
from .reports.views import reports_router
from .runs import rest_router_runs
from .tags import tags_router
from .version import version_router

main_router = APIRouter(prefix="/rest")
main_router.include_router(rest_router_runs)
main_router.include_router(tags_router)
main_router.include_router(rest_router_experiments)
main_router.include_router(rest_router_projects)
main_router.include_router(version_router)
main_router.include_router(dashboards_router, prefix="/dashboards", tags=["dashboards"])
main_router.include_router(dashboard_apps_router, prefix="/apps", tags=["apps"])
main_router.include_router(reports_router, prefix="/reports", tags=["reports"])
