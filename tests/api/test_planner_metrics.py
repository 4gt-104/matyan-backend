"""Tests for planner path Prometheus metrics (fast vs lazy path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import REGISTRY

from matyan_backend.metrics import PLANNER_PATH_TOTAL

if TYPE_CHECKING:
    from starlette.testclient import TestClient


def _get_planner_path_samples() -> list:
    """Collect counter value samples (matyan_planner_path_total) from the default registry."""
    samples: list = []
    for metric in REGISTRY.collect():
        if metric.name == "matyan_planner_path":
            samples.extend(s for s in metric.samples if s.name == "matyan_planner_path_total")
            break
    return samples


class TestPlannerPathCounterRegistered:
    def test_planner_path_total_in_registry(self) -> None:
        names = {m.name for m in REGISTRY.collect()}
        assert "matyan_planner_path" in names

    def test_counter_accepts_expected_labels(self) -> None:
        PLANNER_PATH_TOTAL.labels(path="fast", endpoint="metric_search", reason="").inc()
        PLANNER_PATH_TOTAL.labels(path="lazy", endpoint="metric_search", reason="no_candidates").inc()
        PLANNER_PATH_TOTAL.labels(path="fast", endpoint="run_search", reason="exact").inc()
        samples = _get_planner_path_samples()
        assert len(samples) >= 3
        paths = {(s.labels.get("path"), s.labels.get("endpoint"), s.labels.get("reason")) for s in samples}
        assert ("fast", "metric_search", "") in paths
        assert ("lazy", "metric_search", "no_candidates") in paths
        assert ("fast", "run_search", "exact") in paths


class TestPlannerPathMetricSearchIncrement:
    """Assert metric search API increments planner path counter (integration)."""

    def test_metric_search_emits_planner_path_metric(self, client: TestClient) -> None:
        resp = client.get("/api/v1/rest/runs/search/metric/", params={"q": ""})
        assert resp.status_code == 200
        samples = _get_planner_path_samples()
        metric_search_samples = [s for s in samples if s.labels.get("endpoint") == "metric_search"]
        assert len(metric_search_samples) >= 1
        total = sum(int(float(s.value)) for s in metric_search_samples)
        assert total >= 1
