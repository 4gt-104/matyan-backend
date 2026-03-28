"""Tests for cli.py — Click CLI commands.

CLI commands use local imports, so patches target the module where the import
resolves at runtime rather than cli module-level attributes.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from matyan_backend.cli import main

# ruff: noqa: ARG002


class TestStartCommand:
    @patch("uvicorn.run")
    def test_start_default(self, mock_run: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["start"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            "matyan_backend.app:app",
            host="0.0.0.0",
            port=53800,
            workers=1,
            log_level="info",
        )

    @patch("uvicorn.run")
    def test_start_custom_port(self, mock_run: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["start", "--port", "8080", "--host", "127.0.0.1"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            "matyan_backend.app:app",
            host="127.0.0.1",
            port=8080,
            workers=1,
            log_level="info",
        )


class TestReindexCommand:
    @patch("matyan_backend.storage.indexes.rebuild_indexes", return_value=(5, 0))
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_reindex(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_rebuild: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["reindex"])
        assert result.exit_code == 0
        assert "5 run(s)" in result.output
        mock_init.assert_called_once()
        mock_dirs.assert_called_once()
        mock_rebuild.assert_called_once()


class TestIngestWorkerCommand:
    @patch("prometheus_client.start_http_server")
    @patch("asyncio.run")
    @patch("matyan_backend.workers.ingestion.IngestionWorker")
    def test_ingest_worker(
        self, mock_worker_cls: MagicMock, mock_asyncio_run: MagicMock, mock_prom: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ingest-worker"])
        assert result.exit_code == 0
        mock_worker_cls.assert_called_once()
        mock_asyncio_run.assert_called_once()
        mock_prom.assert_called_once()


class TestControlWorkerCommand:
    @patch("prometheus_client.start_http_server")
    @patch("asyncio.run")
    @patch("matyan_backend.workers.control.ControlWorker")
    def test_control_worker(
        self, mock_worker_cls: MagicMock, mock_asyncio_run: MagicMock, mock_prom: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["control-worker"])
        assert result.exit_code == 0
        mock_worker_cls.assert_called_once()
        mock_asyncio_run.assert_called_once()
        mock_prom.assert_called_once()


class TestFinishStaleCommand:
    @patch("matyan_backend.storage.runs.update_run_meta")
    @patch("matyan_backend.storage.runs.get_run_meta", return_value={"created_at": 0.0})
    @patch("matyan_backend.storage.indexes.lookup_by_active", return_value=["stale-1", "stale-2"])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_finish_stale_default(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_lookup: MagicMock,
        mock_meta: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["finish-stale"])
        assert result.exit_code == 0
        assert "2 stale run(s)" in result.output
        assert mock_update.call_count == 2

    @patch("matyan_backend.storage.runs.update_run_meta")
    @patch("matyan_backend.storage.runs.get_run_meta")
    @patch("matyan_backend.storage.indexes.lookup_by_active", return_value=["fresh-1"])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_finish_stale_skips_fresh_runs(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_lookup: MagicMock,
        mock_meta: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        mock_meta.return_value = {"created_at": time.time() - 3600}
        runner = CliRunner()
        result = runner.invoke(main, ["finish-stale", "--timeout-hours", "24"])
        assert result.exit_code == 0
        assert "0 stale run(s)" in result.output
        mock_update.assert_not_called()

    @patch("matyan_backend.storage.indexes.lookup_by_active", return_value=[])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_finish_stale_no_active_runs(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_lookup: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["finish-stale"])
        assert result.exit_code == 0
        assert "0 stale run(s)" in result.output


class TestCleanupOrphanBlobs:
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("r1", 100.0), ("r2", 200.0)])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_dry_run(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-orphan-blobs", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run]" in result.output
        assert "r1/" in result.output
        assert "r2/" in result.output

    @patch("matyan_backend.storage.blob.delete_blob_prefix", return_value=3)
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("r1", 100.0)])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_actual_run(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-orphan-blobs"])
        assert result.exit_code == 0
        assert "3 blob(s)" in result.output
        mock_delete.assert_called_once()

    @patch("matyan_backend.storage.blob.delete_blob_prefix", return_value=1)
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("r1", 1.0), ("r2", 2.0), ("r3", 3.0)])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_limit(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-orphan-blobs", "--limit", "2"])
        assert result.exit_code == 0
        assert mock_delete.call_count == 2

    @patch("matyan_backend.jobs.lock.try_acquire", return_value=False)
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_lock_already_held(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-orphan-blobs", "--lock-ttl-seconds", "60"])
        assert result.exit_code != 0
        assert "Could not acquire lock" in result.output

    @patch("matyan_backend.jobs.lock.release")
    @patch("matyan_backend.jobs.lock.try_acquire", return_value=True)
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_lock_acquired_and_released(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-orphan-blobs", "--lock-ttl-seconds", "60"])
        assert result.exit_code == 0
        mock_acquire.assert_called_once_with(mock_init.return_value, "cleanup_orphan_blobs", 60)
        mock_release.assert_called_once_with(mock_init.return_value, "cleanup_orphan_blobs")


class TestCleanupTombstones:
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("r1", 0.0), ("r2", time.time())])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_dry_run(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-tombstones", "--dry-run", "--older-than-hours", "1"])
        assert result.exit_code == 0
        assert "[dry-run]" in result.output
        assert "r1" in result.output

    @patch("matyan_backend.storage.indexes.clear_run_tombstone")
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("old1", 0.0), ("old2", 1.0)])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_clears_old_tombstones(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_clear: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-tombstones", "--older-than-hours", "1"])
        assert result.exit_code == 0
        assert mock_clear.call_count == 2
        assert "Cleared 2 tombstone(s)" in result.output

    @patch("matyan_backend.storage.indexes.clear_run_tombstone")
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[("recent", time.time())])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_skips_recent_tombstones(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_clear: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-tombstones", "--older-than-hours", "1"])
        assert result.exit_code == 0
        mock_clear.assert_not_called()
        assert "Cleared 0 tombstone(s)" in result.output

    @patch("matyan_backend.jobs.lock.try_acquire", return_value=False)
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_lock_already_held(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_acquire: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-tombstones", "--lock-ttl-seconds", "60"])
        assert result.exit_code != 0
        assert "Could not acquire lock" in result.output

    @patch("matyan_backend.jobs.lock.release")
    @patch("matyan_backend.jobs.lock.try_acquire", return_value=True)
    @patch("matyan_backend.storage.indexes.list_tombstones", return_value=[])
    @patch("matyan_backend.storage.fdb_client.ensure_directories")
    @patch("matyan_backend.storage.fdb_client.init_fdb")
    def test_lock_acquired_and_released(
        self,
        mock_init: MagicMock,
        mock_dirs: MagicMock,
        mock_tombstones: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-tombstones", "--lock-ttl-seconds", "120"])
        assert result.exit_code == 0
        mock_acquire.assert_called_once_with(mock_init.return_value, "cleanup_tombstones", 120)
        mock_release.assert_called_once_with(mock_init.return_value, "cleanup_tombstones")
