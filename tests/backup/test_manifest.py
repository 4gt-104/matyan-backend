"""Tests for backup manifest read/write/validate."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from matyan_api_models.backup import FORMAT_VERSION, BackupManifest

if TYPE_CHECKING:
    from pathlib import Path


class TestManifestRoundTrip:
    def test_write_and_read(self, tmp_path: Path) -> None:
        m = BackupManifest(
            run_count=2,
            run_hashes=["aaa", "bbb"],
            entity_counts={"experiment": 1, "tag": 2},
            blob_count=5,
            blob_bytes=1024,
            filters={"experiment": "baseline"},
        )
        m.write(tmp_path)
        loaded = BackupManifest.read(tmp_path)
        assert loaded.format_version == FORMAT_VERSION
        assert loaded.run_count == 2
        assert loaded.run_hashes == ["aaa", "bbb"]
        assert loaded.entity_counts == {"experiment": 1, "tag": 2}
        assert loaded.blob_count == 5
        assert loaded.blob_bytes == 1024
        assert loaded.filters == {"experiment": "baseline"}

    def test_read_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"No manifest\.json"):
            BackupManifest.read(tmp_path)

    def test_read_wrong_version(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text(json.dumps({"format_version": 999}))
        with pytest.raises(ValueError, match="Unsupported backup format version"):
            BackupManifest.read(tmp_path)


class TestManifestValidation:
    def test_validate_missing_runs_dir(self, tmp_path: Path) -> None:
        m = BackupManifest(run_hashes=["aaa"])
        m.write(tmp_path)
        errors = m.validate(tmp_path)
        assert any("Missing 'runs/' directory" in e for e in errors)

    def test_validate_missing_run_directory(self, tmp_path: Path) -> None:
        (tmp_path / "runs").mkdir()
        m = BackupManifest(run_hashes=["aaa"])
        m.write(tmp_path)
        errors = m.validate(tmp_path)
        assert any("Missing run directories" in e for e in errors)

    def test_validate_missing_required_files(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "aaa"
        run_dir.mkdir(parents=True)
        m = BackupManifest(run_hashes=["aaa"])
        m.write(tmp_path)
        errors = m.validate(tmp_path)
        assert any("missing run.json" in e for e in errors)

    def test_validate_success(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "aaa"
        run_dir.mkdir(parents=True)
        for fname in ("run.json", "attrs.json", "traces.json", "contexts.json"):
            (run_dir / fname).write_text("{}")
        m = BackupManifest(run_hashes=["aaa"])
        m.write(tmp_path)
        errors = m.validate(tmp_path)
        assert errors == []
