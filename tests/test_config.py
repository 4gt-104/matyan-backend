"""Tests for config.py — Settings and production validation."""

from __future__ import annotations

import pytest

from matyan_backend.config import Settings, validate_production_settings


def _settings_production_overrides() -> dict[str, str]:
    """Minimal overrides so validate_production_settings passes for environment=production."""
    return {
        "blob_uri_secret": "prod-fernet-key-override",
        "s3_access_key": "prod-ak",
        "s3_secret_key": "prod-sk",
        "s3_endpoint": "https://s3.example.com",
        "kafka_bootstrap_servers": "kafka.example.com:9092",
        "fdb_cluster_file": "/etc/foundationdb/fdb.cluster",
    }


class TestValidateProductionSettings:
    """Production guard: when environment is production, dev defaults are rejected."""

    def test_development_environment_skips_validation(self) -> None:
        settings = Settings.model_construct(environment="development")
        validate_production_settings(settings)  # does not raise

    def test_production_with_defaults_raises_on_first_offender(self) -> None:
        settings = Settings.model_construct(environment="production")
        with pytest.raises(ValueError, match="BLOB_URI_SECRET"):
            validate_production_settings(settings)

    def test_production_with_all_overrides_passes(self) -> None:
        settings = Settings.model_construct(
            environment="production",
            **_settings_production_overrides(),  # ty:ignore[invalid-argument-type]
        )
        validate_production_settings(settings)  # does not raise

    def test_production_with_default_blob_uri_secret_raises(self) -> None:
        overrides = _settings_production_overrides()
        overrides["blob_uri_secret"] = "Juw5-cLlemQI2jAWvceOUB3_CrVfBmI99YIzkpGUXR4="  # noqa: S105
        settings = Settings.model_construct(environment="production", **overrides)  # ty:ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="BLOB_URI_SECRET"):
            validate_production_settings(settings)

    def test_production_with_default_s3_endpoint_raises(self) -> None:
        overrides = _settings_production_overrides()
        overrides["s3_endpoint"] = "http://localhost:9000"
        settings = Settings.model_construct(environment="production", **overrides)  # ty:ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="S3_ENDPOINT"):
            validate_production_settings(settings)

    def test_production_with_azure_missing_creds_raises(self) -> None:
        overrides = _settings_production_overrides()
        overrides["blob_backend_type"] = "azure"
        overrides["azure_container"] = "matyan-artifacts"
        # No azure_conn_str or azure_account_url
        settings = Settings.model_construct(environment="production", **overrides)  # ty:ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="AZURE_CONN_STR or AZURE_ACCOUNT_URL"):
            validate_production_settings(settings)

    def test_production_with_azure_valid_creds_passes(self) -> None:
        overrides = _settings_production_overrides()
        overrides["blob_backend_type"] = "azure"
        overrides["azure_container"] = "matyan-artifacts"
        overrides["azure_conn_str"] = "DefaultEndpointsProtocol=...;"
        settings = Settings.model_construct(environment="production", **overrides)  # ty:ignore[invalid-argument-type]
        validate_production_settings(settings)  # does not raise
