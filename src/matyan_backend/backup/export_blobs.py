"""Download S3 artifacts for backed-up runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from google.cloud import storage

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from pathlib import Path

    from types_boto3_s3 import S3Client


def _make_s3_client() -> S3Client:
    return boto3.client(
        "s3",
        endpoint_url=SETTINGS.s3_endpoint,
        aws_access_key_id=SETTINGS.s3_access_key,
        aws_secret_access_key=SETTINGS.s3_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name=SETTINGS.s3_region,
    )


def _make_gcs_client() -> storage.Client:
    return storage.Client()


def export_blobs_for_run(
    run_hash: str,
    backup_dir: Path,
    *,
    s3_client: S3Client | None = None,
    gcs_client: storage.Client | None = None,
) -> tuple[int, int]:
    """Download all objects under ``<run_hash>/`` into the backup.

    Returns ``(blob_count, total_bytes)``.
    """
    prefix = f"{run_hash}/"
    blobs_dir = backup_dir / "runs" / run_hash / "blobs"

    blob_count = 0
    total_bytes = 0

    if SETTINGS.blob_backend_type == "gcs":
        client = gcs_client or _make_gcs_client()
        bucket = client.bucket(SETTINGS.gcs_bucket)
        for blob in bucket.list_blobs(prefix=prefix):
            s3_key = blob.name
            relative = s3_key[len(prefix) :]
            if not relative:
                continue

            local_path = blobs_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            data = blob.download_as_bytes()
            local_path.write_bytes(data)

            blob_count += 1
            total_bytes += len(data)

    else:
        client = s3_client or _make_s3_client()
        bucket_name = SETTINGS.s3_bucket
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative = s3_key[len(prefix) :]
                if not relative:
                    continue

                local_path = blobs_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)

                resp = client.get_object(Bucket=bucket_name, Key=s3_key)
                data = resp["Body"].read()
                local_path.write_bytes(data)

                blob_count += 1
                total_bytes += len(data)

    return blob_count, total_bytes
