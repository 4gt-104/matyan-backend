"""Download S3 artifacts for backed-up runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig

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
        region_name="us-east-1",
    )


def export_blobs_for_run(run_hash: str, backup_dir: Path, *, s3_client: S3Client | None = None) -> tuple[int, int]:
    """Download all S3 objects under ``<run_hash>/`` into the backup.

    Returns ``(blob_count, total_bytes)``.
    """
    client = s3_client or _make_s3_client()
    bucket = SETTINGS.s3_bucket
    prefix = f"{run_hash}/"
    blobs_dir = backup_dir / "runs" / run_hash / "blobs"

    blob_count = 0
    total_bytes = 0

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key: str = obj["Key"]
            relative = s3_key[len(prefix) :]
            if not relative:
                continue

            local_path = blobs_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            resp = client.get_object(Bucket=bucket, Key=s3_key)
            data = resp["Body"].read()
            local_path.write_bytes(data)

            blob_count += 1
            total_bytes += len(data)

    return blob_count, total_bytes
