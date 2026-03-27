"""Download S3 artifacts for backed-up runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
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


def _make_azure_client() -> BlobServiceClient:
    if SETTINGS.azure_conn_str:
        return BlobServiceClient.from_connection_string(SETTINGS.azure_conn_str)
    return BlobServiceClient(
        account_url=SETTINGS.azure_account_url,
        credential=DefaultAzureCredential(),
    )


def export_blobs_for_run(  # noqa: C901
    run_hash: str,
    backup_dir: Path,
    *,
    s3_client: S3Client | None = None,
    gcs_client: storage.Client | None = None,
    azure_client: BlobServiceClient | None = None,
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
            blob_key = blob.name
            relative = blob_key[len(prefix) :]
            if not relative:
                continue

            local_path = blobs_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            data = blob.download_as_bytes()
            local_path.write_bytes(data)

            blob_count += 1
            total_bytes += len(data)

    elif SETTINGS.blob_backend_type == "azure":
        client = azure_client or _make_azure_client()
        container_client = client.get_container_client(SETTINGS.azure_container)
        for blob in container_client.list_blobs(name_starts_with=prefix):
            blob_key = blob.name
            relative = blob_key[len(prefix) :]
            if not relative:
                continue

            local_path = blobs_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            data = container_client.get_blob_client(blob.name).download_blob().readall()
            if isinstance(data, str):
                data = data.encode()
            local_path.write_bytes(data)

            blob_count += 1
            total_bytes += len(data)

    else:
        client = s3_client or _make_s3_client()
        bucket_name = SETTINGS.s3_bucket
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                blob_key = obj["Key"]
                relative = blob_key[len(prefix) :]
                if not relative:
                    continue

                local_path = blobs_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)

                resp = client.get_object(Bucket=bucket_name, Key=blob_key)
                data = resp["Body"].read()
                local_path.write_bytes(data)

                blob_count += 1
                total_bytes += len(data)

    return blob_count, total_bytes
