"""Lazy S3 client for reading blob objects from S3-compatible storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from types_boto3_s3 import S3Client

_client: S3Client | None = None


def _get_client() -> S3Client:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=SETTINGS.s3_endpoint,
            aws_access_key_id=SETTINGS.s3_access_key,
            aws_secret_access_key=SETTINGS.s3_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name=SETTINGS.s3_region,
        )
    return _client


def get_blob(s3_key: str) -> bytes:
    """Fetch an object's body from S3 as raw bytes."""
    resp = _get_client().get_object(Bucket=SETTINGS.s3_bucket, Key=s3_key)
    return resp["Body"].read()


def stream_blob(s3_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield an S3 object in fixed-size chunks (default 1 MB)."""
    resp = _get_client().get_object(Bucket=SETTINGS.s3_bucket, Key=s3_key)
    body = resp["Body"]
    while chunk := body.read(chunk_size):
        yield chunk


def get_blob_size(s3_key: str) -> int:
    """Return the size in bytes of an S3 object."""
    resp = _get_client().head_object(Bucket=SETTINGS.s3_bucket, Key=s3_key)
    return resp["ContentLength"]


def delete_blobs(keys: list[str]) -> int:
    """Delete multiple objects from S3 in batches. Returns count of deleted objects."""
    if not keys:
        return 0
    client = _get_client()
    bucket = SETTINGS.s3_bucket
    objects = [{"Key": k} for k in keys]
    deleted_count = 0
    # S3 maximum limits deletion to 1000 per request
    for i in range(0, len(objects), 1000):
        chunk = objects[i : i + 1000]
        resp = client.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})
        deleted_count += len(chunk) - len(resp.get("Errors", []))
    return deleted_count


def delete_blob_prefix(prefix: str) -> int:
    """Delete all objects under a prefix from S3. Returns count of deleted objects."""
    client = _get_client()
    bucket = SETTINGS.s3_bucket
    deleted_count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        client.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
        deleted_count += len(objects)
    return deleted_count
