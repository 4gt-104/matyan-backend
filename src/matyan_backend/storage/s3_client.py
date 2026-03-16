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
            region_name="us-east-1",
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
