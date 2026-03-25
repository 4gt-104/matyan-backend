"""Unified blob abstraction for S3 and GCS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.config import SETTINGS
from matyan_backend.storage import gcs_client, s3_client

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_blob(blob_key: str) -> bytes:
    """Fetch an object's body from the active blob backend as raw bytes."""
    if SETTINGS.blob_backend_type == "gcs":
        return gcs_client.get_blob(blob_key)
    return s3_client.get_blob(blob_key)


def stream_blob(blob_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield an object in fixed-size chunks from the active blob backend."""
    if SETTINGS.blob_backend_type == "gcs":
        yield from gcs_client.stream_blob(blob_key, chunk_size)
    else:
        yield from s3_client.stream_blob(blob_key, chunk_size)


def get_blob_size(blob_key: str) -> int:
    """Return the size in bytes of an object from the active blob backend."""
    if SETTINGS.blob_backend_type == "gcs":
        return gcs_client.get_blob_size(blob_key)
    return s3_client.get_blob_size(blob_key)
