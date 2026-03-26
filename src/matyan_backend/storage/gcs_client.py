"""Lazy GCS client for reading blob objects from Google Cloud Storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.cloud import storage

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterator

_client: storage.Client | None = None
_bucket: storage.Bucket | None = None


def _get_bucket() -> storage.Bucket:
    global _client, _bucket  # noqa: PLW0603
    if _client is None:
        _client = storage.Client()
        _bucket = _client.bucket(SETTINGS.gcs_bucket)
    return _bucket  # ty:ignore[invalid-return-type]


def get_blob(s3_key: str) -> bytes:
    """Fetch an object's body from GCS as raw bytes."""
    blob = _get_bucket().blob(s3_key)
    # Return empty bytes if not exists to mimic S3 client potential behavior, or let it raise.
    # The SDK raises google.cloud.exceptions.NotFound
    return blob.download_as_bytes()


def stream_blob(s3_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield a GCS object in fixed-size chunks."""
    blob = _get_bucket().blob(s3_key)
    with blob.open("rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def get_blob_size(s3_key: str) -> int:
    """Return the size in bytes of a GCS object."""
    blob = _get_bucket().get_blob(s3_key)
    if blob is None:
        return 0
    return blob.size or 0
