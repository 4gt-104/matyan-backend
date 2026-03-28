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


def delete_blobs(keys: list[str]) -> int:
    """Delete multiple objects from GCS in batches. Returns count of deleted objects."""
    if not keys:
        return 0
    bucket = _get_bucket()
    deleted_count = 0
    # GCS batch delete
    for i in range(0, len(keys), 100):
        chunk = keys[i : i + 100]
        # delete_blobs accepts a list of blob names
        bucket.delete_blobs(chunk)
        deleted_count += len(chunk)
    return deleted_count


def delete_blob_prefix(prefix: str) -> int:
    """Delete all objects under a prefix from GCS. Returns count of deleted objects."""
    bucket = _get_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return 0
    # Batch delete up to 100 at a time (or let SDK chunk it if supported,
    # but chunking explicitly is safer).
    deleted_count = 0
    for i in range(0, len(blobs), 100):
        chunk = blobs[i : i + 100]
        bucket.delete_blobs(chunk)
        deleted_count += len(chunk)
    return deleted_count
