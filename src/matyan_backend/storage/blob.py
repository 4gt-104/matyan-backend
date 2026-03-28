"""Unified blob abstraction for S3, GCS, and Azure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.config import SETTINGS
from matyan_backend.storage import azure_client, gcs_client, s3_client

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_blob(blob_key: str) -> bytes:
    """Fetch an object's body from the active blob backend as raw bytes."""
    match SETTINGS.blob_backend_type:
        case "azure":
            return azure_client.get_blob(blob_key)
        case "gcs":
            return gcs_client.get_blob(blob_key)
        case _:
            return s3_client.get_blob(blob_key)


def stream_blob(blob_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield an object in fixed-size chunks from the active blob backend."""
    match SETTINGS.blob_backend_type:
        case "azure":
            yield from azure_client.stream_blob(blob_key, chunk_size)
        case "gcs":
            yield from gcs_client.stream_blob(blob_key, chunk_size)
        case _:
            yield from s3_client.stream_blob(blob_key, chunk_size)


def get_blob_size(blob_key: str) -> int:
    """Return the size in bytes of an object from the active blob backend."""
    match SETTINGS.blob_backend_type:
        case "azure":
            return azure_client.get_blob_size(blob_key)
        case "gcs":
            return gcs_client.get_blob_size(blob_key)
        case _:
            return s3_client.get_blob_size(blob_key)


def delete_blobs(keys: list[str]) -> int:
    """Delete a list of objects from the active blob backend. Returns count of deleted objects."""
    match SETTINGS.blob_backend_type:
        case "azure":
            return azure_client.delete_blobs(keys)
        case "gcs":
            return gcs_client.delete_blobs(keys)
        case _:
            return s3_client.delete_blobs(keys)


def delete_blob_prefix(prefix: str) -> int:
    """Delete all objects under a prefix from the active blob backend. Returns count of deleted objects."""
    match SETTINGS.blob_backend_type:
        case "azure":
            return azure_client.delete_blob_prefix(prefix)
        case "gcs":
            return gcs_client.delete_blob_prefix(prefix)
        case _:
            return s3_client.delete_blob_prefix(prefix)
