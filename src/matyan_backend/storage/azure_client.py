"""Lazy Azure client for reading blob objects from Azure Blob Storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobServiceClient
from loguru import logger

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterator

_client: BlobServiceClient | None = None


def _get_client() -> BlobServiceClient:
    global _client  # noqa: PLW0603
    if _client is None:
        if SETTINGS.azure_conn_str:
            _client = BlobServiceClient.from_connection_string(SETTINGS.azure_conn_str)
        else:
            _client = BlobServiceClient(
                account_url=SETTINGS.azure_account_url,
                credential=DefaultAzureCredential(),
            )
    return _client


def _get_blob_client(blob_key: str) -> BlobClient:
    return _get_client().get_blob_client(container=SETTINGS.azure_container, blob=blob_key)


def get_blob(blob_key: str) -> bytes:
    """Fetch an object's body from Azure as raw bytes."""
    return cast("bytes", _get_blob_client(blob_key).download_blob().readall())


def stream_blob(blob_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:  # noqa: ARG001
    """Yield an Azure object in chunks."""
    stream = _get_blob_client(blob_key).download_blob()
    yield from stream.chunks()


def get_blob_size(blob_key: str) -> int:
    """Return the size in bytes of an Azure object."""
    props = _get_blob_client(blob_key).get_blob_properties()
    return props.size


def delete_blobs(keys: list[str]) -> int:
    """Delete multiple objects from Azure. Returns count of deleted objects."""
    if not keys:
        return 0
    client = _get_client()
    container = client.get_container_client(SETTINGS.azure_container)
    count = 0
    for k in keys:
        try:
            container.delete_blob(k)
            count += 1
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete blob {} from Azure", k)
    return count


def delete_blob_prefix(prefix: str) -> int:
    """Delete all objects under a prefix from Azure. Returns count of deleted objects."""
    client = _get_client()
    container = client.get_container_client(SETTINGS.azure_container)
    blobs = list(container.list_blobs(name_starts_with=prefix))
    if not blobs:
        return 0
    count = 0
    for b in blobs:
        try:
            container.delete_blob(b.name)
            count += 1
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete blob {} from Azure", b.name)
    return count
