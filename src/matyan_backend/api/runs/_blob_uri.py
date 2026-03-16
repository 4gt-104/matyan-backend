"""Blob URI service for custom object sequences.

Encrypts / decrypts opaque URI strings that reference sequence blobs.
Uses Fernet symmetric encryption with a configurable secret key.
"""

from __future__ import annotations

from cryptography.fernet import Fernet

from matyan_backend.config import SETTINGS


def _get_fernet() -> Fernet:
    key = SETTINGS.blob_uri_secret.encode() if SETTINGS.blob_uri_secret else Fernet.generate_key()
    return Fernet(key)


def generate_uri(run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> str:
    """Create an encrypted blob URI from run + sequence coordinates."""
    payload = f"{run_hash}__seqs__{ctx_id}__{seq_name}__{step}__{index}"
    return _get_fernet().encrypt(payload.encode()).decode()


def decode_uri(uri: str) -> tuple[str, int, str, int, int]:
    """Decrypt a blob URI back to ``(run_hash, ctx_id, seq_name, step, index)``."""
    payload = _get_fernet().decrypt(uri.encode()).decode()
    parts = payload.split("__seqs__")
    run_hash = parts[0]
    rest = parts[1].rsplit("__", maxsplit=2)
    ctx_and_name = rest[0]
    step = int(rest[1])
    index = int(rest[2])
    ctx_sep = ctx_and_name.index("__")
    ctx_id = int(ctx_and_name[:ctx_sep])
    seq_name = ctx_and_name[ctx_sep + 2 :]
    return run_hash, ctx_id, seq_name, step, index
