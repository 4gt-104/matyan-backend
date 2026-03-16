"""Matyan storage layer — FDB-native key-value storage with msgpack encoding."""

from .encoding import decode_value, encode_value
from .fdb_client import (
    Directories,
    ensure_directories,
    get_db,
    get_directories,
    init_fdb,
)
from .tree import tree_delete, tree_get, tree_keys, tree_set

__all__ = [
    "Directories",
    "decode_value",
    "encode_value",
    "ensure_directories",
    "get_db",
    "get_directories",
    "init_fdb",
    "tree_delete",
    "tree_get",
    "tree_keys",
    "tree_set",
]
