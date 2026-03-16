"""Tests for api/runs/_blob_uri.py — Fernet-encrypted blob URI round-trip."""

from __future__ import annotations

import pytest
from cryptography.fernet import InvalidToken

from matyan_backend.api.runs._blob_uri import decode_uri, generate_uri


class TestBlobUriRoundTrip:
    def test_basic(self) -> None:
        uri = generate_uri("run123", 0, "images", 5, 0)
        run_hash, ctx_id, seq_name, step, index = decode_uri(uri)
        assert run_hash == "run123"
        assert ctx_id == 0
        assert seq_name == "images"
        assert step == 5
        assert index == 0

    def test_with_nonzero_index(self) -> None:
        uri = generate_uri("abc", 42, "audios", 10, 3)
        run_hash, ctx_id, seq_name, step, index = decode_uri(uri)
        assert run_hash == "abc"
        assert ctx_id == 42
        assert seq_name == "audios"
        assert step == 10
        assert index == 3

    def test_with_long_run_hash(self) -> None:
        h = "a" * 64
        uri = generate_uri(h, 1, "texts", 0, 0)
        run_hash, _ctx_id, _seq_name, _step, _index = decode_uri(uri)
        assert run_hash == h

    def test_invalid_uri_raises(self) -> None:
        with pytest.raises((InvalidToken, Exception)):
            decode_uri("not-a-valid-encrypted-uri")

    def test_different_inputs_produce_different_uris(self) -> None:
        uri1 = generate_uri("run1", 0, "images", 0, 0)
        uri2 = generate_uri("run2", 0, "images", 0, 0)
        assert uri1 != uri2
