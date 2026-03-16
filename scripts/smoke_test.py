"""Smoke test to verify FoundationDB and S3 (RustFS) connectivity.

Usage:
    cd matyan-backend
    docker compose up -d
    # wait ~10s for fdb-init to configure the database
    python scripts/smoke_test.py
"""

import sys

import boto3
import fdb
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from matyan_backend.config import SETTINGS


def test_fdb() -> None:
    """Test FoundationDB connectivity: write, read, delete a test key."""
    print(f"[FDB] Connecting via cluster file: {SETTINGS.fdb_cluster_file}")
    fdb.api_version(SETTINGS.fdb_api_version)
    db = fdb.open(SETTINGS.fdb_cluster_file)

    test_key = b"matyan_smoke_test"
    test_value = b"ok"

    db[test_key] = test_value
    print("[FDB] Write: OK")

    result = db[test_key]
    assert result == test_value, f"Expected {test_value!r}, got {result!r}"
    print("[FDB] Read:  OK")

    del db[test_key]
    result = db[test_key]
    assert result is None, f"Expected None after delete, got {result!r}"
    print("[FDB] Delete: OK")

    print("[FDB] All checks passed.")


def test_s3() -> None:
    """Test S3 connectivity: create bucket if not exists, write/read/delete."""
    print(f"\n[S3] Connecting to: {SETTINGS.s3_endpoint}")
    s3 = boto3.client(
        "s3",
        endpoint_url=SETTINGS.s3_endpoint,
        aws_access_key_id=SETTINGS.s3_access_key,
        aws_secret_access_key=SETTINGS.s3_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )

    bucket = SETTINGS.s3_bucket

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[S3] Bucket '{bucket}' already exists.")
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"[S3] Created bucket '{bucket}'.")

    s3.put_object(Bucket=bucket, Key="smoke_test", Body=b"ok")
    print("[S3] Write: OK")

    resp = s3.get_object(Bucket=bucket, Key="smoke_test")
    body = resp["Body"].read()
    assert body == b"ok", f"Expected b'ok', got {body!r}"
    print("[S3] Read:  OK")

    s3.delete_object(Bucket=bucket, Key="smoke_test")
    print("[S3] Delete: OK")

    print("[S3] All checks passed.")


def main() -> None:
    print("=== Matyan Smoke Test ===\n")

    fdb_ok = True
    s3_ok = True

    try:
        test_fdb()
    except Exception as e:
        print(f"[FDB] FAILED: {e}")
        fdb_ok = False

    try:
        test_s3()
    except Exception as e:
        print(f"[S3] FAILED: {e}")
        s3_ok = False

    print("\n=== Summary ===")
    print(f"  FoundationDB: {'PASS' if fdb_ok else 'FAIL'}")
    print(f"  S3:           {'PASS' if s3_ok else 'FAIL'}")

    if not (fdb_ok and s3_ok):
        sys.exit(1)

    print("\nAll systems operational.")


if __name__ == "__main__":
    main()
