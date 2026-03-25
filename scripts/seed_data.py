#!/usr/bin/env python3
"""Seed FoundationDB with realistic demo data for UI testing.

Usage:
    # Seed data
    python scripts/seed_data.py seed

    # Remove all seeded data
    python scripts/seed_data.py clean
"""

from __future__ import annotations

import math
import random
import struct
import sys
import time
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

    from matyan_backend.fdb_types import Database

SEED_PREFIX = "seed-"
RUN_HASHES = [f"{SEED_PREFIX}{i:04d}" for i in range(12)]
STUCK_RUN_HASH = f"{SEED_PREFIX}stuck"

_BLOB_RUNS = RUN_HASHES[:4]


def _init() -> Database:
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415

    db = init_fdb()
    ensure_directories(db)
    return db


class BlobUploader:
    def __init__(self) -> None:
        from botocore.config import Config as BotoConfig  # noqa: PLC0415
        from botocore.exceptions import ClientError  # noqa: PLC0415
        import os  # noqa: PLC0415
        from matyan_backend.config import SETTINGS  # noqa: PLC0415

        self.backend_type = SETTINGS.blob_backend_type
        
        if self.backend_type == "gcs":
            from google.cloud import storage  # noqa: PLC0415
            from google.auth.credentials import AnonymousCredentials  # noqa: PLC0415
            
            if os.environ.get("STORAGE_EMULATOR_HOST"):
                self.gcs_client = storage.Client(credentials=AnonymousCredentials(), project="test-project")
            else:
                self.gcs_client = storage.Client()
                
            self.gcs_bucket_name = SETTINGS.gcs_bucket
            self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            if not self.gcs_bucket.exists():
                self.gcs_bucket.create()
                
        else:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=SETTINGS.s3_endpoint,
                aws_access_key_id=SETTINGS.s3_access_key,
                aws_secret_access_key=SETTINGS.s3_secret_key,
                config=BotoConfig(signature_version="s3v4"),
                region_name="us-east-1",
            )
            self.s3_bucket_name = SETTINGS.s3_bucket
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket_name)
            except ClientError:
                self.s3_client.create_bucket(Bucket=self.s3_bucket_name)

    def put_blob(self, key: str, data: bytes, content_type: str) -> None:
        if self.backend_type == "gcs":
            blob = self.gcs_bucket.blob(key)
            blob.upload_from_string(data, content_type=content_type)
        else:
            self.s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

    def clean_prefix(self, prefix: str) -> int:
        deleted = 0
        if self.backend_type == "gcs":
            blobs = list(self.gcs_bucket.list_blobs(prefix=prefix))
            if blobs:
                self.gcs_bucket.delete_blobs(blobs)
                deleted += len(blobs)
        else:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.s3_bucket_name, Prefix=prefix):
                contents = page.get("Contents", [])
                if not contents:
                    continue
                objects = [{"Key": obj["Key"]} for obj in contents]
                self.s3_client.delete_objects(Bucket=self.s3_bucket_name, Delete={"Objects": objects, "Quiet": True})
                deleted += len(objects)
        return deleted

def _get_blob_uploader() -> BlobUploader:
    return BlobUploader()


def _make_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Generate a minimal valid PNG with a solid colour fill."""
    import zlib

    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + bytes([r, g, b]) * width
    compressed = zlib.compress(raw_rows)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


def _make_distribution(seed_val: int) -> dict:
    """Generate a synthetic histogram distribution.

    Stores weights as raw float64 bytes in ``data`` (matching Aim's
    ``BLOB(data=hist.tobytes())`` storage) so the backend can serve
    them directly as ``numpy_to_encodable``-style blobs.
    """
    import struct

    rng = random.Random(seed_val)
    bin_count = 32
    weights = [max(0, rng.gauss(50, 20)) for _ in range(bin_count)]
    lo = rng.uniform(-3, -1)
    hi = rng.uniform(1, 3)
    blob = struct.pack(f"<{bin_count}d", *(round(w, 4) for w in weights))
    return {
        "type": "distribution",
        "bin_count": bin_count,
        "range": [round(lo, 4), round(hi, 4)],
        "dtype": "float64",
        "data": blob,
    }


def _make_wav(duration_ms: int = 2000, freqs: list[int] | None = None, rate: int = 22050) -> bytes:
    """Generate a WAV file with a sequence of sine tones."""
    if freqs is None:
        freqs = [440]
    num_samples = rate * duration_ms // 1000
    samples_per_freq = num_samples // len(freqs)
    pcm_parts: list[bytes] = []
    for fi, freq in enumerate(freqs):
        n = samples_per_freq if fi < len(freqs) - 1 else num_samples - fi * samples_per_freq
        fade_len = min(200, n // 4)
        for i in range(n):
            t = i / rate
            amplitude = 0.8 * math.sin(2 * math.pi * freq * t)
            if i < fade_len:
                amplitude *= i / fade_len
            elif i > n - fade_len:
                amplitude *= (n - i) / fade_len
            val = int(32767 * amplitude)
            pcm_parts.append(struct.pack("<h", max(-32768, min(32767, val))))
    pcm = b"".join(pcm_parts)
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def _make_figure(seed_val: int) -> dict:
    """Generate a synthetic plotly figure JSON."""
    rng = random.Random(seed_val)
    n = 20
    x = list(range(n))
    y = [round(rng.gauss(0, 1) + math.sin(i / 3), 3) for i in range(n)]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": x,
                "y": y,
                "name": f"series-{seed_val % 5}",
            },
        ],
        "layout": {
            "title": f"Figure (seed {seed_val})",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
        },
    }


def _seed_blobs(db: Database, uploader: BlobUploader, run_hash: str, ctx_id: int, run_time: float) -> None:
    """Seed image, text, and distribution sequences for a single run."""
    from matyan_backend.storage import runs, sequences

    rng = random.Random(run_hash + "-blobs")
    num_image_steps = 5
    num_text_steps = 8
    num_dist_steps = 10

    for step in range(num_image_steps):
        r, g, b = rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)
        w, h = 64, 64
        png_bytes = _make_png(w, h, r, g, b)
        s3_key = f"{run_hash}/seq/samples/{step}.png"
        uploader.put_blob(s3_key, png_bytes, "image/png")
        metadata = {
            "type": "image",
            "format": "PNG",
            "caption": f"Sample at step {step}",
            "width": w,
            "height": h,
            "s3_key": s3_key,
        }
        sequences.write_sequence_step(
            db,
            run_hash,
            ctx_id,
            "samples",
            step,
            metadata,
            epoch=step,
            timestamp=run_time + step * 600,
        )
    runs.set_trace_info(
        db,
        run_hash,
        ctx_id,
        "samples",
        dtype="image",
        last=0.0,
        last_step=num_image_steps - 1,
    )

    for step in range(num_text_steps):
        text_data = {
            "type": "text",
            "data": f"Prediction at step {step}: class={rng.choice(['cat', 'dog', 'bird', 'fish'])} "
            f"conf={rng.uniform(0.5, 0.99):.2f}",
        }
        sequences.write_sequence_step(
            db,
            run_hash,
            ctx_id,
            "predictions",
            step,
            text_data,
            epoch=step,
            timestamp=run_time + step * 450,
        )
    runs.set_trace_info(
        db,
        run_hash,
        ctx_id,
        "predictions",
        dtype="text",
        last=0.0,
        last_step=num_text_steps - 1,
    )

    for step in range(num_dist_steps):
        dist_data = _make_distribution(hash(run_hash) + step)
        sequences.write_sequence_step(
            db,
            run_hash,
            ctx_id,
            "weight_dist",
            step,
            dist_data,
            epoch=step,
            timestamp=run_time + step * 360,
        )
    runs.set_trace_info(
        db,
        run_hash,
        ctx_id,
        "weight_dist",
        dtype="distribution",
        last=0.0,
        last_step=num_dist_steps - 1,
    )

    # --- Audio sequences ---
    num_audio_steps = 4
    tone_sets = [
        [262, 330, 392],  # C major chord
        [294, 370, 440],  # D major chord
        [330, 415, 494],  # E major chord
        [262, 330, 392, 523],  # C major arpeggio
    ]
    for step in range(num_audio_steps):
        wav_bytes = _make_wav(duration_ms=3000, freqs=tone_sets[step % len(tone_sets)])
        s3_key = f"{run_hash}/seq/audio_clips/{step}.wav"
        uploader.put_blob(s3_key, wav_bytes, "audio/wav")
        metadata = {
            "type": "audio",
            "format": "wav",
            "caption": f"Audio clip at step {step}",
            "s3_key": s3_key,
        }
        sequences.write_sequence_step(
            db,
            run_hash,
            ctx_id,
            "audio_clips",
            step,
            metadata,
            epoch=step,
            timestamp=run_time + step * 500,
        )
    runs.set_trace_info(
        db,
        run_hash,
        ctx_id,
        "audio_clips",
        dtype="audio",
        last=0.0,
        last_step=num_audio_steps - 1,
    )

    # --- Figure sequences ---
    num_figure_steps = 6
    for step in range(num_figure_steps):
        fig_data = {
            "type": "figure",
            "data": _make_figure(hash(run_hash) + step * 7),
        }
        sequences.write_sequence_step(
            db,
            run_hash,
            ctx_id,
            "plots",
            step,
            fig_data,
            epoch=step,
            timestamp=run_time + step * 400,
        )
    runs.set_trace_info(
        db,
        run_hash,
        ctx_id,
        "plots",
        dtype="figure",
        last=0.0,
        last_step=num_figure_steps - 1,
    )


def _seed_file_artifacts(db: Database, uploader: BlobUploader, run_hash: str) -> int:
    """Seed file artifacts (uploaded via log_artifact) for a run. Returns count."""
    from matyan_backend.config import SETTINGS
    from matyan_backend.storage import runs

    artifacts = [
        ("model_checkpoint.pt", b"fake-pytorch-checkpoint-data-" * 100, "application/octet-stream"),
        ("config.yaml", b"lr: 0.001\nbatch_size: 32\noptimizer: adam\nepochs: 50\n", "text/yaml"),
        ("results/metrics.json", b'{"loss": 0.12, "accuracy": 0.95, "f1": 0.93}', "application/json"),
        ("results/confusion_matrix.csv", b"true,pred\ncat,cat\ndog,dog\nbird,cat\n", "text/csv"),
        ("README.md", b"# Run notes\nThis run uses ResNet50 with Adam optimizer.\n", "text/markdown"),
    ]
    for artifact_path, data, content_type in artifacts:
        s3_key = f"{run_hash}/{artifact_path}"
        uploader.put_blob(s3_key, data, content_type)
        runs.set_run_attrs(
            db,
            run_hash,
            ("__blobs__", artifact_path),
            {"s3_key": s3_key, "content_type": content_type},
        )
    return len(artifacts)


_LOG_LINES = [
    "INFO: Starting training...",
    "INFO: Loading dataset cifar10 from cache",
    "WARNING: GPU memory usage at 85%",
    "INFO: Epoch 1/50 — loss: 2.301, acc: 0.112",
    "INFO: Epoch 2/50 — loss: 1.843, acc: 0.289",
    "\033[32mINFO\033[0m: Checkpoint saved to /tmp/ckpt-002.pt",
    "INFO: Epoch 3/50 — loss: 1.521, acc: 0.401",
    "WARNING: Learning rate scheduler reduced lr to 0.0005",
    "INFO: Epoch 4/50 — loss: 1.245, acc: 0.502",
    "INFO: Epoch 5/50 — loss: 1.089, acc: 0.571",
    "\033[32mINFO\033[0m: Checkpoint saved to /tmp/ckpt-005.pt",
    "INFO: Validation accuracy improved: 0.571 -> 0.583",
    "DEBUG: Batch 128/390 processed in 0.042s",
    "INFO: Epoch 10/50 — loss: 0.634, acc: 0.712",
    "WARNING: NaN gradient detected in layer 2, skipping update",
    "INFO: Epoch 15/50 — loss: 0.423, acc: 0.783",
    "INFO: Epoch 20/50 — loss: 0.312, acc: 0.831",
    "\033[32mINFO\033[0m: Checkpoint saved to /tmp/ckpt-020.pt",
    "INFO: Early stopping patience: 3/10",
    "INFO: Training complete. Best accuracy: 0.842",
]

_LOG_RECORDS = [
    (20, "Training started with config: lr=0.001, bs=32"),
    (20, "Dataset loaded: 50000 train, 10000 val samples"),
    (30, "GPU memory usage high — consider reducing batch size"),
    (20, "Epoch 1 completed in 42.3s"),
    (20, "Epoch 2 completed in 41.8s"),
    (20, "Checkpoint saved: epoch 5"),
    (30, "Learning rate reduced by scheduler: 0.001 -> 0.0005"),
    (40, "NaN gradient in layer conv2 at step 1283"),
    (20, "Validation improved: 0.712 -> 0.783"),
    (20, "Training finished — best accuracy 0.842 at epoch 20"),
]


def _seed_logs(db: Database, run_hash: str, run_time: float) -> None:
    from matyan_backend.storage import runs, sequences

    runs.set_context(db, run_hash, 0, {})

    for step, line in enumerate(_LOG_LINES):
        sequences.write_sequence_step(
            db,
            run_hash,
            0,
            "logs",
            step,
            line,
            timestamp=run_time + step * 15,
        )
    runs.set_trace_info(
        db,
        run_hash,
        0,
        "logs",
        dtype="logs",
        last=0.0,
        last_step=len(_LOG_LINES) - 1,
    )

    for step, (level, message) in enumerate(_LOG_RECORDS):
        record = {
            "message": message,
            "log_level": level,
            "timestamp": run_time + step * 30,
            "args": None,
            "logger_info": ["train.py", 100 + step * 5],
        }
        sequences.write_sequence_step(
            db,
            run_hash,
            0,
            "__log_records",
            step,
            record,
            timestamp=run_time + step * 30,
        )
    runs.set_trace_info(
        db,
        run_hash,
        0,
        "__log_records",
        dtype="log_records",
        last=0.0,
        last_step=len(_LOG_RECORDS) - 1,
    )


_SYSTEM_METRICS = [
    "__system__cpu",
    "__system__p_memory_percent",
    "__system__memory_percent",
    "__system__disk_percent",
]

_GPU_METRICS = [
    "__system__gpu",
    "__system__gpu_memory_percent",
    "__system__gpu_power_watts",
    "__system__gpu_temp",
]


def _seed_system_metrics(
    db: Database,
    run_hash: str,
    ctx_id: int,
    num_steps: int,
    run_time: float,
) -> None:
    """Seed system resource metrics that the UI shows in the System tab."""
    from matyan_backend.storage import runs, sequences

    rng = random.Random(run_hash + "_system")
    sys_steps = max(num_steps // 10, 5)

    base_cpu = rng.uniform(20, 60)
    base_mem = rng.uniform(30, 55)
    base_p_mem = rng.uniform(5, 25)
    base_disk = rng.uniform(40, 70)

    for step in range(sys_steps):
        t = step / max(sys_steps - 1, 1)
        ts = run_time + step * 60
        vals = {
            "__system__cpu": round(base_cpu + rng.gauss(0, 8) + 10 * t, 2),
            "__system__p_memory_percent": round(base_p_mem + 5 * t + rng.gauss(0, 1.5), 2),
            "__system__memory_percent": round(base_mem + 3 * t + rng.gauss(0, 2), 2),
            "__system__disk_percent": round(base_disk + 0.5 * t + rng.gauss(0, 0.3), 2),
        }
        for name, val in vals.items():
            sequences.write_sequence_step(db, run_hash, ctx_id, name, step, max(0, val), timestamp=ts)

    for name in _SYSTEM_METRICS:
        runs.set_trace_info(db, run_hash, ctx_id, name, dtype="float", last=0.0, last_step=sys_steps - 1)

    gpu_ctx_id = 10
    runs.set_context(db, run_hash, gpu_ctx_id, {"gpu": 0})

    base_gpu = rng.uniform(40, 90)
    base_gpu_mem = rng.uniform(30, 70)
    base_power = rng.uniform(80, 200)
    base_temp = rng.uniform(45, 70)

    for step in range(sys_steps):
        t = step / max(sys_steps - 1, 1)
        ts = run_time + step * 60
        vals = {
            "__system__gpu": round(base_gpu + rng.gauss(0, 5) + 5 * t, 2),
            "__system__gpu_memory_percent": round(base_gpu_mem + 8 * t + rng.gauss(0, 2), 2),
            "__system__gpu_power_watts": round(base_power + rng.gauss(0, 10) + 15 * t, 2),
            "__system__gpu_temp": round(base_temp + 5 * t + rng.gauss(0, 1), 2),
        }
        for name, val in vals.items():
            sequences.write_sequence_step(db, run_hash, gpu_ctx_id, name, step, max(0, val), timestamp=ts)

    for name in _GPU_METRICS:
        runs.set_trace_info(db, run_hash, gpu_ctx_id, name, dtype="float", last=0.0, last_step=sys_steps - 1)


def seed(db: Database) -> None:  # noqa: PLR0915
    from matyan_backend.storage import entities, runs, sequences

    exp_baseline = entities.create_experiment(db, "baseline", description="Initial baseline runs")
    exp_tuning = entities.create_experiment(db, "lr-tuning", description="Learning rate sweep")
    exp_arch = entities.create_experiment(db, "architecture-v2", description="New architecture tests")

    tag_best = entities.create_tag(db, "best", color="#4caf50", description="Best performing run")
    tag_prod = entities.create_tag(db, "production", color="#2196f3", description="Deployed to production")
    tag_debug = entities.create_tag(db, "debug", color="#ff9800", description="Debug / test run")

    experiments = [exp_baseline, exp_tuning, exp_arch]
    tags = [tag_best, tag_prod, tag_debug]

    configs = [
        {"exp": 0, "lr": 0.001, "batch_size": 32, "optimizer": "adam", "layers": 3, "dropout": 0.1},
        {"exp": 0, "lr": 0.001, "batch_size": 64, "optimizer": "adam", "layers": 3, "dropout": 0.2},
        {"exp": 0, "lr": 0.01, "batch_size": 32, "optimizer": "sgd", "layers": 4, "dropout": 0.1},
        {"exp": 0, "lr": 0.01, "batch_size": 128, "optimizer": "sgd", "layers": 4, "dropout": 0.3},
        {"exp": 1, "lr": 0.0001, "batch_size": 32, "optimizer": "adam", "layers": 3, "dropout": 0.1},
        {"exp": 1, "lr": 0.0005, "batch_size": 32, "optimizer": "adam", "layers": 3, "dropout": 0.1},
        {"exp": 1, "lr": 0.005, "batch_size": 32, "optimizer": "adam", "layers": 3, "dropout": 0.1},
        {"exp": 1, "lr": 0.05, "batch_size": 32, "optimizer": "adam", "layers": 3, "dropout": 0.1},
        {"exp": 2, "lr": 0.001, "batch_size": 64, "optimizer": "adamw", "layers": 6, "dropout": 0.15},
        {"exp": 2, "lr": 0.001, "batch_size": 64, "optimizer": "adamw", "layers": 8, "dropout": 0.2},
        {"exp": 2, "lr": 0.0005, "batch_size": 128, "optimizer": "adamw", "layers": 6, "dropout": 0.1},
        {"exp": 2, "lr": 0.0005, "batch_size": 128, "optimizer": "adamw", "layers": 8, "dropout": 0.25},
    ]

    uploader = _get_blob_uploader()
    base_time = time.time() - 3600 * 24 * 7

    for idx, (run_hash, cfg) in enumerate(zip(RUN_HASHES, configs, strict=True)):
        run_time = base_time + idx * 3600 * 6
        exp = experiments[cfg["exp"]]

        _run = runs.create_run(
            db,
            run_hash,
            name=f"run-{cfg['optimizer']}-lr{cfg['lr']}-bs{cfg['batch_size']}",
            experiment_id=exp["id"],
            description=f"Experiment: {exp['name']}, layers={cfg['layers']}, dropout={cfg['dropout']}",
        )

        entities.set_run_experiment(db, run_hash, exp["id"])

        runs.set_run_attrs(
            db,
            run_hash,
            ("hparams",),
            {
                "learning_rate": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "optimizer": cfg["optimizer"],
                "num_layers": cfg["layers"],
                "dropout": cfg["dropout"],
                "model": "resnet50" if cfg["layers"] <= 4 else "resnet101",
                "dataset": "cifar10",
                "epochs": 50,
            },
        )

        is_finished = idx < 10
        run_duration = 3600.0 * 2  # 2 hours
        runs.update_run_meta(db, run_hash, client_start_ts=run_time, duration=run_duration)
        if is_finished:
            runs.update_run_meta(db, run_hash, active=False, finalized_at=run_time + run_duration)

        if idx == 3:
            runs.update_run_meta(db, run_hash, is_archived=True)

        ctx_id = 0
        runs.set_context(db, run_hash, ctx_id, {})

        random.seed(run_hash)
        num_steps = 200 if is_finished else random.randint(50, 120)
        base_loss = 2.5 - cfg["lr"] * 100
        final_acc = 0.7 + cfg["lr"] * 20 + cfg["layers"] * 0.02

        for step in range(num_steps):
            t = step / max(num_steps - 1, 1)
            noise = random.gauss(0, 0.02 * (1 - t * 0.5))

            loss = base_loss * math.exp(-3 * t) + 0.1 + noise
            acc = final_acc * (1 - math.exp(-4 * t)) + noise * 0.5
            acc = max(0, min(1, acc))

            step_ts = run_time + step * 30
            sequences.write_sequence_step(
                db,
                run_hash,
                ctx_id,
                "loss",
                step,
                round(loss, 5),
                epoch=step // 20,
                timestamp=step_ts,
            )
            sequences.write_sequence_step(
                db,
                run_hash,
                ctx_id,
                "accuracy",
                step,
                round(acc, 5),
                epoch=step // 20,
                timestamp=step_ts,
            )

        last_loss = base_loss * math.exp(-3) + 0.1
        last_acc = final_acc * (1 - math.exp(-4))

        runs.set_trace_info(
            db,
            run_hash,
            ctx_id,
            "loss",
            dtype="float",
            last=round(last_loss, 5),
            last_step=num_steps - 1,
        )
        runs.set_trace_info(
            db,
            run_hash,
            ctx_id,
            "accuracy",
            dtype="float",
            last=round(last_acc, 5),
            last_step=num_steps - 1,
        )

        val_ctx_id = 1
        runs.set_context(db, run_hash, val_ctx_id, {"subset": "val"})
        val_steps = num_steps // 5
        for step in range(val_steps):
            t = step / max(val_steps - 1, 1)
            noise = random.gauss(0, 0.03)
            val_loss = base_loss * math.exp(-2.5 * t) + 0.15 + noise
            val_acc = (final_acc - 0.05) * (1 - math.exp(-3.5 * t)) + noise * 0.5
            val_acc = max(0, min(1, val_acc))
            step_ts = run_time + step * 150
            sequences.write_sequence_step(
                db,
                run_hash,
                val_ctx_id,
                "loss",
                step,
                round(val_loss, 5),
                epoch=step // 4,
                timestamp=step_ts,
            )
            sequences.write_sequence_step(
                db,
                run_hash,
                val_ctx_id,
                "accuracy",
                step,
                round(val_acc, 5),
                epoch=step // 4,
                timestamp=step_ts,
            )

        val_last_loss = base_loss * math.exp(-2.5) + 0.15
        val_last_acc = (final_acc - 0.05) * (1 - math.exp(-3.5))
        runs.set_trace_info(
            db,
            run_hash,
            val_ctx_id,
            "loss",
            dtype="float",
            last=round(val_last_loss, 5),
            last_step=val_steps - 1,
        )
        runs.set_trace_info(
            db,
            run_hash,
            val_ctx_id,
            "accuracy",
            dtype="float",
            last=round(val_last_acc, 5),
            last_step=val_steps - 1,
        )

        # --- System metrics (__system__*) ---
        _seed_system_metrics(db, run_hash, ctx_id, num_steps, run_time)

        # --- __system_params (shown in run overview) ---
        runs.set_run_attrs(
            db,
            run_hash,
            ("__system_params",),
            {
                "packages": ["torch==2.2.0", "numpy==1.26.4", "matyan-client==0.1.0"],
                "env_variables": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/workspace"},
                "git_info": {
                    "branch": "main",
                    "commit": f"abc{idx:04d}",
                    "remote_url": "https://github.com/example/project.git",
                },
                "executable": "/usr/bin/python3",
                "arguments": ["train.py", "--epochs", "50"],
            },
        )

        _seed_logs(db, run_hash, run_time)

        if run_hash in _BLOB_RUNS:
            _seed_blobs(db, uploader, run_hash, ctx_id, run_time)
            _seed_file_artifacts(db, uploader, run_hash)
            print(f"  Created run {run_hash} ({exp['name']}, blobs + artifacts + logs seeded)")
        else:
            print(
                f"  Created run {run_hash} ({exp['name']}, {'active' if not is_finished else 'finished'}, logs seeded)",
            )

    entities.add_tag_to_run(db, RUN_HASHES[0], tag_best["id"])
    entities.add_tag_to_run(db, RUN_HASHES[0], tag_prod["id"])
    entities.add_tag_to_run(db, RUN_HASHES[4], tag_best["id"])
    entities.add_tag_to_run(db, RUN_HASHES[8], tag_debug["id"])
    entities.add_tag_to_run(db, RUN_HASHES[9], tag_debug["id"])

    entities.create_note(db, "Baseline run with default hyperparameters. Good starting point.", run_hash=RUN_HASHES[0])
    entities.create_note(db, "Best LR found in sweep — 0.0001 gives smoothest convergence.", run_hash=RUN_HASHES[4])

    # --- Stuck run: active for 3 days, simulating a crashed process ---
    stuck_time = time.time() - 3600 * 24 * 3
    runs.create_run(
        db,
        STUCK_RUN_HASH,
        name="stuck-crashed-training",
        experiment_id=exp_baseline["id"],
        description="This run crashed 3 days ago but was never finalized.",
    )
    entities.set_run_experiment(db, STUCK_RUN_HASH, exp_baseline["id"])
    runs.set_run_attrs(
        db,
        STUCK_RUN_HASH,
        ("hparams",),
        {"learning_rate": 0.001, "batch_size": 64, "optimizer": "adam", "num_layers": 4, "dropout": 0.2, "model": "resnet50", "dataset": "cifar10", "epochs": 100},
    )
    runs.update_run_meta(db, STUCK_RUN_HASH, client_start_ts=stuck_time)
    runs.set_run_attrs(
        db,
        STUCK_RUN_HASH,
        ("__system_params",),
        {
            "packages": ["torch==2.2.0", "numpy==1.26.4"],
            "env_variables": {"CUDA_VISIBLE_DEVICES": "0,1"},
            "git_info": {"branch": "feature/long-training", "commit": "deadbeef", "remote_url": "https://github.com/example/project.git"},
            "executable": "/usr/bin/python3",
            "arguments": ["train.py", "--epochs", "100", "--patience", "20"],
        },
    )
    stuck_ctx = 0
    runs.set_context(db, STUCK_RUN_HASH, stuck_ctx, {})
    for step in range(50):
        ts = stuck_time + step * 60
        loss = 2.5 * math.exp(-0.02 * step) + random.gauss(0, 0.05)
        sequences.write_sequence_step(db, STUCK_RUN_HASH, stuck_ctx, "loss", step, round(loss, 5), timestamp=ts)
    runs.set_trace_info(db, STUCK_RUN_HASH, stuck_ctx, "loss", dtype="float", last=round(loss, 5), last_step=49)
    _seed_logs(db, STUCK_RUN_HASH, stuck_time)
    print(f"  Created stuck run {STUCK_RUN_HASH} (active for 3 days, simulating crash)")

    total_runs = len(RUN_HASHES) + 1
    blob_count = len(_BLOB_RUNS) * (5 + 8 + 10 + 4 + 6)
    log_count = total_runs * (len(_LOG_LINES) + len(_LOG_RECORDS))
    print(
        f"\nSeeded {total_runs} runs (incl. 1 stuck), {len(experiments)} experiments, {len(tags)} tags, "
        f"{blob_count} blob objects, {log_count} log entries.",
    )


def _clean_blobs(uploader: BlobUploader) -> int:
    """Delete all blob objects for seeded runs. Returns count deleted."""
    deleted = 0
    for run_hash in _BLOB_RUNS:
        prefix = f"{run_hash}/"
        deleted += uploader.clean_prefix(prefix)
    return deleted


def clean(db: Database) -> None:
    from matyan_backend.storage import entities, runs
    from matyan_backend.storage.indexes import rebuild_indexes

    uploader = _get_blob_uploader()
    blobs_deleted = _clean_blobs(uploader)

    deleted = 0
    for run_hash in [*RUN_HASHES, STUCK_RUN_HASH]:
        run = runs.get_run(db, run_hash)
        if run:
            tag_uuids = runs.get_run_tag_uuids(db, run_hash)
            for tu in tag_uuids:
                entities.remove_tag_from_run(db, run_hash, tu)
            runs.delete_run(db, run_hash)
            deleted += 1

    for name in ("baseline", "lr-tuning", "architecture-v2"):
        exp = entities.get_experiment_by_name(db, name)
        if exp:
            entities.delete_experiment(db, exp["id"])

    for name in ("best", "production", "debug"):
        tag = entities.get_tag_by_name(db, name)
        if tag:
            entities.delete_tag(db, tag["id"])

    rebuild_indexes(db)
    print(f"Cleaned {deleted} runs, experiments, tags, and {blobs_deleted} blob object(s).")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("seed", "clean"):
        print("Usage: python scripts/seed_data.py [seed|clean]")
        sys.exit(1)

    db = _init()
    if sys.argv[1] == "seed":
        seed(db)
    else:
        clean(db)


if __name__ == "__main__":
    main()
