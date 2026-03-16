# Matyan Backend

REST API and workers for the Matyan experiment-tracking stack (fork of Aim). Serves reads and control operations from **FoundationDB**; consumes ingestion and control events from **Kafka**; uses **S3** for artifact blobs. The UI talks to this API; training clients send data via the frontier, which publishes to Kafka consumed by these workers.

## Layout

- **`src/matyan_backend/`** — Python package: FastAPI app (`app.py`), API routes under `api/` (runs, experiments, tags, projects, dashboards, reports, streaming), `storage/` (FDB + S3), `workers/` (ingestion + control Kafka consumers), `jobs/` (FDB lock, used by CLI cleanup commands), `backup/` (export/restore), CLI in `cli.py`.
- **Entrypoints**: `matyan-backend start` (API server, default port 53800), `matyan-backend ingest-worker`, `matyan-backend control-worker`; plus one-off CLI commands (reindex, backup, restore, finish-stale, cleanup-orphan-s3, cleanup-tombstones, convert tensorboard).

## Prerequisites

- Python 3.12+. The package uses `uv` in the repo: `uv run matyan-backend` or install then `matyan-backend` CLI.
- **Runtime dependencies**: FoundationDB (cluster file), Kafka (for workers), S3-compatible store (MinIO/RustFS in dev, AWS S3 in prod). For local dev, typically run FDB + Kafka + S3 via docker-compose.

## Run

- **API server**: `uv run matyan-backend start` (or `matyan-backend start`). Options: `--host`, `--port` (defaults: `0.0.0.0`, 53800). API is under `/api/v1`; health at `/health/ready/`, `/health/live/`, metrics at `/metrics/` when enabled.
- **Workers**: `uv run matyan-backend ingest-worker` and `uv run matyan-backend control-worker`. Both require Kafka and FDB; ingestion worker also writes to FDB and reads S3 config for blob references.
- **CLI (one-off)**: `reindex` (rebuild indexes), `backup` / `restore`, `finish-stale`, `cleanup-orphan-s3`, `cleanup-tombstones`. Cleanup commands are intended for CronJobs or cron; use `--dry-run` to preview and `--lock-ttl-seconds` for FDB-based single-run locking. Optional: `convert tensorboard` to convert TensorBoard logs to backup format.

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MATYAN_ENVIRONMENT` / `ENVIRONMENT` | `development` | When `production`, strict checks apply (see Production configuration). |
| `LOG_LEVEL` | `INFO` | Log level (loguru + uvicorn). |
| `FDB_CLUSTER_FILE` | `fdb.cluster` | Path to FoundationDB cluster file. |
| `S3_ENDPOINT` | `http://localhost:9000` | S3-compatible API URL. |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | (dev defaults) | S3 credentials. |
| `S3_BUCKET` | `matyan-artifacts` | Bucket for artifacts. |
| `BLOB_URI_SECRET` | (dev default) | Fernet key for blob URIs; must be set in production. |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker list. |
| `KAFKA_DATA_INGESTION_TOPIC` | `data-ingestion` | Topic for ingestion messages. |
| `KAFKA_CONTROL_EVENTS_TOPIC` | `control-events` | Topic for control events. |
| `KAFKA_SECURITY_PROTOCOL` / `KAFKA_SASL_*` | (empty) | Optional Kafka SASL. |
| `METRICS_ENABLED` | `true` | Expose Prometheus metrics. |
| `METRICS_PORT` | `9090` | Port for metrics HTTP server (workers). |
| `INGEST_MAX_MESSAGES_PER_TXN` | `100` | Max messages per FDB transaction (ingestion worker). |
| `INGEST_MAX_TXN_BYTES` | `8388608` (8 MB) | Target max transaction size; FDB limit is 10 MB. |
| `CORS_ORIGINS` | (localhost list) | Allowed origins for CORS. |

Source of truth: [config.py](src/matyan_backend/config.py).

## Production configuration

See **[docs/PRODUCTION_CONFIG.md](docs/PRODUCTION_CONFIG.md)** for enabling production mode (`MATYAN_ENVIRONMENT=production`), required overrides, and supplying secrets via env or a secrets backend.

## Deployment

- **Docker**: Build the backend image (context from repo root); run API and workers as separate processes or containers.
- **Kubernetes/Helm**: The chart in `deploy/helm/matyan` deploys the backend API, ingestion worker, and control worker as separate Deployments; optional CronJobs for `cleanup-orphan-s3` and `cleanup-tombstones`. Configure FDB, S3, and Kafka via chart values; see the chart README. Set `MATYAN_ENVIRONMENT=production` and required env for production.

## Related

- **UI**: matyan-ui calls this backend REST API.
- **Frontier**: matyan-frontier publishes to Kafka; backend workers consume.
- **API models**: matyan-api-models shared types (Kafka messages, run creation, etc.).
- **Monorepo**: This package lives under `extra/matyan-backend` in the matyan-core repo.
