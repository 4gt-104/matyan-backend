# Production configuration

When running the Matyan backend (API server, ingestion worker, control worker, or CLI commands) in production, you can enable strict configuration checks so the process refuses to start if sensitive or critical settings are still at development defaults.

## Enabling production mode

Set the environment variable **`MATYAN_ENVIRONMENT=production`** (or **`ENVIRONMENT=production`**). With this set, the application and workers will not start unless the required settings are explicitly overridden; if any remain at dev defaults, startup fails with a clear error.

Leaving `MATYAN_ENVIRONMENT` unset or set to `development` keeps the current behavior (dev defaults allowed).

## Required overrides in production

When `MATYAN_ENVIRONMENT=production`, the following must be set and must **not** be the default dev values:

| Setting | Env var (typical) | Purpose |
|--------|--------------------|--------|
| Blob URI secret | `BLOB_URI_SECRET` | Fernet key for blob URIs; must be unique and secret in prod |
| S3 endpoint | `S3_ENDPOINT` | S3-compatible API URL (e.g. `https://s3.amazonaws.com`) |
| S3 access key | `S3_ACCESS_KEY` | S3 credentials |
| S3 secret key | `S3_SECRET_KEY` | S3 credentials |
| Kafka bootstrap servers | `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address(es) |
| FDB cluster file | `FDB_CLUSTER_FILE` | Path to the FoundationDB cluster file |

`S3_BUCKET` and other optional settings keep their defaults unless you override them.

## Supplying secrets

Secrets (blob URI secret, S3 credentials, Kafka SASL credentials when used) should be provided via:

- **Environment variables** — e.g. from Kubernetes Secrets via `envFrom` or `valueFrom.secretKeyRef`, or from a `.env` file that is not committed.
- **A secrets backend** — e.g. HashiCorp Vault, AWS Secrets Manager; inject the values into the process environment or into a file that is then loaded. The application does not integrate with a secrets backend directly; use your orchestration or entrypoint to pull secrets and set env vars before starting the backend.

Never commit production secrets to the repository or use the default dev values (e.g. the default `blob_uri_secret`, `rustfsadmin` S3 credentials) when `MATYAN_ENVIRONMENT=production`.

## Helm / Kubernetes

Production deployments typically set the above env vars from Kubernetes Secrets. See the Helm chart values and README for how to configure `backend`, `ingestionWorker`, and `controlWorker` with `existingSecret` and env. When using the chart in production, set `MATYAN_ENVIRONMENT=production` (or `ENVIRONMENT=production`) in the container env and ensure all required variables are overridden.
