# Production configuration

When running the Matyan backend (API server, ingestion worker, control worker, or CLI commands) in production, you can enable strict configuration checks so the process refuses to start if sensitive or critical settings are still at development defaults.

## Enabling production mode

Set the environment variable **`MATYAN_ENVIRONMENT=production`** (or **`ENVIRONMENT=production`**). With this set, the application and workers will not start unless the required settings are explicitly overridden; if any remain at dev defaults, startup fails with a clear error.

Leaving `MATYAN_ENVIRONMENT` unset or set to `development` keeps the current behavior (dev defaults allowed).

## Required overrides in production

When `MATYAN_ENVIRONMENT=production`, the following must be set and must **not** be the default dev values (depending on the `BLOB_BACKEND_TYPE`):

| Setting | Env var (typical) | Backend Type | Purpose |
|--------|--------------------|--------------|---------|
| Blob URI secret | `BLOB_URI_SECRET` | All | Fernet key for blob URIs; must be unique and secret in prod |
| Kafka bootstrap servers | `KAFKA_BOOTSTRAP_SERVERS` | All | Kafka broker address(es) |
| FDB cluster file | `FDB_CLUSTER_FILE` | All | Path to the FoundationDB cluster file |
| S3 endpoint | `S3_ENDPOINT` | `s3` (default) | S3-compatible API URL (e.g. `https://s3.amazonaws.com`) |
| S3 access key | `S3_ACCESS_KEY` | `s3` (default) | S3 credentials |
| S3 secret key | `S3_SECRET_KEY` | `s3` (default) | S3 credentials |
| GCS bucket | `GCS_BUCKET` | `gcs` | GCS bucket name; credentials from ADC |
| Azure container | `AZURE_CONTAINER` | `azure` | Azure container name |
| Azure credentials | `AZURE_CONN_STR` or `AZURE_ACCOUNT_URL` | `azure` | Azure connection string or account URL |

`S3_BUCKET`, `S3_REGION` (defaults: `us-east-1`), and other optional settings keep their defaults unless you override them.

## Supplying secrets

Secrets (blob URI secret, S3 credentials, Kafka SASL credentials when used) should be provided via:

- **Environment variables** — e.g. from Kubernetes Secrets via `envFrom` or `valueFrom.secretKeyRef`, or from a `.env` file that is not committed.
- **A secrets backend** — e.g. HashiCorp Vault, AWS Secrets Manager; inject the values into the process environment or into a file that is then loaded. The application does not integrate with a secrets backend directly; use your orchestration or entrypoint to pull secrets and set env vars before starting the backend.

Never commit production secrets to the repository or use the default dev values (e.g. the default `blob_uri_secret`, `rustfsadmin` S3 credentials) when `MATYAN_ENVIRONMENT=production`.

## Helm / Kubernetes

Production deployments typically set the above env vars from Kubernetes Secrets. See the Helm chart values and README for how to configure `backend`, `ingestionWorker`, and `controlWorker` with `existingSecret` and env. When using the chart in production, set `MATYAN_ENVIRONMENT=production` (or `ENVIRONMENT=production`) in the container env and ensure all required variables are overridden.
