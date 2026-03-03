# Dev Loki Server

A self-contained dev Loki stack for testing the SkyPilot usage/telemetry pipeline without hitting production (`usage-v2.skypilot.co`).

## Architecture

```
Internet
  |
  +--- Port 9090 --> Collector Nginx (write-only: POST /loki/api/v1/push)
  |
  +--- Port 3225 --> Viewer Nginx (basic auth -> Grafana UI + Loki read APIs)
  |
  (internal)  --> Loki (port 9091, filesystem storage)
  (internal)  --> Grafana (port 3000, Loki auto-provisioned as datasource)
```

## Quick Start

```bash
# Deploy on EC2
sky launch -c loki-dev sky/usage/dev/deploy-loki-dev.yaml

# Get the IP from the job output
sky logs loki-dev --no-follow | tail -20

# Point SkyPilot at the dev server
export SKYPILOT_LOG_URL=http://<IP>:9090

# Verify
bash sky/usage/dev/test-push.sh http://<IP>:9090 http://<IP>:3225
```

## Endpoints

| Endpoint | Port | Auth | Description |
|----------|------|------|-------------|
| Collector | 9090 | None | Write-only. Accepts `POST /` and `POST /loki/api/v1/push` |
| Viewer | 3225 | `admin` / `skypilot-dev` | Grafana UI + Loki read APIs (GET only) |

## Testing with SkyPilot

The `SKYPILOT_LOG_URL` env var (added in `sky/usage/constants.py`) overrides the default production Loki URL:

```bash
export SKYPILOT_LOG_URL=http://<IP>:9090
sky status  # Usage data now goes to dev Loki
```

Then open Grafana at `http://<IP>:3225`, go to **Explore**, select the **Loki** datasource, and query:

```
{type="usage"}
```

## Files

| File | Purpose |
|------|---------|
| `deploy-loki-dev.yaml` | SkyPilot task YAML to launch EC2 and deploy the stack |
| `docker-compose.yaml` | Orchestrates Loki, collector, viewer, and Grafana containers |
| `loki-local-config.yaml` | Loki config with local filesystem storage (adapted from production S3 config) |
| `nginx-collector.conf` | Write-only nginx proxy matching `usage_lib.py` POST behavior |
| `nginx-viewer.conf` | Read-only nginx proxy with basic auth for Grafana and Loki queries |
| `grafana/datasources.yaml` | Auto-provisions Loki as the default Grafana datasource |
| `test-push.sh` | Sends a test log entry and optionally queries it back |

## Local Development (without EC2)

You can also run the stack locally with Docker Compose:

```bash
cd sky/usage/dev

# Generate htpasswd
python3 -c "import hashlib,base64;s=base64.b64encode(hashlib.sha1(b'skypilot-dev').digest()).decode();open('.htpasswd','w').write('admin:{SHA}'+s+'\n')"

# Start
docker compose up -d

# Test
bash test-push.sh http://localhost:9090 http://localhost:3225

# Stop
docker compose down
```

## Cleanup

```bash
sky down loki-dev     # Terminate instance (all data lost)
# Or:
sky stop loki-dev     # Stop instance, preserve data
sky start loki-dev    # Resume later
```
