# Identity Engine Worker

Standalone GPU worker for the Identity Engine.

The first version is intentionally a safe handshake worker. It polls the portal,
claims jobs, heartbeats, and reports a controlled failure until real GPU inference
is enabled. This lets us prove Railway-to-Unraid sync before trusting embeddings.

## Architecture

```text
Railway Portal + Postgres
  -> identity_jobs table
  <- Unraid worker polls over HTTPS
Unraid GTX 1660 Super
  -> future SCRFD/ArcFace/AdaFace inference
  -> posts templates/results back to Railway
```

Railway does not need inbound access to your home network.

## Required Railway Variable

Set this on the Railway portal service:

```text
IDENTITY_WORKER_TOKEN=<long random secret>
```

The Unraid worker uses the same value.

## Unraid Prerequisites

1. Install the Unraid NVIDIA Driver plugin.
2. Confirm GPU visibility from Unraid terminal:

```bash
nvidia-smi
```

3. Install/use Docker Compose Manager, or run the `docker run` command below.

## Recommended Unraid Install With Compose

Copy this project folder to Unraid, for example:

```text
/mnt/user/appdata/foxmar-identity-engine-src
```

Create appdata folders:

```bash
mkdir -p /mnt/user/appdata/foxmar-identity-engine/models
mkdir -p /mnt/user/appdata/foxmar-identity-engine/cache
mkdir -p /mnt/user/appdata/foxmar-identity-engine/logs
```

Create `.env`:

```bash
cd /mnt/user/appdata/foxmar-identity-engine-src
cp .env.example .env
nano .env
```

Set:

```text
PORTAL_BASE_URL=https://your-portal-domain.example
IDENTITY_WORKER_TOKEN=<same token from Railway>
IDENTITY_WORKER_ID=unraid-gtx1660
ENGINE_MODE=stub
STUB_JOB_RESULT=fail
```

Build and start:

```bash
docker compose up -d --build
```

Watch logs:

```bash
docker logs -f foxmar-identity-engine
```

Expected startup logs:

```text
Starting Identity Engine worker id=unraid-gtx1660 mode=stub
NVIDIA GPU detected: NVIDIA GeForce GTX 1660 SUPER, ...
Portal identity health: ...
```

## Docker Run Alternative

```bash
docker build -t foxmar/identity-engine:local .

docker run -d \
  --name foxmar-identity-engine \
  --restart unless-stopped \
  --gpus all \
  -e PORTAL_BASE_URL="https://your-portal-domain.example" \
  -e IDENTITY_WORKER_TOKEN="<same token from Railway>" \
  -e IDENTITY_WORKER_ID="unraid-gtx1660" \
  -e ENGINE_MODE="stub" \
  -e STUB_JOB_RESULT="fail" \
  -e NVIDIA_VISIBLE_DEVICES="all" \
  -e NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
  -v /mnt/user/appdata/foxmar-identity-engine/models:/models \
  -v /mnt/user/appdata/foxmar-identity-engine/cache:/cache \
  -v /mnt/user/appdata/foxmar-identity-engine/logs:/logs \
  foxmar/identity-engine:local
```

## Manual Portal Check

From Unraid terminal:

```bash
curl -s "$PORTAL_BASE_URL/api/v1/identity/health"
```

Worker-auth check:

```bash
curl -s -X POST "$PORTAL_BASE_URL/api/v1/identity/jobs/next" \
  -H "Authorization: Bearer $IDENTITY_WORKER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"worker_id":"unraid-manual-test"}'
```

Expected before jobs exist:

```json
{"success":true,"job":null}
```

## Notes

- `ENGINE_MODE=stub` is safe. It will not create templates.
- `STUB_JOB_RESULT=fail` proves the full claim/fail loop without pretending a
  student was indexed.
- Once the portal can create enrollment jobs, this container can prove the whole
  Railway/Unraid job lifecycle.
- Real model inference will be added behind the same worker API.
