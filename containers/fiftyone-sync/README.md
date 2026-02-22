# FiftyOne Sync Compose Stack

MongoDB + fiftyone-sync API. **Default: project 1** uses database `fiftyone_project_1` on this instance; other projects use `fiftyone_project_2`, etc., on the same server.

## Run (Docker)

From the repo root:

```bash
docker compose -f containers/fiftyone-sync/compose.yaml up -d
```

- **API**: http://localhost:8001  
- **MongoDB**: localhost:27017 (exposed for optional host-side workers)

Optional: copy `.env.example` to `.env` in this directory to set `FASTVSS_API_URL`, `REDIS_HOST`, etc. The API container uses `FIFTYONE_DATABASE_URI=mongodb://mongo:27017` by default.

## Development (API on host)

To run only MongoDB and the API locally (e.g. for debugging or general development):

```bash
# From repo root: start MongoDB only
docker compose -f containers/fiftyone-sync/compose.yaml up -d mongo
```

Then in `services/fiftyone-sync`:

```bash
cd services/fiftyone-sync
export FIFTYONE_DATABASE_URI=mongodb://localhost:27017
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

See `services/fiftyone-sync/README.md` for venv setup and Redis/worker options.

## Network

The stack uses network `fiftyone-sync`. To use the main Tator Redis for queued sync, connect the API to that network or set `REDIS_HOST` to the Redis service name.
