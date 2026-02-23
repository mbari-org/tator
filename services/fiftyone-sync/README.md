# FiftyOne Sync Service

Backend service for the Tator dashboard that integrates a Voxel51/FiftyOne embedded viewer for editing localizations.

## Features

- **Embedding API**: Delegates to Fast-VSS (`http://localhost:8000/embeddings/{project}/`)
  - `POST /embed` - Submit images (multipart/form-data) + project, returns UUID
  - `GET /embed/{uuid}` - Poll for results (proxies to Fast-VSS `/predict/job/{job_id}/{project}`)
  - Set `FASTVSS_API_URL` env var to override Fast-VSS base URL

- **Port isolation**: One FiftyOne App instance per Tator project (one port per project)
  - Port = 5151 + (project_id - 1)

- **MongoDB database isolation**: A single MongoDB is launched by default (see `containers/fiftyone-sync`). **Project 1** uses database `fiftyone_project_1` on that instance; other projects use `fiftyone_project_2`, etc.
  - Default: database name = `{FIFTYONE_DATABASE_DEFAULT}_{project_id}` (env `FIFTYONE_DATABASE_DEFAULT` defaults to `fiftyone_project`)
  - Override (all projects): set env `FIFTYONE_DATABASE_NAME` to use a single shared database
  - Override (per request): pass optional query param `database_name` on `GET /launch` and `POST /sync`

- **Launcher**: HostedTemplate integration
  - `GET /message` - Minimal template with single `{{ message }}` (for simple Hosted Template testing)
  - `GET /render` - Launcher template with **Open FiftyOne** (opens app in new window) and **Sync from Tator** (tparams: project, iframe_host, base_port, sync_service_url, api_url; user enters API token in the applet and clicks "Test token" to enable sync)
  - `GET /launch` - Allocate port, return FiftyOne App URL
  - `POST /sync` - Trigger Tator-to-FiftyOne sync: fetch media + localizations, crop, build FiftyOne dataset, launch app (requires fiftyone + tator). When Redis is configured, the job is **queued** and the UI stays responsive; poll `GET /sync/status/{job_id}` for completion.
  - `GET /sync/status/{job_id}` - Poll status of a queued sync job (when using Redis).
  - `POST /sync-to-tator` - Push FiftyOne dataset edits (labels, confidence) back to Tator localizations
  - `GET /versions` - Return Tator versions for a project (query params: `project_id`, `api_url`; token via `Authorization: Token <token>` header only)
  - FiftyOne opens in a **new browser tab** (not in an iframe). Set **`iframe_host`** to the host where the FiftyOne app runs so the Open FiftyOne URL is correct (e.g. same host as Tator or `localhost`).

- **Background sync (Redis)**: For projects with millions of localizations, sync can run for a long time. To avoid blocking the web UI, use the **Redis queue** (same as the Tator compose stack). Set `REDIS_HOST=redis` (or your Redis host) for the API; run the sync worker with the same Redis and env (MongoDB, etc.): `python sync_worker.py`. The launcher will then enqueue sync on click and poll until the worker finishes. Redis env: `REDIS_HOST`, `REDIS_PORT` (default `6379`), `REDIS_PASSWORD`, `REDIS_USE_SSL`; or a single `REDIS_URL` (e.g. `redis://host:6380/0`).

## Run (Docker)

The service is intended to be run via the compose stack, which starts MongoDB and the API:

```bash
# From repo root
docker compose -f containers/fiftyone-sync/compose.yaml up -d
```

API: http://localhost:8001. Optional env: copy `containers/fiftyone-sync/.env.example` to `containers/fiftyone-sync/.env` to set `FASTVSS_API_URL`, `REDIS_HOST`, etc.

## Development

For local iteration (no Docker for the API):

```bash
cd services/fiftyone-sync
export FIFTYONE_DATABASE_URI=mongodb://localhost:27017
uvicorn main:app --host 0.0.0.0 --port 8001
```

Use a venv and install deps first: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Start MongoDB separately (e.g. `docker compose -f containers/fiftyone-sync/compose.yaml up -d mongo`).

## Setup (for development)

```bash
cd services/fiftyone-sync
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional: install tator-py from repo
pip install -e ../../scripts/packages/tator-py
```

## Testing

### 1. Without Redis (inline sync)

- **Docker**: `docker compose -f containers/fiftyone-sync/compose.yaml up -d` (API at http://localhost:8001).
- **Development**: Start MongoDB only (`docker compose -f containers/fiftyone-sync/compose.yaml up -d mongo`), then run the API locally (see **Development** above).
- In Tator, open a project that has the FiftyOne applet, then click **Sync from Tator**. The request runs to completion (may block a long time on large projects); when done, FiftyOne opens.
- Or call the API directly (replace with your project id, API URL, token):

```bash
curl -X POST "http://localhost:8001/sync?project_id=1&api_url=https://your-tator.example.com&token=YOUR_TOKEN"
```

Response is the full sync result (no `job_id`).

### 2. With Redis (queued sync)

**Docker**: Add `REDIS_HOST=redis` (and network to Tator’s Redis) or run Redis and set `REDIS_HOST` in `containers/fiftyone-sync/.env`. Run the sync worker in a separate container or on the host with the same env.

**Development**:

1. Start Redis (e.g. `docker run -d --name redis -p 6379:6379 redis:7` or use Tator compose Redis).

2. Start the API with Redis:
   ```bash
   cd services/fiftyone-sync
   export FIFTYONE_DATABASE_URI=mongodb://localhost:27017
   export REDIS_HOST=localhost
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

3. Start the sync worker (another terminal, same env):
   ```bash
   cd services/fiftyone-sync
   source .venv/bin/activate
   export REDIS_HOST=localhost
   export FIFTYONE_DATABASE_URI=mongodb://localhost:27017
   python sync_worker.py
   ```

4. **Trigger sync** (e.g. from the Tator dashboard **Sync from Tator** button, or via curl):
   ```bash
   curl -X POST "http://localhost:8001/sync?project_id=1&api_url=https://your-tator.example.com&token=YOUR_TOKEN"
   ```
   Response should be `{"job_id":"...", "status":"queued", "port":...}`.

5. **Poll status** (replace `JOB_ID` with the returned `job_id`):
   ```bash
   curl "http://localhost:8001/sync/status/JOB_ID"
   ```
   Keep polling until `"status":"finished"` (then `result` has `dataset_name`, `sample_count`) or `"status":"failed"` (then `error` is set).

6. In the **browser**, after clicking **Sync from Tator** you should see “Sync queued…”, then “Sync in progress…”, then “Sync done. Opening FiftyOne…” without the tab freezing.

### 3. Status endpoint without Redis

If Redis is not configured, `GET /sync/status/{job_id}` returns 503 (no queue). That’s expected when not using the worker.

## Hosted Template applet (recommended)

This service exposes a Jinja2 template for [Tator Hosted Templates](https://www.tator.io/docs/developer-guide/applets-and-dashboards/hosted-templates). When an applet uses it, Tator fetches the template and renders it with template parameters. The dashboard shows **Open FiftyOne** (opens the app in a new tab) and sync controls (when sync_service_url and api_url are set). Users enter their Tator API token in the applet and click **Test token** to verify it; the Version dropdown and **Sync from Tator** / **Sync to Tator** buttons then become enabled.

### 1. Register the Hosted Template (organization level)

1. In Tator, go to **Organizations** in the main menu and open your organization.
2. In the left sidebar, under **Hosted Templates**, click **+ Add new**.
3. Fill in:
   - **Name**: e.g. `FiftyOne Viewer`
   - **URL**: `http://<fiftyone-sync-host>:<port>/render`  
     For a minimal test (message only), use `/message` instead and add a template parameter **message** (e.g. `A is for Apple`).  
     Example: `http://localhost:8001/render` (use the URL where this service is running).
   - **Headers**: Leave empty unless the service requires auth.
   - **Template parameters** (optional defaults):
     - `base_port`: `5151`
     - `iframe_host`: host for the FiftyOne app URL when opening in a new tab. **Use the same host you use to open Tator.** If you open Tator at `http://134.89.17.13:8080`, set `iframe_host` to `134.89.17.13` so the app URL is `http://134.89.17.13:5181/...`. If you use `localhost`, the app will try to load from the user’s machine (connection refused when Tator is opened from another host). Do **not** use `host.docker.internal` (browsers cannot resolve it).
     - `message`: optional header text (if set, replaces the default status line)
     - `config_yaml`: optional YAML config string for FiftyOne; shown in the header and exposed as `window.FIFTYONE_CONFIG_YAML` for scripts
     - **Sync from Tator / Sync to Tator**: set `sync_service_url` and `api_url` (do **not** set a token tparam). In the applet, the user enters their Tator API token in the password field and clicks **Test token**; once the token is verified, the Version dropdown is filled and **Sync from Tator** / **Sync to Tator** are enabled. Optionally set **version_id** to preselect a version. The token is used only in the browser for sync and is not stored.

Click **Save**.

### 2. Register an applet using the Hosted Template (project level)

1. Open **Project Settings** for the project where you want the FiftyOne dashboard.
2. In the left sidebar, click **Applets** → **+ Add new**.
3. Set **Name** (e.g. `FiftyOne`) and **Description**.
4. Leave **HTML File** blank.
5. Under **Hosted Template**, select the template you created (e.g. `FiftyOne Viewer`).
6. Under **Template parameters**, add:
   - **name**: `project`  
   - **value**: the project ID (same as this project’s ID).

   Set **`iframe_host`** to the host you use to open Tator (e.g. `134.89.17.13`). If you use `localhost` but open Tator at a different host, the iframe will show "connection refused" because the browser will try to load FiftyOne from the user’s machine, not the server.

7. Click **Save**.

### 3. Open the applet

Go to the project, then **Analytics** → **Dashboards**. Open the applet. Click **Open FiftyOne** to open the viewer in a new tab, or **Sync from Tator** first to sync data and then open the viewer.

### Recommended settings: Tator in Docker (localhost:8080), fiftyone-sync on host (port 8001)

| Setting | Value | Who uses it |
|---------|-------|-------------|
| **URL** (Hosted Template) | `http://host.docker.internal:8001/render` | Tator (in Docker) fetches the template from the host. On Docker Desktop (Mac/Windows) use `host.docker.internal`. On Linux use the host IP (e.g. `172.17.0.1:8001`) or run sync in Docker on the same network. |
| **base_port** | `5151` | Must match `database_manager.BASE_PORT`. |
| **iframe_host** | `localhost` | Host for the FiftyOne app URL when opening in a new tab. |
| **sync_service_url** | `http://localhost:8001` | Required for the "Sync from Tator" button; same machine as Tator from the user's perspective. |
| **api_url** | `http://localhost:8080` | Sync service calls Tator's API; from the host, Tator is at localhost:8080. |

There is no template parameter named **host**; the **URL** field is where Tator fetches the template from. Do **not** set a token tparam; users enter their Tator API token in the applet and click **Test token** to enable sync. The version list and sync requests use the token in the `Authorization` header (for `/versions`) or in the request (for sync); the token is not stored.

### Tator in Docker: "Connection refused" and "host.docker.internal server IP address not found"

The Hosted Template URL is **fetched by Tator’s backend** (gunicorn), not by the user’s browser. So the URL must be reachable from inside the container (or host) where gunicorn runs.

- **Connection refused**: Hosted Template URL is fetched by gunicorn — use e.g. `http://host.docker.internal:8001/render`. **"host.docker.internal server IP address not found"**: Set template param **`iframe_host`** to `localhost` (not `host.docker.internal`) so the Open FiftyOne URL works in the browser.
- **If Tator runs in Docker and this service runs on the host**
  - Hosted Template URL: `http://host.docker.internal:8001/render`. Template params: `iframe_host`: `localhost`, `sync_service_url`: `http://localhost:8001`, `api_url`: `http://localhost:8080`.
  - Start this service with `--host 0.0.0.0`. Ensure port 8001 is reachable.

- **If both run in Docker**
  - Put both services on the same Docker network and set the Hosted Template URL to the service name and port, e.g. `http://fiftyone-sync:8001/message`.

- **Check**
  - From the host: `curl http://localhost:8001/message` should return HTML.
  - From inside the Tator/gunicorn container: the same URL you put in the Hosted Template (e.g. `http://host.docker.internal:8001/message`) must work (e.g. `curl` from that container).

## MongoDB (compose stack)

The compose stack in **`containers/fiftyone-sync`** launches a single MongoDB for the service. **Project 1** uses database `fiftyone_project_1` by default; other projects use `fiftyone_project_2`, etc., on the same instance.

```bash
# From repo root
docker compose -f containers/fiftyone-sync/compose.yaml up -d
```

Set `FIFTYONE_DATABASE_URI=mongodb://localhost:27017` when running the API on the host (or `mongodb://mongo:27017` from a container on the same network). See `containers/fiftyone-sync/.env.example`.

## Database and port allocation

Single MongoDB (launched via `containers/fiftyone-sync`); each Tator project gets its own database for isolation. One port per project: project 1 → 5151, project 2 → 5152, etc.

| Env var | Purpose |
|--------|---------|
| `FIFTYONE_DATABASE_URI` | MongoDB connection URI (default `mongodb://localhost:27017`). Override for remote/alternative MongoDB. |
| `FIFTYONE_DATABASE_DEFAULT` | Prefix for per-project database names. Default `fiftyone_project` → `fiftyone_project_1`, `fiftyone_project_2`, etc. |
| `FIFTYONE_DATABASE_NAME` | Override: use this single database name for all projects (ignores default pattern). |

Optional query param **`database_name`** on `GET /launch` and `POST /sync` overrides the database for that project.

## Sync and FiftyOne Dataset

`POST /sync` fetches Tator media and localizations, crops bounding boxes, builds a FiftyOne dataset, and launches the FiftyOne app.

### Query parameters

| Param | Required | Description |
|-------|----------|-------------|
| `project_id` | yes | Tator project ID |
| `api_url` | yes | Tator REST API base URL |
| `token` | yes | Tator API token |
| `version_id` | no | Version ID filter for localizations |
| `database_name` | no | Override MongoDB database name |
| `config_path` | no | Path to YAML/JSON config file for dataset build |
| `launch_app` | no | Launch FiftyOne app after sync (default: true) |

**Sync-from-Tator flow (dashboard):** The launcher template calls `POST /sync`, then opens the FiftyOne app in a new tab. The frontend opens the **dataset URL** (e.g. `http://host:port/datasets/MyProject_MyVersion`) using `dataset_name` from the response so the App loads the synced dataset directly. The default dataset name is `get_project(project_id).name + "_" + version_name` (from the Tator API). A short delay (~1.2s) before opening gives the FiftyOne server time to serve the correct session state ([Session lifecycle](https://docs.voxel51.com/api/fiftyone.core.session.session.html)). After launch we call `session.refresh()` so the first client connection receives the current dataset.

### Config file (YAML/JSON)

Use `config_path` to pass a config file path:

```yaml
dataset_name: tator_project_dataset
include_classes: [Larvacean, Copepod]   # optional: filter labels
image_extensions: ["*.png", "*.jpg"]
max_samples: 500                         # optional: limit for testing
```

### Embeddings and UMAP visualization

You can optionally compute **embeddings** and a **UMAP** 2D visualization after the dataset is built. Embeddings are fetched from the **embed service** at `{service_url}/embed/{project}`. By default `{project}` is the **Tator project ID** (many services expect this). Set `embeddings.project_name` in config to use a project name or other key instead. Add an `embeddings` block to your config and pass it via `config_path`:

```yaml
embeddings:
  embeddings_field: embeddings         # field to store embedding vectors
  brain_key: umap_viz                 # FiftyOne brain key for UMAP
  umap_seed: 51
  force_embeddings: false             # set true to recompute embeddings
  force_umap: false                   # set true to recompute UMAP
  batch_size: 32                     # batch size for embed service requests
  service_url: null                   # optional; default FASTVSS_API_URL or http://localhost:8000
  project_name: null                  # optional; override for embed service URL path (default: project_id)
```

**Requirements:** The embed service must be running (e.g. Fast-VSS at the URL above). Set `FASTVSS_API_URL` to override the base URL. For UMAP visualization, install `umap-learn` in the sync service venv:

```bash
pip install umap-learn
```

If the embed service is unavailable or UMAP is not installed, sync still runs; embeddings/UMAP are skipped and a message is logged. Embeddings and UMAP results are cached on the dataset; use `force_embeddings` / `force_umap` to recompute.

### Data layout

- Media: `/tmp/fiftyone_sync_project_{id}/download/{media_id}_{name}.jpg`
- Localizations: `/tmp/fiftyone_sync_project_{id}/localizations.jsonl` (JSONL)
- Crops: `/tmp/fiftyone_sync_project_{id}/crops/{media_stem}/{elemental_id}.png`

Labels come from `attributes.Label` (or `attributes.label`) in localizations.

### Sync edits back to Tator

`POST /sync-to-tator` pushes FiftyOne dataset edits (labels, confidence) back to Tator localizations. Run after editing in the FiftyOne app.

| Param | Required | Description |
|-------|----------|-------------|
| `project_id` | yes | Tator project ID |
| `version_id` | yes | Tator version ID (localizations must be in this version) |
| `api_url` | yes | Tator REST API base URL |
| `token` | yes | Tator API token |
| `dataset_name` | no | FiftyOne dataset name (default: `get_project(project_id).name + "_" + version name`) |

```bash
curl -X POST "http://localhost:8001/sync-to-tator?project_id=4&version_id=1&api_url=https://tator.example.com&token=YOUR_TOKEN"
```

Returns `{"status": "ok", "updated": N, "failed": M, "errors": [...]}`.

## Embedding API Usage

```bash
# Submit batch of images (project maps to Fast-VSS /embeddings/{project}/)
curl -X POST http://localhost:8000/embed \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "project=testproject"
# Returns: {"uuid": "..."}

# Fetch results
curl http://localhost:8000/embed/{uuid}
# Returns: {"status": "completed", "embeddings": [[...], [...]]}
```
