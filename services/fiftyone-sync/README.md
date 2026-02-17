# FiftyOne Sync Service

Backend service for the Tator dashboard that integrates a Voxel51/FiftyOne embedded viewer for editing localizations.

## Features

- **Embedding API**: Delegates to Fast-VSS (`http://localhost:8000/embeddings/{project}/`)
  - `POST /embed` - Submit images (multipart/form-data) + project, returns UUID
  - `GET /embed/{uuid}` - Poll for results (proxies to Fast-VSS `/predict/job/{job_id}/{project}`)
  - Set `FASTVSS_API_URL` env var to override Fast-VSS base URL

- **Port isolation**: One FiftyOne App instance per Tator project
  - Port = 5151 + project_id

- **MongoDB database isolation**: Each project uses its own FiftyOne (MongoDB) database by default
  - Default: database name = `{FIFTYONE_DATABASE_DEFAULT}_{project_id}` (env `FIFTYONE_DATABASE_DEFAULT` defaults to `fiftyone_project`, so e.g. `fiftyone_project_1`, `fiftyone_project_2`)
  - Override (all projects): set env `FIFTYONE_DATABASE_NAME` to use a single shared database
  - Override (per request): pass optional query param `database_name` on `GET /launch` and `POST /sync`

- **Launcher**: HostedTemplate integration
  - `GET /message` - Minimal template with single `{{ message }}` (for simple Hosted Template testing)
  - `GET /render` - Full Jinja2 template for FiftyOne viewer iframe (tparams: project, host, base_port, message)
  - `GET /launch` - Allocate port, return FiftyOne App URL
  - `POST /sync` - Trigger Tator-to-FiftyOne sync (stub; requires fiftyone + tator)

## Setup

```bash
cd services/fiftyone-sync
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional: install tator-py from repo
pip install -e ../../scripts/packages/tator-py

# Optional: install fiftyone (requires MongoDB) for full sync
# pip install fiftyone
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Hosted Template applet (recommended)

This service exposes a Jinja2 template for [Tator Hosted Templates](https://www.tator.io/docs/developer-guide/applets-and-dashboards/hosted-templates). When an applet uses it, Tator fetches the template from this service and renders it with template parameters. The result is a dashboard that embeds the FiftyOne viewer in an iframe.

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
     - `iframe_host`: `localhost` — host the **browser** uses to load the FiftyOne iframe. Use a hostname the user’s browser can resolve (e.g. `localhost`). Do **not** use `host.docker.internal` here (browsers cannot resolve it).
     - `message`: optional header text (if set, replaces the default status line)

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

   You can override `iframe_host` or `base_port` here if needed. Keep `iframe_host` as `localhost` (or your Tator host) so the dashboard iframe loads in the browser.

7. Click **Save**.

### 3. Open the applet

Go to the project, then **Analytics** → **Dashboards**. Open the applet you registered. The page will show the FiftyOne viewer iframe for this project (port = `base_port` + `project`).

### Tator in Docker: "Connection refused" and "host.docker.internal server IP address not found"

The Hosted Template URL is **fetched by Tator’s backend** (gunicorn), not by the user’s browser. So the URL must be reachable from inside the container (or host) where gunicorn runs.

- **Connection refused**: Hosted Template URL is fetched by gunicorn — use e.g. `http://host.docker.internal:8001/render`. **"host.docker.internal server IP address not found"**: Shown in the browser; the iframe uses `iframe_host`. Set template param **`iframe_host`** to `localhost` (not `host.docker.internal`).
- **If Tator runs in Docker and this service runs on the host**
  - Hosted Template URL: `http://host.docker.internal:8001/render`. Template param `iframe_host`: `localhost`.
  - Start this service with `--host 0.0.0.0`. Ensure port 8001 is reachable.

- **If both run in Docker**
  - Put both services on the same Docker network and set the Hosted Template URL to the service name and port, e.g. `http://fiftyone-sync:8001/message`.

- **Check**
  - From the host: `curl http://localhost:8001/message` should return HTML.
  - From inside the Tator/gunicorn container: the same URL you put in the Hosted Template (e.g. `http://host.docker.internal:8001/message`) must work (e.g. `curl` from that container).

## MongoDB database (FiftyOne)

Each Tator project gets its own MongoDB database in FiftyOne for isolation unless overridden.

| Env var | Purpose |
|--------|---------|
| `FIFTYONE_DATABASE_DEFAULT` | Prefix for per-project database names. Default `fiftyone_project` → databases `fiftyone_project_1`, `fiftyone_project_2`, etc. |
| `FIFTYONE_DATABASE_NAME` | Override: use this single database name for all projects (ignores default pattern). |

Optional query param **`database_name`** on `GET /launch` and `POST /sync` overrides the database for that project for the request (and for the session on `/launch`). Responses include `database_name` so clients know which DB is used.

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
