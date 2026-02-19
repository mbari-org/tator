"""
FiftyOne sync service: embedding API, port manager, and launcher for Tator dashboards.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from embedding_service import (
    get_or_poll_embedding_result,
    init_disk_cache,
    queue_embedding_job,
)
from port_manager import ensure_session, get_database_name, get_port_for_project, get_session

app = FastAPI(
    title="FiftyOne Sync Service",
    description="Embedding API and FiftyOne launcher for Tator dashboards",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(tags=["embedding"])
app_launch = APIRouter(tags=["launcher"])


# --- Embedding API ---


class EmbedResponse(BaseModel):
    uuid: str


@router.post("/embed", response_model=EmbedResponse)
async def post_embed(
    files: list[UploadFile] = File(..., description="Batch of image files"),
    project: str = Form("default", description="Project name for Fast-VSS /embeddings/{project}/"),
) -> EmbedResponse:
    """
    Submit a batch of images for embedding. Forwards to Fast-VSS POST /embeddings/{project}/.
    Returns UUID to fetch results via GET /embed/{uuid}.
    """
    init_disk_cache()
    image_bytes_list: list[bytes] = []
    for f in files:
        data = await f.read()
        image_bytes_list.append(data)
    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="No images provided")
    job_id = await queue_embedding_job(image_bytes_list, project=project)
    return EmbedResponse(uuid=job_id)


@router.get("/embed/{job_id}")
async def get_embed(job_id: str) -> dict:
    """
    Fetch embedding results by UUID. Polls Fast-VSS if pending. Returns status and embeddings when ready.
    """
    result = await get_or_poll_embedding_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


# --- Launcher (HostedTemplate) ---


# Jinja2 template for Tator Hosted Template (applet/dashboard).
# See: https://www.tator.io/docs/developer-guide/applets-and-dashboards/hosted-templates
# Required tparams: project (int, Tator project ID).
# Optional: iframe_host, base_port (5151), ports_per_project (10, must match port_manager).
# For "Sync from Tator" button: sync_service_url, api_url, token; optional version_id, database_name.
# FiftyOne opens in a new window/tab (Open FiftyOne button); no iframe.
LAUNCHER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FiftyOne Viewer – Project {{ project }}</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; font-family: system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0; padding: 1rem; }
    .applet-header { padding: 0.75rem 0; font-size: 0.875rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
    .applet-header p { margin: 0; }
    .applet-header .config-yaml { margin-top: 0.5rem; font-size: 0.75rem; color: #999; white-space: pre-wrap; word-break: break-all; max-height: 6em; overflow: auto; }
    .applet-header .sync-status { font-size: 0.8rem; color: #9c9; }
    .applet-header .sync-status.error { color: #c99; }
    .applet-header button { padding: 0.35rem 0.75rem; cursor: pointer; background: #3a7bd5; color: #fff; border: none; border-radius: 4px; font-size: 0.8rem; }
    .applet-header button:hover { background: #2d6ac4; }
    .applet-header button:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="applet-header">
    <div>
      {% if message %}
      <p>{{ message }}</p>
      {% else %}
      <p>Voxel51 FiftyOne viewer – Project {{ project }} (port {{ (base_port | default(5151) | int) + ((project | int) - 1) * (ports_per_project | default(10) | int) }})</p>
      {% endif %}
      {% if config_yaml %}
      <div id="config-yaml-data" class="config-yaml" title="config_yaml" data-config="{{ config_yaml | e }}">{{ config_yaml }}</div>
      {% endif %}
    </div>
    <button type="button" id="open-fiftyone-btn">Open FiftyOne</button>
    {% if sync_service_url and api_url and token %}
    <button type="button" id="sync-from-tator-btn">Sync from Tator</button>
    <span id="sync-status" class="sync-status" aria-live="polite"></span>
    {% endif %}
  </div>
  <script>
    (function() {
      var project = parseInt("{{ project }}", 10) || 0;
      var iframeHost = "{{ (iframe_host | default('localhost')) }}";
      var basePort = {{ (base_port | default(5151)) | int }};
      var portsPerProject = {{ (ports_per_project | default(10)) | int }};
      var port = basePort + (project - 1) * portsPerProject;
      var appUrl = 'http://' + iframeHost + ':' + port + '/';
      var configYamlEl = document.getElementById('config-yaml-data');
      var configYaml = configYamlEl ? (configYamlEl.getAttribute('data-config') || '') : '';
      if (configYaml) window.FIFTYONE_CONFIG_YAML = configYaml;
      var openBtn = document.getElementById('open-fiftyone-btn');
      if (openBtn) openBtn.addEventListener('click', function() { window.open(appUrl, '_blank'); });
      var syncServiceUrl = "{{ (sync_service_url | default('') | e) }}".replace(/\/$/, '');
      var apiUrl = "{{ api_url | default('') | e }}";
      var token = "{{ token | default('') | e }}";
      var versionId = "{{ version_id | default('') | e }}";
      var databaseName = "{{ database_name | default('') | e }}" || ('fiftyone_project_' + project);
      var syncBtn = document.getElementById('sync-from-tator-btn');
      var syncStatus = document.getElementById('sync-status');
      if (syncBtn && syncStatus && syncServiceUrl && apiUrl && token) {
        syncBtn.addEventListener('click', function() {
          syncBtn.disabled = true;
          syncStatus.textContent = 'Syncing…';
          syncStatus.classList.remove('error');
          var params = new URLSearchParams({ project_id: String(project), api_url: apiUrl, token: token, launch_app: 'true', database_name: databaseName });
          if (versionId) params.set('version_id', versionId);
          var fullSyncUrl = syncServiceUrl + '/sync?' + params.toString();
          fetch(fullSyncUrl, { method: 'POST' })
            .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
            .then(function(result) {
              if (result.ok) {
                syncStatus.textContent = 'Sync done. Opening FiftyOne…';
                var openUrl = appUrl;
                if (result.data.dataset_name) {
                  openUrl = appUrl.replace(/\/$/, '') + '/datasets/' + encodeURIComponent(result.data.dataset_name);
                }
                setTimeout(function() {
                  window.open(openUrl, '_blank');
                  syncStatus.textContent = result.data.sample_count != null
                    ? 'Opened with ' + result.data.sample_count + ' samples.'
                    : 'Opened.';
                }, 3500);
                setTimeout(function() { syncStatus.textContent = ''; }, 5000);
              } else {
                syncStatus.textContent = 'Sync failed: ' + (result.data.detail || result.data.message || 'Unknown error');
                syncStatus.classList.add('error');
              }
            })
            .catch(function(err) {
              syncStatus.textContent = 'Sync error: ' + (err.message || 'Network error');
              syncStatus.classList.add('error');
            })
            .finally(function() { syncBtn.disabled = false; });
        });
      }
    })();
  </script>
</body>
</html>
"""


# Minimal template: single {{ message }} parameter (for simple Hosted Template testing).
MESSAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<body>
<p style="color: white;">{{ message }}</p>
</body>
</html>
"""


@app.get("/message", response_class=HTMLResponse)
async def message_template() -> HTMLResponse:
    """
    Return a minimal Jinja2 template with a single {{ message }} parameter as HTML.
    For Tator Hosted Template: set URL to this endpoint, then use tparam message (e.g. "Hello").
    """
    return HTMLResponse(content=MESSAGE_TEMPLATE, status_code=200)


@app_launch.get("/render", response_class=HTMLResponse)
async def render_launcher() -> HTMLResponse:
    """
    Return Jinja2 template for HostedTemplate. Tator fetches this URL and renders with tparams.
    Required tparams: project (Tator project ID). Optional: iframe_host (host for app URL), base_port (5151), ports_per_project (10).
    "Open FiftyOne" opens the app in a new window. For "Sync from Tator" set: sync_service_url, api_url, token.
    Port = base_port + (project - 1) * ports_per_project.
    """
    return HTMLResponse(LAUNCHER_TEMPLATE)


@app_launch.get("/launch")
async def launch(
    project_id: int = Query(..., description="Tator project ID"),
    version_id: int | None = Query(None),
    api_url: str | None = Query(None, description="Tator REST API base URL"),
    token: str | None = Query(None, description="Tator API token"),
    database_name: str | None = Query(None, description="Override MongoDB database name for this project"),
) -> dict:
    """
    Allocate port and return FiftyOne App URL for the project.
    Sync (fetch Tator data, build dataset) is triggered separately via POST /sync.
    """
    port = ensure_session(project_id, database_name=database_name)
    sess = get_session(project_id)
    host = os.environ.get("FIFTYONE_HOST", "localhost")
    return {
        "project_id": project_id,
        "port": port,
        "url": f"http://{host}:{port}/",
        "database_name": (sess.get("database_name") or get_database_name(project_id)) if sess else get_database_name(project_id),
    }


@app_launch.post("/sync")
async def sync(
    project_id: int = Query(..., description="Tator project ID"),
    version_id: int | None = Query(None),
    api_url: str = Query(..., description="Tator REST API base URL (e.g. https://tator.example.com)"),
    token: str = Query(..., description="Tator API token"),
    database_name: str | None = Query(None, description="Override MongoDB database name for this project"),
    config_path: str | None = Query(None, description="Path to YAML/JSON config file for dataset build"),
    launch_app: bool = Query(True, description="Launch FiftyOne app after sync"),
) -> dict:
    """
    Trigger sync: fetch Tator media + localizations, build FiftyOne dataset, launch App.
    Requires fiftyone and tator packages. Returns port, status, dataset_name, and database_name.
    Config file may specify: dataset_name, include_classes, image_extensions, max_samples.
    """
    from sync import sync_project_to_fiftyone
    from port_manager import get_port_for_project
    port = get_port_for_project(project_id)
    result = sync_project_to_fiftyone(
        project_id=project_id,
        version_id=version_id,
        api_url=api_url.rstrip("/"),
        token=token,
        port=port,
        database_name=database_name,
        config_path=config_path,
        launch_app=launch_app,
    )
    result["port"] = port
    return result


@app_launch.post("/sync-to-tator")
async def sync_to_tator(
    project_id: int = Query(..., description="Tator project ID"),
    version_id: int = Query(..., description="Tator version ID (required for update)"),
    api_url: str = Query(..., description="Tator REST API base URL"),
    token: str = Query(..., description="Tator API token"),
    database_name: str | None = Query(None),
    dataset_name: str | None = Query(None, description="FiftyOne dataset name (default: tator_project_{id})"),
    label_attr: str = Query("Label", description="Tator attribute name for label"),
    score_attr: str | None = Query(None, description="Tator attribute name for score/confidence; omit or empty to skip"),
    debug: bool = Query(False, description="Print per-sample SKIP/UPDATE debug (or set FIFTYONE_SYNC_DEBUG=1)"),
) -> dict:
    """
    Push FiftyOne dataset edits (labels, confidence) back to Tator localizations.
    Requires the dataset to exist (run POST /sync first). Uses elemental_id to match samples.
    """
    from sync import sync_edits_to_tator
    result = sync_edits_to_tator(
        project_id=project_id,
        version_id=version_id,
        api_url=api_url.rstrip("/"),
        token=token,
        dataset_name=dataset_name,
        database_name=database_name,
        label_attr=label_attr,
        score_attr=score_attr,
        debug=debug,
    )
    return result


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Include routers after routes are defined
app.include_router(router)
app.include_router(app_launch)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
