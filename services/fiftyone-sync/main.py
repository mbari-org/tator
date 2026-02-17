"""
FiftyOne sync service: embedding API, port manager, and launcher for Tator dashboards.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi import FastAPI
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
# Optional: iframe_host (host the browser loads; use localhost or your Tator host—do NOT use host.docker.internal), base_port, message.
LAUNCHER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FiftyOne Viewer – Project {{ project }}</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0; }
    .applet-header { padding: 0.75rem 1rem; background: #2a2a2a; border-bottom: 1px solid #444; font-size: 0.875rem; }
    .applet-header p { margin: 0; }
    .applet-iframe { display: block; width: 100%; height: calc(100% - 48px); border: none; }
  </style>
</head>
<body>
  <div class="applet-header">
    {% if message %}
    <p>{{ message }}</p>
    {% else %}
    <p>Voxel51 FiftyOne viewer – Project {{ project }} (port {{ (base_port | default(5151) | int) + (project | int) }})</p>
    {% endif %}
  </div>
  <iframe
    id="fiftyone-iframe"
    class="applet-iframe"
    src="http://{{ iframe_host | default('localhost') }}:{{ (base_port | default(5151) | int) + (project | int) }}/"
    title="FiftyOne App">
  </iframe>
  <script>
    (function() {
      var project = parseInt("{{ project }}", 10) || 0;
      var iframeHost = "{{ (iframe_host | default('localhost')) }}";
      var basePort = {{ (base_port | default(5151)) | int }};
      var port = basePort + project;
      var iframeSrc = 'http://' + iframeHost + ':' + port + '/';
      console.log('[FiftyOne Dashboard] Launcher loaded', {
        project: project,
        iframe_host: iframeHost,
        base_port: basePort,
        port: port,
        iframe_src: iframeSrc
      });
      var iframe = document.getElementById('fiftyone-iframe');
      if (iframe) {
        iframe.addEventListener('load', function() {
          console.log('[FiftyOne Dashboard] Iframe loaded successfully:', iframeSrc);
        });
        iframe.addEventListener('error', function() {
          console.warn('[FiftyOne Dashboard] Iframe failed to load (e.g. connection refused):', iframeSrc);
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
    Required tparams: project (Tator project ID). Optional: host, base_port (default 5151).
    Port for FiftyOne App = base_port + project.
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
) -> dict:
    """
    Trigger sync: fetch Tator media + localizations, build FiftyOne dataset, launch App.
    Requires fiftyone and tator packages. Returns port, status, and database_name.
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
    )
    result["port"] = port
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
