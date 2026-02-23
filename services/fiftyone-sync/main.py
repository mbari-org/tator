"""
FiftyOne sync service: embedding API, port manager, and launcher for Tator dashboards.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import httpx

from embedding_service import (
    FASTVSS_BASE_URL,
    get_or_poll_embedding_result,
    init_disk_cache,
    queue_embedding_job,
)
from database_manager import get_database_entry
from database_uri_config import DatabaseUriConfig, database_name_from_uri

TATOR_INTERNAL_API_URL = os.environ.get("TATOR_INTERNAL_API_URL", "").strip().rstrip("/")

def _resolve_api_url(api_url: str) -> str:
    """Return the Docker-internal API URL when running inside a container.

    The browser sends api_url=http://localhost:8080 which is unreachable from
    inside a Docker container (localhost == the container itself).  When
    TATOR_INTERNAL_API_URL is set, swap out the host so the tator SDK can
    reach the Tator nginx service on the Docker network.
    """
    if not TATOR_INTERNAL_API_URL:
        return api_url.rstrip("/")
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    if parsed.hostname in ("localhost", "127.0.0.1"):
        return TATOR_INTERNAL_API_URL
    return api_url.rstrip("/")

# Origins allowed for CORS (dashboard and applet iframes). When allow_credentials=True,
# browsers require a specific origin; "*" is not valid.
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_cors_origins_env = os.environ.get("FIFTYONE_SYNC_CORS_ORIGINS", "").strip()
if _cors_origins_env:
    CORS_ORIGINS.extend(origin.strip() for origin in _cors_origins_env.split(",") if origin.strip())

app = FastAPI(
    title="FiftyOne Sync Service",
    description="Embedding API and FiftyOne launcher for Tator dashboards",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

router = APIRouter(tags=["embedding"])
app_launch = APIRouter(tags=["launcher"])

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
    if not FASTVSS_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable: FASTVSS_API_URL is not set",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{FASTVSS_BASE_URL}/projects")
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {e!s}",
        ) from e
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


@router.get("/vss-embedding")
async def get_vss_embedding() -> dict:
    """
    Proxy to embedding service GET /projects. Returns {"projects": ["name", ...]}.
    Uses FASTVSS_API_URL. Used by the launcher applet to show embedding service
    availability and whether the current project is registered. On failure returns 503.
    """
    if not FASTVSS_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="FASTVSS_API_URL is not set",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{FASTVSS_BASE_URL}/projects")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {e!s}",
        ) from e


# --- Launcher (HostedTemplate) ---


# Jinja2 template for Tator Hosted Template (applet/dashboard).
# See: https://www.tator.io/docs/developer-guide/applets-and-dashboards/hosted-templates
# Required tparams: project (int, Tator project ID).
# Optional: iframe_host, base_port (5151).
# For "Sync from Tator" / "Sync to Tator": set sync_service_url, api_url (no token tparam).
# User enters their Tator API token in the applet and clicks "Test token"; sync controls enable after token is verified.
# Optional tparams: version_id, database_name, project_name (vss_project for embedding status only).
# database-info is called with project_id; send api_url and Authorization to resolve to config key (project name).
# Embedding service status: server uses FASTVSS_API_URL; GET /vss-embedding shows availability.
# FiftyOne opens in a new window/tab (Open FiftyOne button); no iframe.
LAUNCHER_TEMPLATE = r"""
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
    .applet-header .btn-icon { margin-right: 0.25rem; }
    .applet-header .btn-icon.end { margin-right: 0; margin-left: 0.25rem; }
    .applet-header select { padding: 0.35rem 0.5rem; font-size: 0.8rem; background: #2a2a2a; color: #e0e0e0; border: 1px solid #555; border-radius: 4px; min-width: 10rem; }
    .applet-header input[type="password"] { padding: 0.35rem 0.5rem; font-size: 0.8rem; background: #2a2a2a; color: #e0e0e0; border: 1px solid #555; border-radius: 4px; min-width: 12rem; }
    .applet-header a.token-link { font-size: 0.8rem; color: #6ab; }
    .applet-header a.token-link:hover { color: #8cd; text-decoration: underline; }
    .applet-header table { border-collapse: collapse; width: 100%; max-width: 56rem; }
    .applet-header th { text-align: left; padding: 0.5rem 1rem 0.25rem 0; font-size: 0.75rem; color: #999; font-weight: 600; vertical-align: top; white-space: nowrap; }
    .applet-header td { padding: 0.25rem 0; vertical-align: middle; }
    .applet-header tr + tr th { padding-top: 0.75rem; }
    .applet-header .cell-controls { display: flex; align-items: center; gap: 0.5rem 1rem; flex-wrap: wrap; }
  </style>
</head>
<body>
  <div class="applet-header">
    <table>
      <tbody>
        <tr>
          <th>Info</th>
          <td>
            {% if message %}
            <p>{{ message }}</p>
            {% else %}
            <p>Voxel51 FiftyOne viewer – Project {{ project }} (port {{ (base_port | default(5151) | int) + ((project | int) - 1) }})</p>
            {% endif %}
            {% if config_yaml %}
            <div id="config-yaml-data" class="config-yaml" title="config_yaml" data-config="{{ config_yaml | e }}">{{ config_yaml }}</div>
            {% endif %}
          </td>
        </tr>
        {% if sync_service_url and api_url %}
        <tr>
          <th>Token</th>
          <td>
            <div class="cell-controls">
              <input type="password" id="user-token" placeholder="Your Tator API token" autocomplete="off" />
              <a href="{{ ((api_url | default('')).rstrip('/') ~ '/token') | e }}" target="_blank" rel="noopener" class="token-link">Get your token</a>
              <button type="button" id="test-token-btn">Test token</button>
            </div>
          </td>
        </tr>
        <tr>
          <th>Version</th>
          <td>
            <div class="cell-controls">
              <select id="version-select" aria-label="version" disabled>
                <option value="">Enter token and click Test</option>
              </select>
            </div>
          </td>
        </tr>
        <tr>
          <th>Sync</th>
          <td>
            <div class="cell-controls">
              <button type="button" id="sync-from-tator-btn" disabled title="Loads the selected version and launches a Voxel51 (FiftyOne) viewer in another tab. If the Embedding Service is not available, the viewer will still launch but will not contain embeddings."><span class="btn-icon" aria-hidden="true">←</span>Load from Tator</button>
              <button type="button" id="sync-to-tator-btn" disabled title="Pushes any revised data from FiftyOne back to the selected version.">Sync to Tator<span class="btn-icon end" aria-hidden="true">→</span></button>
              <span id="sync-status" class="sync-status" aria-live="polite"></span>
            </div>
          </td>
        </tr>
        {% endif %}
        <tr>
          <th>Embedding service</th>
          <td>
            <span id="embedding-status" class="sync-status" aria-live="polite">Checking…</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
  <script>
    (function() {
      var project = parseInt("{{ project }}", 10) || 0;
      var iframeHost = "{{ (iframe_host | default('localhost')) }}";
      var port = {{ (port | default(5151)) | int }};
      var appUrl = 'http://' + iframeHost + ':' + port + '/';
      var syncServiceUrl = "{{ (sync_service_url | default('') | e) }}".replace(/\/$/, '');
      var projectName = "{{ (project_name | default('') | e) }}".trim();
      var databaseName = "{{ database_name | default('') | e }}" || ('fiftyone_project_' + project);
      var databaseUri = '';
      // Do not call /database-info on load: it requires valid Authorization. Resolve port/database only after token is provided and tested.
      if (project && syncServiceUrl) {
        console.log('[FiftyOne applet] database-info deferred until token is tested; initial port=', port);
      }
      var configYamlEl = document.getElementById('config-yaml-data');
      var configYaml = configYamlEl ? (configYamlEl.getAttribute('data-config') || '') : '';
      if (configYaml) window.FIFTYONE_CONFIG_YAML = configYaml;
      var apiUrl = "{{ api_url | default('') | e }}";
      var initialVersionId = "{{ version_id | default('') | e }}";
      var syncBtn = document.getElementById('sync-from-tator-btn');
      var syncStatus = document.getElementById('sync-status');
      var versionSelect = document.getElementById('version-select');
      var syncToTatorBtn = document.getElementById('sync-to-tator-btn');
      var tokenInput = document.getElementById('user-token');
      var testTokenBtn = document.getElementById('test-token-btn');
      var tokenVerified = false;
      var hasDatabaseEntry = false;
      var versionId = '';
      function getToken() {
        return tokenInput ? tokenInput.value.trim() : '';
      }
      function setVersionFromDropdown() {
        versionId = versionSelect && versionSelect.value ? versionSelect.value : '';
        if (syncBtn) syncBtn.disabled = !tokenVerified || !hasDatabaseEntry;
        if (syncToTatorBtn) syncToTatorBtn.disabled = !tokenVerified || !hasDatabaseEntry || !versionId;
      }
      function setSyncControlsEnabled(enabled) {
        tokenVerified = enabled;
        if (versionSelect) versionSelect.disabled = !enabled;
        if (syncBtn) syncBtn.disabled = !enabled || !hasDatabaseEntry;
        if (syncToTatorBtn) syncToTatorBtn.disabled = !enabled || !hasDatabaseEntry || !versionId;
      }
      function loadVersions(token) {
        if (!versionSelect || !syncServiceUrl || !apiUrl || !token) return;
        versionSelect.innerHTML = '<option value="">Loading…</option>';
        fetch(syncServiceUrl + '/versions?project_id=' + project + '&api_url=' + encodeURIComponent(apiUrl), {
          headers: { 'Authorization': 'Token ' + token }
        })
          .then(function(r) {
            if (!r.ok) return r.json().then(function(d) { throw new Error(d.detail || r.statusText); });
            return r.json();
          })
          .then(function(versions) {
            versionSelect.innerHTML = '';
            (versions || []).forEach(function(v) {
              var opt = document.createElement('option');
              opt.value = String(v.id);
              opt.textContent = v.name + (v.number != null ? ' (' + v.number + ')' : '');
              versionSelect.appendChild(opt);
            });
            if (initialVersionId && versionSelect.querySelector('option[value="' + initialVersionId + '"]')) {
              versionSelect.value = initialVersionId;
            } else if (versionSelect.options.length) {
              versionSelect.selectedIndex = 0;
            }
            setVersionFromDropdown();
          })
          .catch(function(err) {
            versionSelect.innerHTML = '';
            var opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Failed to load versions';
            versionSelect.appendChild(opt);
            setVersionFromDropdown();
          });
      }
      if (tokenInput) {
        tokenInput.addEventListener('input', function() {
          setSyncControlsEnabled(false);
          if (versionSelect) {
            versionSelect.innerHTML = '';
            var opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Enter token and click Test';
            versionSelect.appendChild(opt);
          }
          if (syncStatus) { syncStatus.textContent = ''; syncStatus.classList.remove('error'); }
        });
      }
      if (testTokenBtn && syncStatus && syncServiceUrl && apiUrl) {
        testTokenBtn.addEventListener('click', function() {
          var token = getToken();
          if (!token) {
            syncStatus.textContent = 'Enter your API token first.';
            syncStatus.classList.add('error');
            return;
          }
          testTokenBtn.disabled = true;
          syncStatus.textContent = 'Testing token…';
          syncStatus.classList.remove('error');
          fetch(syncServiceUrl + '/versions?project_id=' + project + '&api_url=' + encodeURIComponent(apiUrl), {
            headers: { 'Authorization': 'Token ' + token }
          })
            .then(function(r) {
              if (!r.ok) return r.json().then(function(d) { throw new Error(d.detail || r.statusText); });
              return r.json();
            })
            .then(function(versions) {
              setSyncControlsEnabled(true);
              loadVersions(token);
              syncStatus.textContent = 'Token OK. Resolving port/database…';
              var databaseInfoUrl = syncServiceUrl + '/database-info?project_id=' + project + '&api_url=' + encodeURIComponent(apiUrl) + '&port=' + port;
              fetch(databaseInfoUrl, { headers: { 'Authorization': 'Token ' + token } })
                .then(function(r) {
                  if (!r.ok) return r.json().catch(function() { return null; }).then(function(d) {
                    var detail = (d && d.detail) ? d.detail : 'No database entry for this project/port';
                    throw new Error(detail);
                  });
                  return r.json();
                })
                .then(function(d) {
                  hasDatabaseEntry = true;
                  if (d && d.port != null) {
                    port = d.port;
                    appUrl = 'http://' + iframeHost + ':' + port + '/';
                    console.log('[FiftyOne applet] database-info resolved: port=', port, 'database_name=', d.database_name, 'appUrl=', appUrl);
                  }
                  if (d && d.database_name) databaseName = d.database_name;
                  if (d && d.database_uri) databaseUri = d.database_uri;
                  setSyncControlsEnabled(true);
                  syncStatus.textContent = 'Token OK. Sync enabled.';
                  syncStatus.classList.remove('error');
                  setTimeout(function() { syncStatus.textContent = ''; }, 3000);
                })
                .catch(function(err) {
                  hasDatabaseEntry = false;
                  setSyncControlsEnabled(true);
                  syncStatus.textContent = 'Token OK but sync disabled: ' + (err.message || 'no database entry for this project');
                  syncStatus.classList.add('error');
                });
            })
            .catch(function(err) {
              setSyncControlsEnabled(false);
              syncStatus.textContent = 'Token invalid: ' + (err.message || 'Unknown error');
              syncStatus.classList.add('error');
              if (versionSelect) {
                versionSelect.innerHTML = '';
                var opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'Enter token and click Test';
                versionSelect.appendChild(opt);
              }
            })
            .finally(function() { testTokenBtn.disabled = false; });
        });
      }
      if (versionSelect) versionSelect.addEventListener('change', setVersionFromDropdown);
      (function checkEmbeddingService() {
        var el = document.getElementById('embedding-status');
        if (!el) return;
        var base = syncServiceUrl || window.location.origin;
        var url = base.replace(/\/$/, '') + '/vss-embedding';
        fetch(url)
          .then(function(r) {
            if (!r.ok) throw new Error(r.status === 503 ? (r.statusText || 'Service unavailable') : (r.status + ' ' + r.statusText));
            return r.json();
          })
          .then(function(data) {
            var projects = (data && data.projects) ? data.projects : [];
            if (projectName) {
              var inList = projects.indexOf(projectName) !== -1;
              el.textContent = inList
                ? 'Available, project registered'
                : 'Available, but project not registered; cannot compute embeddings or UMAP but you can still edit the localizations';
              if (!inList) el.classList.add('error');
              else el.classList.remove('error');
            } else {
              el.textContent = 'Available (' + (projects.length || 0) + ' project(s)); set project_name tparam (vss_project) to check this project';
              el.classList.remove('error');
            }
          })
          .catch(function(err) {
            el.textContent = err.message || 'Network error';
            el.classList.add('error');
          });
      })();
      if (syncBtn && syncStatus && syncServiceUrl && apiUrl) {
        syncBtn.addEventListener('click', function() {
          var token = getToken();
          if (!token || !tokenVerified) return;
          var v = versionSelect ? versionSelect.value : '';
          syncBtn.disabled = true;
          syncStatus.textContent = 'Syncing…';
          syncStatus.classList.remove('error');
          var params = new URLSearchParams({ project_id: String(project), api_url: apiUrl, token: token, launch_app: 'true', port: port });
          if (v) params.set('version_id', v);
          var fullSyncUrl = syncServiceUrl + '/sync?' + params.toString();
          fetch(fullSyncUrl, { method: 'POST' })
            .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
            .then(function(result) {
              if (!result.ok) {
                syncStatus.textContent = 'Sync failed: ' + (result.data.detail || result.data.message || 'Unknown error');
                syncStatus.classList.add('error');
                syncBtn.disabled = false;
                return;
              }
              var data = result.data;
              if (data.job_id) {
                syncStatus.textContent = 'Sync queued. Waiting for worker…';
                var statusUrl = syncServiceUrl + '/sync/status/' + encodeURIComponent(data.job_id);
                var poll = function() {
                  fetch(statusUrl)
                    .then(function(r) { return r.json(); })
                    .then(function(s) {
                      if (s.status === 'queued' || s.status === 'started' || s.status === 'deferred') {
                        syncStatus.textContent = s.status === 'started' ? 'Sync in progress…' : 'Sync queued…';
                        setTimeout(poll, 2500);
                        return;
                      }
                      if (s.status === 'failed') {
                        syncStatus.textContent = 'Sync failed: ' + (s.error || 'Unknown error');
                        syncStatus.classList.add('error');
                        syncBtn.disabled = false;
                        return;
                      }
                      if (s.status === 'finished' && s.result) {
                        var res = s.result;
                        if (res.status === 'busy') {
                          syncStatus.textContent = res.message || 'Dataset is being updated. Please try again in a few minutes.';
                          syncStatus.classList.add('error');
                          syncBtn.disabled = false;
                          return;
                        }
                        syncStatus.textContent = 'Sync done. Opening FiftyOne…';
                        var openUrl = appUrl;
                        if (res.dataset_name) {
                          openUrl = appUrl.replace(/\/$/, '') + '/datasets/' + encodeURIComponent(res.dataset_name);
                        }
                        setTimeout(function() {
                          window.open(openUrl, '_blank');
                          syncStatus.textContent = res.sample_count != null
                            ? 'Opened with ' + res.sample_count + ' samples.'
                            : 'Opened.';
                        }, 1500);
                        setTimeout(function() { syncStatus.textContent = ''; }, 5000);
                        syncBtn.disabled = false;
                        return;
                      }
                      syncStatus.textContent = 'Sync: ' + (s.status || 'unknown');
                      setTimeout(poll, 2500);
                    })
                    .catch(function(err) {
                      syncStatus.textContent = 'Status check failed: ' + (err.message || 'Network error');
                      syncStatus.classList.add('error');
                      syncBtn.disabled = false;
                    });
                };
                poll();
                return;
              }
              syncStatus.textContent = 'Sync done. Opening FiftyOne…';
              var openUrl = appUrl;
              if (data.dataset_name) {
                openUrl = appUrl.replace(/\/$/, '') + '/datasets/' + encodeURIComponent(data.dataset_name);
              }
              setTimeout(function() {
                window.open(openUrl, '_blank');
                syncStatus.textContent = data.sample_count != null
                  ? 'Opened with ' + data.sample_count + ' samples.'
                  : 'Opened.';
              }, 3500);
              setTimeout(function() { syncStatus.textContent = ''; }, 5000);
              syncBtn.disabled = false;
            })
            .catch(function(err) {
              syncStatus.textContent = 'Sync error: ' + (err.message || 'Network error');
              syncStatus.classList.add('error');
              syncBtn.disabled = false;
            });
        });
      }
      if (syncToTatorBtn && syncStatus && syncServiceUrl && apiUrl) {
        syncToTatorBtn.addEventListener('click', function() {
          var token = getToken();
          if (!token || !tokenVerified) return;
          var v = versionSelect && versionSelect.value ? versionSelect.value : '';
          if (!v) return;
          syncToTatorBtn.disabled = true;
          syncStatus.textContent = 'Pushing to Tator…';
          syncStatus.classList.remove('error');
          var params = new URLSearchParams({
            project_id: String(project),
            version_id: v,
            api_url: apiUrl,
            token: token,
            port: String(port),
            database_name: databaseName
          });
          var fullUrl = syncServiceUrl + '/sync-to-tator?' + params.toString();
          fetch(fullUrl, { method: 'POST' })
            .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
            .then(function(result) {
              if (result.ok) {
                syncStatus.textContent = 'Sync to Tator done.';
                if (result.data.updated != null)
                  syncStatus.textContent += ' Updated: ' + result.data.updated;
                if (result.data.skipped != null)
                  syncStatus.textContent += ', Skipped: ' + result.data.skipped;
                if (result.data.failed != null && result.data.failed > 0)
                  syncStatus.textContent += ', Failed: ' + result.data.failed;
                setTimeout(function() { syncStatus.textContent = ''; }, 5000);
              } else {
                syncStatus.textContent = 'Sync to Tator failed: ' + (result.data.detail || result.data.message || 'Unknown error');
                syncStatus.classList.add('error');
              }
            })
            .catch(function(err) {
              syncStatus.textContent = 'Sync to Tator error: ' + (err.message || 'Network error');
              syncStatus.classList.add('error');
            })
            .finally(function() { setVersionFromDropdown(); });
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


@app_launch.get("/database-info")
async def get_database_info(
    project_id: int = Query(..., description="Tator project ID; resolved to config key (project name) via API"),
    api_url: str = Query(..., description="Tator API base URL (required, same as /versions)"),
    port: int = Query(..., description="Port for this project"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict:
    """
    Resolve database (and port) from DatabaseUriConfig. Requires authentication the same as GET /versions:
    api_url query param and Authorization header (e.g. "Token <token>"). Returns 401 if missing.
    Project name is resolved via get_project(project_id).name. Returns { port, database_name, database_uri }.
    """
    token = _token_from_authorization(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    project_name: str | None = None
    try:
        import tator
        api = tator.get_api(_resolve_api_url(api_url), token)
        proj = api.get_project(project_id)
        project_name = getattr(proj, "name", None) or str(project_id)
    except Exception as e:
        logger.warning(f"get_project({project_id}) failed: {e}")
    if not project_name or not project_name.strip():
        project_name = str(project_id)
    project_name = project_name.strip()
    logger.info(f"request project_id={project_id} -> project_name={project_name!r} port={port}")
    database_entry = get_database_entry(project_id, port, project_name=project_name)
    if database_entry is None:
        raise HTTPException(status_code=404, detail=f"No DatabaseUriConfig entry for project_id={project_id} (project_name={project_name!r}). Set FIFTYONE_DATABASE_URI_CONFIG and add this project.")
    return { "port": database_entry.port, "database_name": database_name_from_uri(database_entry.uri), "database_uri": database_entry.uri }


@app_launch.get("/render", response_class=HTMLResponse)
async def render_launcher() -> HTMLResponse:
    """
    Return Jinja2 template for HostedTemplate. Tator fetches this URL and renders with tparams.
    Required tparams: project (Tator project ID). Optional: iframe_host (host for app URL), base_port (5151), project_name (vss_project for embedding status only).
    The applet uses GET /database-info?project_id=... to resolve port and database from DatabaseUriConfig.
    "Open FiftyOne" opens the app in a new window. For sync: set sync_service_url and api_url; the user enters their Tator API token in the applet and clicks "Test token" to enable Sync from Tator / Sync to Tator.
    Embedding service status: server uses FASTVSS_API_URL; applet calls GET /vss-embedding to show availability and project registration.
    """
    return HTMLResponse(LAUNCHER_TEMPLATE)


def _token_from_authorization(authorization: str | None) -> str | None:
    """Extract raw token from Authorization header (Token <token> or Bearer <token>)."""
    if not authorization or not authorization.strip():
        return None
    s = authorization.strip()
    if s.lower().startswith("token "):
        return s[6:].strip()
    if s.lower().startswith("bearer "):
        return s[7:].strip()
    return s


@app_launch.get("/versions")
async def get_versions(
    project_id: int = Query(..., description="Tator project ID"),
    api_url: str = Query(..., description="Tator REST API base URL"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> list[dict]:
    """
    Return list of Tator versions for the given project. Used by the launcher template
    to populate the version dropdown. Token must be sent via Authorization header
    (e.g. "Token <token>") and is not accepted in the URL.
    """
    # #region agent log
    _dlog("versions_entry", {"project_id": project_id, "api_url": api_url, "has_auth": authorization is not None, "auth_prefix": (authorization or "")[:10]}, hyp="A,C", loc="main.py:versions")
    # #endregion
    import tator

    token = _token_from_authorization(authorization)
    if not token:
        # #region agent log
        _dlog("versions_no_token", {"authorization_raw": repr(authorization)[:80]}, hyp="C", loc="main.py:versions")
        # #endregion
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    host = _resolve_api_url(api_url)
    # #region agent log
    _dlog("versions_calling_tator", {"host": host, "original_api_url": api_url, "project_id": project_id}, hyp="A,D", loc="main.py:versions")
    # #endregion
    try:
        api = tator.get_api(host, token)
        # #region agent log
        _dlog("versions_api_created", {"host": host}, hyp="A,D", loc="main.py:versions")
        # #endregion
        version_list = api.get_version_list(project_id)
        # #region agent log
        _dlog("versions_success", {"count": len(version_list)}, hyp="A,D", loc="main.py:versions")
        # #endregion
    except Exception as e:
        # #region agent log
        _dlog("versions_exception", {"error": str(e)[:200], "type": type(e).__name__}, hyp="A,D", loc="main.py:versions")
        # #endregion
        raise HTTPException(status_code=401, detail=str(e)) from e
    return [{"id": v.id, "name": v.name, "number": v.number} for v in version_list]
 

@app_launch.post("/sync")
async def sync(
    project_id: int = Query(..., description="Tator project ID"),
    version_id: int | None = Query(None),
    api_url: str = Query(..., description="Tator REST API base URL (e.g. https://tator.example.com)"),
    token: str = Query(..., description="Tator API token"),
    port: int = Query(..., description="Port for this project"),
    config_path: str | None = Query(None, description="Path to YAML/JSON config file for dataset build"),
    launch_app: bool = Query(True, description="Launch FiftyOne app after sync"),
) -> dict:
    """
    Trigger sync: fetch Tator media + localizations, build FiftyOne dataset, launch App.
    When Redis is configured (REDIS_HOST or REDIS_URL), enqueues the job and returns
    job_id immediately; poll GET /sync/status/{job_id} for progress. Otherwise runs
    sync inline (can block for a long time on large projects).
    """
    from sync_queue import is_queue_available, enqueue_sync

    project_name: str | None = None
    try:
        import tator
        api = tator.get_api(_resolve_api_url(api_url), token)
        proj = api.get_project(project_id)
        project_name = getattr(proj, "name", None) or str(project_id)
    except Exception as e:
        logger.warning(f"get_project({project_id}) failed: {e}")
    if not project_name or not project_name.strip():
        project_name = str(project_id)
    project_name = project_name.strip() 
    logger.info(f"project_id={project_id} project_name={project_name!r} -> port={port}")
    api_url_clean = _resolve_api_url(api_url)
    database_entry = get_database_entry(project_id, port, project_name=project_name)
    if database_entry is None:
        raise HTTPException(status_code=404, detail=f"No DatabaseUriConfig entry for project_id={project_id} (project_name={project_name!r}). Set FIFTYONE_DATABASE_URI_CONFIG and add this project.")

    if is_queue_available():
        job_id = enqueue_sync(
            project_id=project_id,
            version_id=version_id,
            api_url=api_url_clean,
            token=token,
            port=database_entry.port,
            project_name=project_name,
            database_name=database_name_from_uri(database_entry.uri),
            config_path=config_path,
            launch_app=launch_app,
        )
        return {"job_id": job_id, "status": "queued", "port": port}
    # No Redis: run inline (blocking; may be slow for large projects)
    from sync import sync_project_to_fiftyone
    try:
      result = sync_project_to_fiftyone(
          project_id=project_id,
          version_id=version_id,
          api_url=api_url_clean,
          token=token,
          port=port,
          database_uri=database_entry.uri,
          database_name=database_name_from_uri(database_entry.uri),
          config_path=config_path,
          launch_app=launch_app,
      )
      if result.get("status") == "busy":
          raise HTTPException(
              status_code=409,
              detail=result.get(
                  "message",
                  "This dataset is being updated by another sync. Please try again in a few minutes.",
              ),
          )
      result["port"] = port
      return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app_launch.get("/sync/status/{job_id}")
async def sync_status(job_id: str) -> dict:
    """
    Poll status of an enqueued sync job. Returns status (queued|started|finished|failed),
    and when finished: result (port, dataset_name, sample_count, etc.) or error.
    """
    from sync_queue import is_queue_available, get_job_status

    if not is_queue_available():
        raise HTTPException(status_code=503, detail="Redis not configured; no job queue")
    return get_job_status(job_id)


@app_launch.post("/sync-to-tator")
async def sync_to_tator(
    project_id: int = Query(..., description="Tator project ID"),
    version_id: int = Query(..., description="Tator version ID (required for update)"),
    api_url: str = Query(..., description="Tator REST API base URL"),
    token: str = Query(..., description="Tator API token"),
    port: int = Query(..., description="Port for this project"),
    dataset_name: str | None = Query(None, description="FiftyOne dataset name (default: get_project(project_id).name + '_' + version name)"),
    label_attr: str = Query("Label", description="Tator attribute name for label"),
    score_attr: str | None = Query(None, description="Tator attribute name for score/confidence; omit or empty to skip"),
    debug: bool = Query(False, description="Print per-sample SKIP/UPDATE debug (or set FIFTYONE_SYNC_DEBUG=1)"),
) -> dict:
    """
    Push FiftyOne dataset edits (labels, confidence) back to Tator localizations.
    Requires the dataset to exist (run POST /sync first). Uses elemental_id to match samples.
    """
    from sync import sync_edits_to_tator
    project_name = None
    try:
        import tator
        api = tator.get_api(_resolve_api_url(api_url), token)
        proj = api.get_project(project_id)
        project_name = getattr(proj, "name", None) or str(project_id)
    except Exception as e:
        logger.warning(f"sync_to_tator get_project({project_id}) failed: {e}")
    if not project_name or not str(project_name).strip():
        project_name = str(project_id)
    try:
      result = sync_edits_to_tator(
          project_id=project_id,
          version_id=version_id,
          port=port,
          api_url=_resolve_api_url(api_url),
          token=token,
          dataset_name=dataset_name,
          label_attr=label_attr,
          score_attr=score_attr,
          debug=debug,
          project_name=project_name.strip(),
      )
      return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Include routers after routes are defined
app.include_router(router)
app.include_router(app_launch)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
