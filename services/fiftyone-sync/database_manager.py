"""
Database manager: maps (project_id, port) to a database (URI + name) and maintains
one unique session per (project_id, port). 
Uses optional YAML config (FIFTYONE_DATABASE_URI_CONFIG) keyed by project name
for per-project, per-port databases.
"""

from __future__ import annotations

import os
import logging
import json
from pathlib import Path
from typing import Any
from database_uri_config import DatabaseEntry, DatabaseUriConfig, database_name_from_uri

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Print to console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
MAX_PROJECTS = 20

# Session key: (project_id, port). Value: session dict (port, process, database_name, database_uri, dataset_name).
_sessions: dict[tuple[int, int], dict[str, Any]] = {}

# Cached config lookup: key is (project_name: str, port: int), value is DatabaseEntry.
_config: dict[tuple[str, int], DatabaseEntry] | None = None

# Full YAML config for get_vss_project (project name -> ProjectConfig).
_yaml_config: DatabaseUriConfig | None = None

# Optional registry: project_id -> project_name (set by run_sync_job so workers can resolve without API).
_project_id_to_name: dict[int, str] = {}


def _load_config() -> dict[tuple[str, int], DatabaseEntry] | None:
    global _config, _yaml_config
    if _config is not None:
        return _config
    path = os.environ.get("FIFTYONE_DATABASE_URI_CONFIG", "").strip()
    if not path or not Path(path).is_file():
        _config = None
        _yaml_config = None
        return None
    try:
        _config = {}
        _yaml_config = DatabaseUriConfig.from_yaml_path(path)
        logger.info(f"loaded config from {path}: projects={list(_yaml_config.projects.keys())}")
        for project_name, project_config in _yaml_config.projects.items():
            for db in project_config.databases:
                key = (project_name, db.port)
                _config[key] = db
        # Log config with string keys (JSON requires str keys, not tuples)
        log_config = {
            f"{k[0]!r}:{k[1]}": {"uri": v.uri, "port": v.port}
            for k, v in _config.items()
        }
        logger.info(f"config: {json.dumps(log_config, indent=2)}")
        return _config
    except Exception as e:
        logger.warning(f"Failed to load FIFTYONE_DATABASE_URI_CONFIG from {path}: {e}")
        _config = None
        _yaml_config = None
        return None

def get_vss_project(project_name: str, port: int) -> str | None:
    """Return vss_project for (project_name, port) from DatabaseUriConfig, or None."""
    _load_config()
    if not _yaml_config or not project_name or not project_name.strip():
        return None
    proj = _yaml_config.projects.get(project_name.strip())
    return getattr(proj, "vss_project", None) if proj else None

def register_project_id_name(project_id: int, project_name: str) -> None:
    """Register project_id -> project_name so get_database_entry can resolve when project_name is not passed."""
    if project_id and project_name and str(project_name).strip():
        _project_id_to_name[project_id] = str(project_name).strip()


def get_database_entry(
    project_id: int, port: int, project_name: str | None = None
) -> DatabaseEntry | None:
    """
    Resolve FiftyOne MongoDB database entry for this project and port.
    Config is keyed by (project_name, port). Uses project_name if provided,
    else _project_id_to_name[project_id], else str(project_id).
    """
    cfg = _load_config()
    if not cfg or not project_id:
        return None
    name = (project_name or _project_id_to_name.get(project_id) or "").strip() or str(project_id)
    key = (name, port)
    return cfg.get(key)


def get_database_uri(
    project_id: int, port: int, project_name: str | None = None
) -> str | None:
    """Return MongoDB URI for (project, port). Uses config or FIFTYONE_DATABASE_URI env."""
    entry = get_database_entry(project_id, port, project_name=project_name)
    if entry:
        return entry.uri
    return os.environ.get("FIFTYONE_DATABASE_URI", "").strip() or None


def get_database_name(
    project_id: int, port: int, project_name: str | None = None
) -> str:
    """Return MongoDB database name for (project, port). Uses config or default naming."""
    entry = get_database_entry(project_id, port, project_name=project_name)
    if entry:
        return database_name_from_uri(entry.uri)
    default = os.environ.get("FIFTYONE_DATABASE_DEFAULT", "fiftyone_project")
    return f"{default}_{project_id}"


def get_port_for_project(
    project_id: int, project_name: str | None = None
) -> int:
    """Return first configured port for this project, or 5151 + (project_id - 1)."""
    cfg = _load_config()
    name = (project_name or _project_id_to_name.get(project_id) or "").strip() or str(project_id)
    if cfg:
        for (pname, port), _ in cfg.items():
            if pname == name:
                return port
    return 5151 + (project_id - 1)


def ensure_session(
    project_id: int,
    port: int,
    dataset_name: str | None = None,
    project_name: str | None = None,
) -> int | None:
    """
    Ensure a FiftyOne session exists for (project_id, port). Returns the port.
    """
    db = get_database_entry(project_id, port, project_name=project_name)
    if db is None:
        return None
    if (project_id, port) not in _sessions:
        _sessions[(project_id, port)] = {
            "port": port,
            "process": None,
            "dataset_name": dataset_name or f"tator_project_{project_id}"
        }
    return port


def get_session(project_id: int, port: int | None = None) -> dict[str, Any] | None:
    """Get session info for (project_id, port)."""
    return _sessions.get((project_id, port)) if _sessions.get((project_id, port)) else None


def release_session(project_id: int, port: int) -> None:
    """Release/stop the FiftyOne session for (project_id, port)."""
    key = (project_id, port)
    if key in _sessions:
        sess = _sessions[key]
        proc = sess.get("process")
        if proc and getattr(proc, "poll", None) and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        del _sessions[key]
        logger.info(f"Released session for project_id={project_id} port={port}")
