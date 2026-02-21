"""
Database manager: maps (project_id, port) to a database (URI + name) and maintains
one unique session per (project_id, port). Uses FIFTYONE_DATABASE_URI / FIFTYONE_DATABASE_NAME
when no config is set; optional YAML config (FIFTYONE_DATABASE_URI_CONFIG) keyed by project name
for per-project, per-port databases.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from database_uri_config import DatabaseEntry, DatabaseUriConfig

logger = logging.getLogger(__name__)
 
MAX_PROJECTS = 20

# Session key: (project_id, port). Value: session dict (port, process, database_name, database_uri, dataset_name).
_sessions: dict[tuple[int, int], dict[str, Any]] = {}

# Cached config lookup: key is (project_name, port), value is DatabaseEntry.
_config: dict[tuple[int, int], DatabaseEntry] | None = None
# Full YAML config for get_vss_project (project name -> ProjectConfig).
_yaml_config: DatabaseUriConfig | None = None
# Runtime mapping so sync (which only has project_id) can resolve via get_database_uri(project_id, port).
_project_id_to_name: dict[int, str] = {}

def _load_config() -> dict[tuple[int, int], DatabaseEntry] | None:
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
        logger.info(
            "[database_manager] loaded config from %s: projects=%s",
            path,
            list(_yaml_config.projects.keys()),
        )
        for project_name, project_config in _yaml_config.projects.items():
            for db in project_config.databases:
                key = (project_name, db.port)
                _config[key] = db
        return _config
    except Exception as e:
        logger.warning("Failed to load FIFTYONE_DATABASE_URI_CONFIG from %s: %s", path, e)
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

def get_database_entry(project_id: int, port: int) -> DatabaseEntry | None:
    """
    Resolve FiftyOne MongoDB database entry for (project_name, port).
    Config is keyed by project name (YAML), so lookup uses (project_name, port).
    """
    cfg = _load_config()
    if not cfg or not project_id or not project_id.strip():
        return None
    key = (project_id, port)
    return _config.get(key) if _config else None


def ensure_session(
    project_id: int,
    port: int,
    dataset_name: str | None = None
) -> int | None:
    """
    Ensure a FiftyOne session exists for (project_id, port). Returns the port.
    """
    db = get_database_entry(project_id, port)
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
        logger.info("Released session for project_id=%s port=%s", project_id, port)
