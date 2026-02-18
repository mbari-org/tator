"""
Port manager: assigns unique ports per Tator project for FiftyOne App instances.
Uses a single MongoDB with per-project databases. Blocks PORTS_PER_PROJECT ports per project
(up to 10 users per project). Override via FIFTYONE_DATABASE_URI.
"""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

BASE_PORT = 5151
PORTS_PER_PROJECT = 10  # Block 10 ports per project for up to 10 users
MAX_PROJECTS = 1000

# In-memory: project_id -> {"port": int, "process": subprocess.Popen | None, "database_name": str, ...}
_sessions: dict[int, dict[str, Any]] = {}


def get_database_uri(_project_id: int | None = None) -> str:
    """
    Resolve FiftyOne MongoDB connection URI. Single MongoDB for all projects.
    Override via FIFTYONE_DATABASE_URI. Per-project isolation is by database_name.
    """
    uri = os.environ.get("FIFTYONE_DATABASE_URI", "").strip()
    return uri or "mongodb://localhost:27017"


def get_database_name(project_id: int, override: str | None = None) -> str:
    """
    Resolve FiftyOne MongoDB database name: per-request override > FIFTYONE_DATABASE_NAME env > default pattern.
    Empty string override is treated as no override.
    """
    if override and override.strip():
        return override.strip()
    env_name = os.environ.get("FIFTYONE_DATABASE_NAME", "").strip()
    if env_name:
        return env_name
    default_prefix = os.environ.get("FIFTYONE_DATABASE_DEFAULT", "fiftyone_project")
    return f"{default_prefix}_{project_id}"


def get_port_for_project(project_id: int) -> int:
    """
    Return the base port for a project. Project N gets ports [base, base + PORTS_PER_PROJECT - 1].
    Project 1: 5151-5160, project 2: 5161-5170, etc. First port used by default.
    """
    block = (project_id - 1) % MAX_PROJECTS
    return BASE_PORT + block * PORTS_PER_PROJECT


def ensure_session(
    project_id: int,
    dataset_name: str | None = None,
    database_name: str | None = None,
) -> int:
    """
    Ensure a FiftyOne session exists for the project. Returns the base port (first in block).
    fo.launch_app() is done by sync worker. database_name is resolved via get_database_name.
    """
    port = get_port_for_project(project_id)
    if project_id not in _sessions:
        resolved_db = get_database_name(project_id, database_name)
        _sessions[project_id] = {
            "port": port,
            "process": None,
            "dataset_name": dataset_name or f"tator_project_{project_id}",
            "database_name": resolved_db,
        }
        logger.info("Allocated port %s for project %s (database=%s)", port, project_id, resolved_db)
    return port


def get_session(project_id: int) -> dict[str, Any] | None:
    """Get session info for a project."""
    return _sessions.get(project_id)


def release_session(project_id: int) -> None:
    """Release/stop the FiftyOne session for a project."""
    if project_id in _sessions:
        sess = _sessions[project_id]
        proc = sess.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        del _sessions[project_id]
        logger.info("Released session for project %s", project_id)
