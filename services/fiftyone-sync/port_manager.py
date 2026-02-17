"""
Port manager: assigns unique ports per Tator project for FiftyOne App instances.
Tracks project_id -> port and manages spawning/stopping FiftyOne sessions.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

BASE_PORT = 5151
MAX_PROJECTS = 1000  # Ports 5151 to 5150 + MAX_PROJECTS

# In-memory: project_id -> {"port": int, "process": subprocess.Popen | None, ...}
_sessions: dict[int, dict[str, Any]] = {}


def get_port_for_project(project_id: int) -> int:
    """Return the port assigned to a project. Port = BASE_PORT + project_id."""
    port = BASE_PORT + (project_id % MAX_PROJECTS)
    return port


def ensure_session(project_id: int, dataset_name: str | None = None) -> int:
    """
    Ensure a FiftyOne session exists for the project. Returns the port.
    For now, we only allocate the port; actual fo.launch_app() is done by sync worker.
    """
    port = get_port_for_project(project_id)
    if project_id not in _sessions:
        _sessions[project_id] = {
            "port": port,
            "process": None,
            "dataset_name": dataset_name or f"tator_project_{project_id}",
        }
        logger.info("Allocated port %s for project %s", port, project_id)
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
