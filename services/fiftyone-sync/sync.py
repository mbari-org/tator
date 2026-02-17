"""
Tator to FiftyOne sync: fetch media + localizations, build FiftyOne dataset, launch app.
Phase 2 implementation. Requires fiftyone package and MongoDB.
"""

from __future__ import annotations

import logging
from typing import Any

from port_manager import get_database_name, get_session

logger = logging.getLogger(__name__)


def sync_project_to_fiftyone(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
    database_name: str | None = None,
) -> dict[str, Any]:
    """
    Fetch Tator media and localizations, build FiftyOne dataset, launch App on given port.
    Uses per-project MongoDB database (resolved via database_name override or get_database_name).
    Returns {"status": "ok", "dataset_name": str, "database_name": str} or raises.
    """
    try:
        import fiftyone as fo
    except ImportError:
        logger.warning("fiftyone not installed; sync disabled")
        return {"status": "skipped", "message": "fiftyone not installed"}

    # Resolve MongoDB database name: explicit override > session > default pattern
    sess = get_session(project_id)
    resolved_db = (
        (database_name.strip() if database_name and database_name.strip() else None)
        or (sess.get("database_name") if sess else None)
        or get_database_name(project_id, None)
    )
    fo.config.database_name = resolved_db

    # Stub: full implementation would:
    # 1. Create tator API client with api_url + token
    # 2. Fetch media list, localization list for project/version
    # 3. Resolve media URLs (download_info or frame URLs)
    # 4. Create fo.Dataset, add samples with filepath, add detections from localizations
    # 5. Optionally compute embeddings via embedding_service and attach
    # 6. fo.launch_app(dataset, port=port, remote=True)
    # 7. Store session/process in port_manager for later shutdown

    logger.info(
        "sync_project_to_fiftyone stub: project=%s version=%s port=%s database=%s",
        project_id, version_id, port, resolved_db,
    )
    return {
        "status": "stub",
        "message": "Sync not yet implemented; install fiftyone and tator",
        "database_name": resolved_db,
    }
