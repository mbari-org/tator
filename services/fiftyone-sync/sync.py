"""
Tator to FiftyOne sync: fetch media + localizations, build FiftyOne dataset, launch app.
Phase 2 implementation. Requires fiftyone package and MongoDB.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sync_project_to_fiftyone(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
) -> dict[str, Any]:
    """
    Fetch Tator media and localizations, build FiftyOne dataset, launch App on given port.
    Returns {"status": "ok", "dataset_name": str} or raises.
    """
    try:
        import fiftyone as fo
    except ImportError:
        logger.warning("fiftyone not installed; sync disabled")
        return {"status": "skipped", "message": "fiftyone not installed"}

    # Stub: full implementation would:
    # 1. Create tator API client with api_url + token
    # 2. Fetch media list, localization list for project/version
    # 3. Resolve media URLs (download_info or frame URLs)
    # 4. Create fo.Dataset, add samples with filepath, add detections from localizations
    # 5. Optionally compute embeddings via embedding_service and attach
    # 6. fo.launch_app(dataset, port=port, remote=True)
    # 7. Store session/process in port_manager for later shutdown

    logger.info("sync_project_to_fiftyone stub: project=%s version=%s port=%s", project_id, version_id, port)
    return {"status": "stub", "message": "Sync not yet implemented; install fiftyone and tator"}
