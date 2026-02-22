"""
Embedding service: delegates to Fast-VSS API for batch image embeddings.
Fast-VSS: POST /embeddings/{project}/ with files -> job_id -> poll GET /predict/job/{job_id}/{project}
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_url = os.environ.get("FASTVSS_API_URL")
FASTVSS_BASE_URL = _url.strip().rstrip("/") if _url else None

# Our job_id -> (fastvss_job_id, project)
_job_map: dict[str, tuple[str, str]] = {}
_queue_results: dict[str, dict[str, Any]] = {}
_queue_lock = asyncio.Lock()


def init_disk_cache() -> None:
    """No-op; Fast-VSS handles caching."""
    pass


def is_embedding_service_available() -> bool:
    """
    Return True if the Fast-VSS embedding service is reachable (GET /projects).
    Used by sync to skip embeddings when service is unavailable; same notion as GET /embedding-projects.
    """
    if not FASTVSS_BASE_URL:
        return False
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{FASTVSS_BASE_URL}/projects")
            resp.raise_for_status()
            return True
    except Exception:
        return False


async def queue_embedding_job(
    image_bytes_list: list[bytes],
    cache_keys: list[str | None] | None = None,
    project: str = "default",
) -> str:
    """
    Forward batch to Fast-VSS POST /embeddings/{project}/, get job_id, return our UUID.
    Poll GET /embed/{uuid} for results (we poll Fast-VSS /predict/job/{job_id}/{project}).
    """
    if not FASTVSS_BASE_URL:
        raise ValueError("FASTVSS_API_URL environment variable is not set")
    job_id = str(uuid.uuid4())
    async with _queue_lock:
        _queue_results[job_id] = {"status": "pending", "embeddings": None, "error": None}

    async def run_job() -> None:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = [("files", (f"img_{i}.jpg", data)) for i, data in enumerate(image_bytes_list)]
                url = f"{FASTVSS_BASE_URL}/embeddings/{project}/"
                resp = await client.post(url, files=files)
                resp.raise_for_status()
                data = resp.json()

            fastvss_job_id = data.get("job_id") or data.get("job-id")
            if fastvss_job_id:
                async with _queue_lock:
                    _job_map[job_id] = (str(fastvss_job_id), project)
                _queue_results[job_id] = {
                    "status": "pending",
                    "embeddings": None,
                    "error": None,
                    "fastvss_job_id": fastvss_job_id,
                }
            else:
                # Sync response with embeddings
                emb = data.get("embeddings") or data
                if isinstance(emb, list):
                    async with _queue_lock:
                        _queue_results[job_id] = {"status": "completed", "embeddings": emb, "error": None}
                else:
                    async with _queue_lock:
                        _queue_results[job_id] = {"status": "completed", "embeddings": [emb], "error": None}
        except Exception as e:
            logger.exception("Fast-VSS embedding request failed")
            async with _queue_lock:
                _queue_results[job_id] = {"status": "failed", "embeddings": None, "error": str(e)}

    asyncio.create_task(run_job())
    return job_id


async def _poll_fastvss_result(job_id: str) -> None:
    """Poll Fast-VSS /predict/job/{job_id}/{project} and update _queue_results when ready."""
    if not FASTVSS_BASE_URL:
        return
    mapped = _job_map.get(job_id)
    if not mapped:
        return
    fastvss_job_id, project = mapped
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{FASTVSS_BASE_URL}/predict/job/{fastvss_job_id}/{project}"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        emb = data.get("embeddings") or data.get("result") or data
        async with _queue_lock:
            _queue_results[job_id] = {"status": "completed", "embeddings": emb, "error": None}
            _job_map.pop(job_id, None)
    except Exception as e:
        if "404" in str(e) or "not ready" in str(e).lower():
            return  # Still pending
        logger.warning("Poll Fast-VSS failed: %s", e)
        async with _queue_lock:
            _queue_results[job_id] = {"status": "failed", "embeddings": None, "error": str(e)}
            _job_map.pop(job_id, None)


def get_embedding_result(job_id: str) -> dict[str, Any] | None:
    """Get cached result for a queued embedding job by UUID."""
    return _queue_results.get(job_id)


async def get_or_poll_embedding_result(job_id: str) -> dict[str, Any] | None:
    """
    Get result; if pending with fastvss_job_id, poll Fast-VSS and return when ready.
    Use from async context (e.g. GET /embed/{job_id} endpoint).
    """
    result = _queue_results.get(job_id)
    if result is None:
        return None
    if result.get("status") == "pending" and result.get("fastvss_job_id"):
        await _poll_fastvss_result(job_id)
        result = _queue_results.get(job_id)
    return result
