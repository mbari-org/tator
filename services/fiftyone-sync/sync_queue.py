"""
Redis queue for FiftyOne sync jobs. When REDIS_HOST (or REDIS_URL) is set,
POST /sync enqueues a job and returns immediately; a separate RQ worker runs
the long-running sync. Compatible with the Tator compose stack (redis service).
"""

from __future__ import annotations

import os
from typing import Any

QUEUE_NAME = "fiftyone_sync"


def _get_redis_url() -> str | None:
    """Return Redis URL if queue is configured."""
    url = os.environ.get("REDIS_URL", "").strip()
    if url:
        return url
    host = os.environ.get("REDIS_HOST", "").strip()
    if not host:
        return None
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD", "")
    use_ssl = os.environ.get("REDIS_USE_SSL", "false").lower() == "true"
    scheme = "rediss" if use_ssl else "redis"
    if password:
        return f"{scheme}://:{password}@{host}:{port}/0"
    return f"{scheme}://{host}:{port}/0"


def is_queue_available() -> bool:
    """True if Redis is configured so we can enqueue sync jobs."""
    return _get_redis_url() is not None


def get_connection():
    """Redis connection for RQ. Use when queue is available."""
    from redis import Redis
    from redis.backoff import ExponentialBackoff
    from redis.retry import Retry
    from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

    url = _get_redis_url()
    if not url:
        raise RuntimeError("Redis not configured (set REDIS_HOST or REDIS_URL)")
    retry = Retry(ExponentialBackoff(), 3)
    return Redis.from_url(
        url,
        retry=retry,
        retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
        health_check_interval=30,
    )


def enqueue_sync(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
    database_name: str | None = None,
    config_path: str | None = None,
    launch_app: bool = True,
) -> str:
    """
    Enqueue a sync job. Returns RQ job id.
    Caller must check is_queue_available() first.
    """
    from rq import Queue

    conn = get_connection()
    queue = Queue(QUEUE_NAME, connection=conn)
    job = queue.enqueue(
        "sync.run_sync_job",
        project_id=project_id,
        version_id=version_id,
        api_url=api_url,
        token=token,
        port=port,
        database_name=database_name,
        config_path=config_path,
        launch_app=launch_app,
        job_timeout=3600 * 24,  # 24h for large projects
        result_ttl=3600 * 24,
        failure_ttl=3600,
    )
    return job.id


def get_job_status(job_id: str) -> dict[str, Any]:
    """
    Return status and result for a sync job.
    Keys: status (queued|started|finished|failed|deferred|canceled),
          result (dict when finished), error (str when failed).
    """
    from rq.job import Job

    conn = get_connection()
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception as e:
        return {"status": "unknown", "error": str(e)}
    status = job.get_status(refresh=True)
    out = {"status": status}
    if status == "finished" and job.result is not None:
        out["result"] = job.result
    if status == "failed" and job.exc_info:
        out["error"] = job.exc_info
    return out
