"""
Mutex for FiftyOne sync: only one sync may update a given dataset at a time.
Uses Redis when available (works across RQ workers); otherwise in-process locks.
"""

from __future__ import annotations

import os
import threading

LOCK_KEY_PREFIX = "fiftyone_sync_lock"
DEFAULT_TTL_SECONDS = 7200  # 2h max hold so crashed workers don't lock forever

# In-memory fallback when Redis is not configured
_lock_meta: threading.Lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def _get_redis_url() -> str | None:
    """Return Redis URL if configured (same as sync_queue)."""
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


def get_sync_lock_key(resolved_db: str, project_id: int, version_id: int | None) -> str:
    """Return a unique key for the dataset being synced (same key = same dataset)."""
    v = version_id if version_id is not None else "default"
    return f"{LOCK_KEY_PREFIX}:{resolved_db}:{project_id}:{v}"


def try_acquire_sync_lock(
    lock_key: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """
    Try to acquire the sync lock. Non-blocking.
    Returns True if acquired, False if another sync holds it.
    """
    redis_url = _get_redis_url()
    if redis_url:
        try:
            from redis import Redis
            from redis.backoff import ExponentialBackoff
            from redis.retry import Retry
            from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

            conn = Redis.from_url(
                redis_url,
                retry=Retry(ExponentialBackoff(), 3),
                retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
                health_check_interval=30,
            )
            acquired = conn.set(lock_key, "1", nx=True, ex=ttl_seconds)
            conn.close()
            return bool(acquired)
        except Exception:
            # Fall back to in-memory lock if Redis fails
            pass

    # In-memory: one lock per key
    with _lock_meta:
        if lock_key not in _locks:
            _locks[lock_key] = threading.Lock()
    return _locks[lock_key].acquire(blocking=False)


def release_sync_lock(lock_key: str) -> None:
    """Release the sync lock. Must be called by the same process that acquired it."""
    redis_url = _get_redis_url()
    if redis_url:
        try:
            from redis import Redis
            from redis.backoff import ExponentialBackoff
            from redis.retry import Retry
            from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError

            conn = Redis.from_url(
                redis_url,
                retry=Retry(ExponentialBackoff(), 3),
                retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
                health_check_interval=30,
            )
            conn.delete(lock_key)
            conn.close()
            return
        except Exception:
            pass

    if lock_key in _locks:
        try:
            _locks[lock_key].release()
        except RuntimeError:
            pass  # already released
