#!/usr/bin/env python3
"""
RQ worker for FiftyOne sync jobs. Run when Redis is used for background sync:
  python sync_worker.py
  # or: rq worker --url redis://localhost:6379/0 fiftyone_sync

Use with Tator compose: set REDIS_HOST=redis and run this in a separate container
or on the same host that can reach Redis.
"""

from __future__ import annotations

import os
import sys
import fiftyone as fo

# Ensure project root is on path so "sync" and "sync_queue" resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rq import Queue, Worker

from sync_queue import QUEUE_NAME, _get_redis_url, get_connection


def main() -> None:
    url = _get_redis_url()
    if not url:
        print("Set REDIS_HOST or REDIS_URL to run the sync worker.", file=sys.stderr)
        sys.exit(1)
    # Launch FiftyOne app with default port 5151
    fo.launch_app(address='0.0.0.0', port=5151)
    conn = get_connection()
    queue = Queue(QUEUE_NAME, connection=conn)
    worker = Worker([queue], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
