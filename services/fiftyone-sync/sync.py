"""
Tator to FiftyOne sync: fetch media + localizations, build FiftyOne dataset, launch app.
Phase 2 implementation. Requires fiftyone package and MongoDB.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from port_manager import get_database_name, get_session

logger = logging.getLogger(__name__)

CHUNK_SIZE = 200
LOCALIZATION_BATCH_SIZE = 5000


def _project_tmp_dir(project_id: int) -> str:
    """Return a project-isolated directory under /tmp for synced media."""
    path = os.path.join("/tmp", f"fiftyone_sync_project_{project_id}")
    os.makedirs(path, exist_ok=True)
    return path


def fetch_project_media_ids(
    api_url: str,
    token: str,
    project_id: int,
    media_ids_filter: list[int] | None = None,
) -> list[int]:
    """
    Fetch all media in the project. Returns list of media ids.
    If media_ids_filter is set, only those media are returned (and must exist in the project).
    """
    print(f"[sync] fetch_project_media_ids: project_id={project_id} filter={media_ids_filter}", flush=True)
    try:
        import tator
    except ImportError:
        logger.warning("tator not installed; cannot fetch media")
        print("[sync] tator not installed", flush=True)
        return []

    host = api_url.rstrip("/")
    api = tator.get_api(host, token)
    if media_ids_filter:
        media_list = api.get_media_list(project_id, media_id=media_ids_filter)
    else:
        media_list = api.get_media_list(project_id)
    media_ids = [m.id for m in media_list]
    print(f"[sync] fetch_project_media_ids: got {len(media_ids)} ids", flush=True)
    logger.info("Project %s media count: %s; ids: %s", project_id, len(media_ids), media_ids)
    return media_ids


def get_media_chunked(api: Any, project_id: int, media_ids: list[int]) -> list[Any]:
    """
    Get media objects in chunks of CHUNK_SIZE. Uses get_media_list_by_id for reliable Media objects.
    Filters out non-Media responses (API quirk). Returns list of tator.models.Media.
    """
    print(f"[sync] get_media_chunked: project_id={project_id} num_ids={len(media_ids)} chunk_size={CHUNK_SIZE}", flush=True)
    try:
        import tator
    except ImportError:
        return []
    if not media_ids:
        print("[sync] get_media_chunked: no ids, returning []", flush=True)
        return []
    all_media = []
    for start in range(0, len(media_ids), CHUNK_SIZE):
        chunk_ids = media_ids[start : start + CHUNK_SIZE]
        media = api.get_media_list_by_id(project_id, {"ids": chunk_ids})
        new_media = [m for m in media if isinstance(m, tator.models.Media)]
        all_media += new_media
        print(f"[sync] get_media_chunked: start={start} chunk_len={len(new_media)} total_media={len(all_media)}", flush=True)
    print(f"[sync] get_media_chunked: done, {len(all_media)} Media objects", flush=True)
    logger.info("get_media_chunked: %s ids -> %s Media objects", len(media_ids), len(all_media))
    return all_media


# Video extensions: skip download (not supported); downloads come directly from Tator for images only.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")


def _is_video_name(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)


def save_media_to_tmp(api: Any, project_id: int, media_objects: list[Any]) -> str:
    """
    Download each media to a project-isolated /tmp folder, always under a 'download' subdir.
    Videos are skipped (download not supported). Existing non-empty files are skipped.
    Retries each download up to 3 times. Returns the download directory path.
    """
    try:
        import tator
    except ImportError:
        logger.warning("tator not installed; cannot save media")
        return ""
    base_dir = _project_tmp_dir(project_id)
    out_dir = os.path.join(base_dir, "download")
    os.makedirs(out_dir, exist_ok=True)
    valid = [m for m in media_objects if isinstance(m, tator.models.Media)]
    total = len(valid)
    logger.info("Saving %s media files to %s", total, out_dir)
    print(f"[sync] Saving {total} media files to {out_dir}", flush=True)
    for idx, m in enumerate(valid, 1):
        safe_name = f"{m.id}_{m.name}"
        out_path = os.path.join(out_dir, safe_name)
        if _is_video_name(m.name):
            print(f"[sync] Skipping video (not supported): {safe_name}", flush=True)
            logger.info("Skipping video %s", m.id)
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[sync] Skipping existing: {safe_name}", flush=True)
            logger.debug("Skipping existing %s", out_path)
            continue
        print(f"[sync] Downloading {m.name} to {out_path}", flush=True)
        num_tries = 0
        success = False
        while num_tries < 3 and not success:
            try:
                for _ in tator.util.download_media(api, m, out_path):
                    pass
                success = True
                logger.info("Saved %s -> %s", m.id, out_path)
                print(f"[sync] Saved media {m.id} -> {out_path}", flush=True)
            except Exception as e:
                logger.warning("Download attempt %s/3 failed for media %s: %s", num_tries + 1, m.id, e)
                print(f"[sync] Attempt {num_tries + 1}/3 failed for {m.id}: {e}", flush=True)
                num_tries += 1
        if not success:
            print(f"[sync] Could not download {m.name} after 3 tries", flush=True)
    logger.info("Saved media files to %s", out_dir)
    print(f"[sync] Done. Files in {out_dir}", flush=True)
    return out_dir


def fetch_and_save_localizations(
    api: Any,
    project_id: int,
    version_id: int | None = None,
) -> str:
    """
    Fetch all localizations in batches (after/stop pagination, Tator style) and write to a JSONL file.
    Returns path to the file (e.g. .../localizations.jsonl). Uses LOCALIZATION_BATCH_SIZE per batch.
    """
    try:
        import tator
    except ImportError:
        logger.warning("tator not installed; cannot fetch localizations")
        print("[sync] tator not installed, localizations skipped", flush=True)
        return ""
    base_dir = _project_tmp_dir(project_id)
    out_path = os.path.join(base_dir, "localizations.jsonl")
    print(f"[sync] Localizations JSONL will be saved to: {out_path}", flush=True)
    batch_size = LOCALIZATION_BATCH_SIZE
    # Get total count (may require type= if API needs it; try without first)
    try:
        count_kwargs = {}
        if version_id is not None:
            count_kwargs["version"] = [version_id]
        loc_count = api.get_localization_count(project_id, **count_kwargs)
        print(f"[sync] get_localization_count(project_id={project_id}) = {loc_count}", flush=True)
    except Exception as e:
        loc_count = None
        print(f"[sync] get_localization_count failed (will still try list): {e}", flush=True)
    total = 0
    after_id = None
    with open(out_path, "w") as f:
        while True:
            kwargs = {"stop": batch_size}
            if version_id is not None:
                kwargs["version"] = [version_id]
            if after_id is not None:
                kwargs["after"] = after_id
            print(f"[sync] get_localization_list(project_id={project_id}, {kwargs})", flush=True)
            try:
                batch = api.get_localization_list(project_id, **kwargs)
            except Exception as e:
                print(f"[sync] get_localization_list failed: {e}", flush=True)
                logger.exception("get_localization_list failed")
                break
            if not batch:
                print(f"[sync] Localizations batch empty (after={after_id}), done", flush=True)
                break
            for loc in batch:
                try:
                    obj = loc.to_dict() if hasattr(loc, "to_dict") else loc
                    f.write(json.dumps(obj) + "\n")
                except Exception as e:
                    logger.warning("Skip localization serialization: %s", e)
                    continue
            total += len(batch)
            after_id = batch[-1].id if batch else None
            print(f"[sync] Localizations batch: count={len(batch)} total_so_far={total} last_id={after_id}", flush=True)
            if len(batch) < batch_size:
                break
    logger.info("Fetched %s localizations -> %s", total, out_path)
    print(f"[sync] Localizations JSONL saved to: {out_path} ({total} rows)", flush=True)
    return out_path


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
    print(f"[sync] sync_project_to_fiftyone CALLED: project_id={project_id} version_id={version_id} api_url={api_url} port={port}", flush=True)
    try:
        import fiftyone as fo
    except ImportError:
        logger.warning("fiftyone not installed; sync disabled")
        print("[sync] fiftyone not installed, skipping", flush=True)
        return {"status": "skipped", "message": "fiftyone not installed"}

    # Resolve MongoDB database name: explicit override > session > default pattern
    sess = get_session(project_id)
    resolved_db = (
        (database_name.strip() if database_name and database_name.strip() else None)
        or (sess.get("database_name") if sess else None)
        or get_database_name(project_id, None)
    )
    fo.config.database_name = resolved_db
    print(f"[sync] database_name={resolved_db}", flush=True)

    # Fetch media IDs, get media objects in chunks, download to project-isolated /tmp; then fetch localizations in batches
    media_ids_list: list[int] = []
    tmp_dir = ""
    localizations_path = ""
    try:
        import tator
        host = api_url.rstrip("/")
        api = tator.get_api(host, token)
        print(f"[sync] Fetching media IDs...", flush=True)
        media_ids_list = fetch_project_media_ids(api_url, token, project_id)
        if not media_ids_list:
            print(f"[sync] No media IDs for project {project_id}", flush=True)
            logger.warning("No media IDs found for project %s; skipping download", project_id)
        else:
            print("media_ids:", media_ids_list, flush=True)
            print(f"[sync] Getting media objects in chunks...", flush=True)
            all_media = get_media_chunked(api, project_id, media_ids_list)
            if not all_media:
                print(f"[sync] No Media objects for {len(media_ids_list)} ids", flush=True)
                logger.warning("No Media objects returned for %s ids; skipping download", len(media_ids_list))
            else:
                print(f"[sync] Saving {len(all_media)} media to tmp...", flush=True)
                tmp_dir = save_media_to_tmp(api, project_id, all_media)
                if tmp_dir:
                    print("saved_media_dir:", tmp_dir, flush=True)
        import pdb;pdb.set_trace()
        print(f"[sync] Fetching localizations...", flush=True)
        localizations_path = fetch_and_save_localizations(api, project_id, version_id=version_id)
        if localizations_path:
            print(f"[sync] saved_localizations_path (JSONL): {localizations_path}", flush=True)
    except Exception as e:
        logger.exception("fetch/save media or localizations failed: %s", e)
        print(f"[sync] Error: {e}", flush=True)

    # Full implementation would:
    # 1. Use version_id and media list for localizations
    # 2. Resolve media URLs (download_info or frame URLs)
    # 3. Create fo.Dataset, add samples with filepath, add detections from localizations
    # 4. Optionally compute embeddings via embedding_service and attach
    # 5. fo.launch_app(dataset, port=port, remote=True)
    # 6. Store session/process in port_manager for later shutdown

    print(f"[sync] sync_project_to_fiftyone done: saved_media_dir={tmp_dir or None} saved_localizations_path={localizations_path or None}", flush=True)
    logger.info(
        "sync_project_to_fiftyone: project=%s port=%s database=%s",
        project_id, port, resolved_db,
    )
    return {
        "status": "stub",
        "message": "Sync not yet implemented; install fiftyone and tator",
        "database_name": resolved_db,
        "saved_media_dir": tmp_dir or None,
        "saved_localizations_path": localizations_path or None,
    }


def main() -> None:
    """Read env (HOST, TOKEN, PROJECT_ID, optional MEDIA_IDS) and print media IDs."""
    host = os.getenv("HOST", "").rstrip("/")
    token = os.getenv("TOKEN")
    project_id_str = os.getenv("PROJECT_ID")
    media_ids_str = os.getenv("MEDIA_IDS", "").strip()
    print(f"[sync] main: HOST={'<set>' if host else '<unset>'} PROJECT_ID={project_id_str or '<unset>'} MEDIA_IDS={'<set>' if media_ids_str else '<unset>'}", flush=True)

    if not host or not token or not project_id_str:
        print("Set HOST, TOKEN, and PROJECT_ID environment variables.")
        return
    project_id = int(project_id_str)
    media_ids_filter: list[int] | None = None
    if media_ids_str:
        media_ids_filter = [int(id_.strip()) for id_ in media_ids_str.split(",") if id_.strip()]

    media_ids = fetch_project_media_ids(host, token, project_id, media_ids_filter=media_ids_filter)
    print("media_ids:", media_ids)
    if media_ids:
        import tator
        api = tator.get_api(host, token)
        all_media = get_media_chunked(api, project_id, media_ids)
        if all_media:
            saved_dir = save_media_to_tmp(api, project_id, all_media)
            print("saved_media_dir:", saved_dir)
        else:
            print("No Media objects returned; download skipped.")


if __name__ == "__main__":
    main()
