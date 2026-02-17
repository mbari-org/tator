"""
Tator to FiftyOne sync: fetch media + localizations, build FiftyOne dataset, launch app.
Phase 2 implementation. Requires fiftyone package and MongoDB.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

from port_manager import get_database_name, get_session

logger = logging.getLogger(__name__)

CHUNK_SIZE = 200
LOCALIZATION_BATCH_SIZE = 5000


def _json_serial(obj: Any) -> Any:
    """Convert datetime/date to epoch seconds (float) for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.timestamp()
    if isinstance(obj, date):
        return datetime.combine(obj, datetime.min.time()).timestamp()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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
                    f.write(json.dumps(obj, default=_json_serial) + "\n")
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


def _crop_one(
    size: int,
    image_path: Path,
    loc: dict,
    out_path: Path,
) -> bool:
    """
    Crop one localization from an image: pad shorter side to square (longest side), resize to size x size.
    Tator box: x, y, width, height as normalized 0-1.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; cannot crop")
        return False
    try:
        x = float(loc.get("x", 0))
        y = float(loc.get("y", 0))
        w = float(loc.get("width", 0))
        h = float(loc.get("height", 0))
        if w <= 0 or h <= 0:
            return False
        img = Image.open(image_path).convert("RGB")
        image_width, image_height = img.size
        x1 = int(image_width * x)
        y1 = int(image_height * y)
        x2 = int(image_width * (x + w))
        y2 = int(image_height * (y + h))
        # Clamp to image bounds
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(1, min(x2, image_width))
        y2 = max(1, min(y2, image_height))
        width = x2 - x1
        height = y2 - y1
        shorter_side = min(height, width)
        longer_side = max(height, width)
        delta = abs(longer_side - shorter_side)
        padding = delta // 2
        if width == shorter_side:
            x1 = max(0, x1 - padding)
            x2 = min(image_width, x2 + padding)
        else:
            y1 = max(0, y1 - padding)
            y2 = min(image_height, y2 + padding)
        cropped = img.crop((x1, y1, x2, y2))
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        resized = cropped.resize((size, size), resample=resample)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(out_path)
        return True
    except Exception as e:
        logger.warning("Crop failed for %s: %s", out_path.stem, e)
        return False


def crop_localizations_parallel(
    download_dir: str,
    localizations_jsonl_path: str,
    crops_dir: str,
    size: int = 224,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """
    Crop all localizations from their media in parallel. Saves using elemental_id as filestem (e.g. elemental_id.png).
    Returns (num_cropped, num_failed).
    """
    if not os.path.exists(download_dir) or not os.path.exists(localizations_jsonl_path):
        logger.warning("Download dir or localizations JSONL missing; skipping crops")
        return (0, 0)
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; skipping crops")
        return (0, 0)
    download_path = Path(download_dir)
    crops_path = Path(crops_dir)
    crops_path.mkdir(parents=True, exist_ok=True)
    media_id_to_path: dict[int, Path] = {}
    for f in download_path.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            stem = f.stem
            if "_" in stem:
                try:
                    mid = int(stem.split("_", 1)[0])
                    media_id_to_path[mid] = f
                except ValueError:
                    pass
    locs_by_media: dict[int, list[dict]] = {}
    with open(localizations_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                loc = json.loads(line)
            except json.JSONDecodeError:
                continue
            media_id = loc.get("media")
            if media_id is None:
                continue
            mid = int(media_id)
            if mid not in locs_by_media:
                locs_by_media[mid] = []
            locs_by_media[mid].append(loc)
    tasks: list[tuple[Path, dict, Path]] = []
    for mid, locs in locs_by_media.items():
        image_path = media_id_to_path.get(mid)
        if image_path is None or not image_path.exists():
            continue
        for loc in locs:
            elemental_id = loc.get("elemental_id") or loc.get("id")
            if elemental_id is None:
                continue
            out_path = crops_path / image_path.stem / f"{elemental_id}.png"
            tasks.append((image_path, loc, out_path))
    if not tasks:
        print("[sync] No localization crops to process", flush=True)
        return (0, 0)
    print(f"[sync] Cropping {len(tasks)} localizations in parallel (size={size}x{size})", flush=True)
    workers = max_workers or min(32, (os.cpu_count() or 4) * 2)
    num_ok = 0
    num_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_crop_one, size, imp, loc, op): (imp, loc, op) for imp, loc, op in tasks}
        for fut in as_completed(futures):
            if fut.exception():
                num_fail += 1
                logger.warning("Crop task failed: %s", fut.exception())
            elif fut.result():
                num_ok += 1
            else:
                num_fail += 1
    print(f"[sync] Crops done: {num_ok} saved to {crops_path}, {num_fail} failed", flush=True)
    return (num_ok, num_fail)


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

    sess = get_session(project_id)
    resolved_db = (
        (database_name.strip() if database_name and database_name.strip() else None)
        or (sess.get("database_name") if sess else None)
        or get_database_name(project_id, None)
    )
    fo.config.database_name = resolved_db
    print(f"[sync] database_name={resolved_db}", flush=True)

    media_ids_list: list[int] = []
    tmp_dir = ""
    localizations_path = ""
    crops_dir = ""
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
        print(f"[sync] Fetching localizations...", flush=True)
        localizations_path = fetch_and_save_localizations(api, project_id, version_id=version_id)
        if localizations_path:
            print(f"[sync] saved_localizations_path (JSONL): {localizations_path}", flush=True)
        if tmp_dir and localizations_path:
            crops_dir = os.path.join(_project_tmp_dir(project_id), "crops")
            crop_localizations_parallel(tmp_dir, localizations_path, crops_dir, size=224)
    except Exception as e:
        logger.exception("fetch/save media or localizations failed: %s", e)
        print(f"[sync] Error: {e}", flush=True)

    print(f"[sync] sync_project_to_fiftyone done: saved_media_dir={tmp_dir or None} saved_localizations_path={localizations_path or None} crops_dir={crops_dir or None}", flush=True)
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
        "saved_crops_dir": crops_dir or None,
    }


def main() -> None:
    """Read env (HOST, TOKEN, PROJECT_ID, optional MEDIA_IDS, VERSION_ID) and fetch media + localizations."""
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
    import tator
    api = tator.get_api(host, token)
    if media_ids:
        all_media = get_media_chunked(api, project_id, media_ids)
        if all_media:
            saved_dir = save_media_to_tmp(api, project_id, all_media)
            print("saved_media_dir:", saved_dir)
        else:
            print("No Media objects returned; download skipped.")
    version_id_str = os.getenv("VERSION_ID", "").strip()
    version_id = int(version_id_str) if version_id_str else None
    localizations_path = fetch_and_save_localizations(api, project_id, version_id=version_id)
    if localizations_path:
        print("saved_localizations_path (JSONL):", localizations_path)
    base_dir = _project_tmp_dir(project_id)
    download_dir = os.path.join(base_dir, "download")
    crops_dir = os.path.join(base_dir, "crops")
    if os.path.isdir(download_dir) and localizations_path:
        crop_localizations_parallel(download_dir, localizations_path, crops_dir, size=224)


if __name__ == "__main__":
    main()
