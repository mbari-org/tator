"""
Tator to FiftyOne sync: fetch media + localizations, build FiftyOne dataset, launch app.
Phase 2 implementation. Requires fiftyone, tator, Pillow, PyYAML and MongoDB.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import fiftyone as fo
import tator
import yaml
from PIL import Image
from database_uri_config import database_name_from_uri
from database_manager import (
    get_database_entry,
    get_database_name,
    get_database_uri,
    get_port_for_project,
    get_session,
    get_vss_project,
)
from sync_lock import get_sync_lock_key, release_sync_lock, try_acquire_sync_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.info to console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
MEDIA_ID_BATCH_SIZE = 200


def _test_mongodb_connection(database_uri: str, timeout_ms: int = 5000) -> None:
    """Verify MongoDB is reachable before doing expensive Tator API work.

    Raises ConnectionError if the server cannot be reached within *timeout_ms*.
    """
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

    client = MongoClient(database_uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        raise ConnectionError(
            f"Cannot connect to MongoDB at {database_uri}: {exc}"
        ) from exc
    finally:
        client.close()


def _stop_process_on_port(port: int) -> None:
    """
    Try to stop any process listening on the given port (e.g. so another
    process can bind to it). Used by the CLI when re-running sync.
    """
    if sys.platform not in ("darwin", "linux", "linux2"):
        return
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout or not out.stdout.strip():
            return
        pids = [p.strip() for p in out.stdout.strip().split() if p.strip()]
        for pid in pids:
            try:
                subprocess.run(["kill", "-TERM", pid], capture_output=True, timeout=2)
            except Exception:
                pass
        if pids:
            time.sleep(1.5)
            logger.info(f"Stopped existing process(es) on port {port} (PIDs: {pids})")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logging.getLogger(__name__).debug(f"Could not stop process on port {port}: {e}")


LOCALIZATION_BATCH_SIZE = 5000
MEDIA_ID_BATCH_SIZE = 200


def _json_serial(obj: Any) -> Any:
    """Convert datetime/date to epoch seconds (float) for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.timestamp()
    if isinstance(obj, date):
        return datetime.combine(obj, datetime.min.time()).timestamp()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_SYNC_BASE = os.environ.get("FIFTYONE_SYNC_BASE", "/tmp/fiftyone_sync")


def _project_tmp_dir(project_id: int) -> str:
    """Return a project-isolated directory under /tmp for synced media.

    .. deprecated:: Use the specialized helpers (_download_dir, _data_dir,
       _crops_dir, _localizations_jsonl_path) instead.
    """
    path = os.path.join("/tmp", f"fiftyone_sync_project_{project_id}")
    os.makedirs(path, exist_ok=True)
    return path


def _version_slug(version_id: int | None) -> str:
    return f"v{version_id}" if version_id is not None else "v_all"


def _download_dir(project_id: int) -> str:
    """Ephemeral media-download directory, isolated from JSONL and crops."""
    path = os.path.join(_SYNC_BASE, "downloads", str(project_id))
    os.makedirs(path, exist_ok=True)
    return path


def _data_dir(project_id: int, version_id: int | None) -> str:
    """Per-project+version directory for JSONL, crops, and manifest."""
    path = os.path.join(_SYNC_BASE, "data", str(project_id), _version_slug(version_id))
    os.makedirs(path, exist_ok=True)
    return path


def _crops_dir(project_id: int, version_id: int | None) -> str:
    """Per-project+version crops directory."""
    path = os.path.join(_data_dir(project_id, version_id), "crops")
    os.makedirs(path, exist_ok=True)
    return path


def _localizations_jsonl_path(project_id: int, version_id: int | None) -> str:
    """Per-project+version JSONL path."""
    return os.path.join(_data_dir(project_id, version_id), "localizations.jsonl")


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
    logger.info(f"fetch_project_media_ids: project_id={project_id} filter={media_ids_filter}")
    host = api_url.rstrip("/")
    api = tator.get_api(host, token)
    if media_ids_filter:
        media_list = api.get_media_list(project_id, media_id=media_ids_filter)
    else:
        media_list = api.get_media_list(project_id)
    media_ids = [m.id for m in media_list]
    logger.info(f"Project {project_id} media count: {len(media_ids)}")
    return media_ids


def get_media_chunked(api: Any, project_id: int, media_ids: list[int]) -> list[Any]:
    """
    Get media objects in chunks of MEDIA_ID_BATCH_SIZE. Uses get_media_list_by_id for reliable Media objects.
    Filters out non-Media responses (API quirk). Returns list of tator.models.Media.
    """
    logger.info(f"get_media_chunked: project_id={project_id} num_ids={len(media_ids)} chunk_size={MEDIA_ID_BATCH_SIZE}")
    if not media_ids:
        logger.info("get_media_chunked: no ids, returning []")
        return []
    all_media = []
    for start in range(0, len(media_ids), MEDIA_ID_BATCH_SIZE):
        chunk_ids = media_ids[start : start + MEDIA_ID_BATCH_SIZE]
        media = api.get_media_list_by_id(project_id, {"ids": chunk_ids})
        new_media = [m for m in media if isinstance(m, tator.models.Media)]
        all_media += new_media
        logger.info(f"get_media_chunked: start={start} chunk_len={len(new_media)} total_media={len(all_media)}")
    logger.info(f"get_media_chunked: done, {len(all_media)} Media objects")
    logger.info(f"get_media_chunked: {len(media_ids)} ids -> {len(all_media)} Media objects")
    return all_media


# Video extensions: skip download (not supported); downloads come directly from Tator for images only.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")


def _is_video_name(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)


def save_media_to_tmp(
    api: Any,
    project_id: int,
    media_objects: list[Any],
    media_ids_filter: set[int] | None = None,
) -> str:
    """
    Download each media to an isolated download directory.
    Videos are skipped (download not supported). Existing non-empty files are skipped.
    When media_ids_filter is provided, only media whose id is in the set are downloaded.
    Retries each download up to 3 times. Returns the download directory path.
    """
    out_dir = _download_dir(project_id)
    valid = [m for m in media_objects if isinstance(m, tator.models.Media)]
    if media_ids_filter is not None:
        valid = [m for m in valid if m.id in media_ids_filter]
    total = len(valid)
    logger.info(f"Saving {total} media files to {out_dir}")
    downloaded = 0
    skipped = 0
    failed = 0
    log_interval = max(1, total // 10)
    for idx, m in enumerate(valid, 1):
        safe_name = f"{m.id}_{m.name}"
        out_path = os.path.join(out_dir, safe_name)
        if _is_video_name(m.name):
            skipped += 1
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue
        num_tries = 0
        success = False
        while num_tries < 3 and not success:
            try:
                for _ in tator.util.download_media(api, m, out_path):
                    pass
                success = True
                downloaded += 1
            except Exception as e:
                logger.debug(f"Download attempt {num_tries + 1}/3 failed for {m.id}: {e}")
                num_tries += 1
        if not success:
            failed += 1
            logger.warning(f"Could not download {m.name} after 3 tries")
        if idx % log_interval == 0 or idx == total:
            logger.info(f"Download progress: {idx}/{total} processed ({downloaded} saved, {skipped} skipped, {failed} failed)")
    logger.info(f"Download complete: {downloaded} saved, {skipped} skipped, {failed} failed -> {out_dir}")
    return out_dir


def fetch_and_save_localizations(
    api: Any,
    project_id: int,
    version_id: int | None = None,
    media_ids: list[int] | None = None,
) -> str:
    """
    Fetch all current localizations from Tator and write to a JSONL file.
    Overwrites the file (mode "w"), so the JSONL is always reconciled with Tator:
    removed localizations are absent, and the file is the single source of truth for the sync.
    Returns path to the file (e.g. .../localizations.jsonl). Uses LOCALIZATION_BATCH_SIZE per batch.
    If media_ids is provided, only localizations for those media are fetched (required when syncing
    a subset of media; avoids empty results when project localizations are scoped to media).

    Media IDs are batched into groups of MEDIA_ID_BATCH_SIZE to avoid 414 Request-URI Too Large
    errors from nginx when the project has many media.
    """
    out_path = _localizations_jsonl_path(project_id, version_id)
    logger.info(f"Localizations JSONL will be saved to: {out_path}")
    batch_size = LOCALIZATION_BATCH_SIZE

    media_id_batches: list[list[int] | None] = (
        [media_ids[i:i + MEDIA_ID_BATCH_SIZE] for i in range(0, len(media_ids), MEDIA_ID_BATCH_SIZE)]
        if media_ids else [None]
    )
    logger.info(f"Media ID batches: {len(media_id_batches)} batch(es) of up to {MEDIA_ID_BATCH_SIZE}")

    def _version_kw() -> dict:
        return {"version": [version_id]} if version_id is not None else {}
    try:
        loc_count = 0
        for mid_batch in media_id_batches:
            kw = _version_kw()
            if mid_batch:
                kw["media_id"] = mid_batch
            loc_count += api.get_localization_count(project_id, **kw)
        logger.info(f"get_localization_count(project_id={project_id}, media_ids={bool(media_ids)}, version={version_id}) = {loc_count}")
        if loc_count == 0 and version_id is not None:
            count_no_ver = 0
            for mid_batch in media_id_batches:
                kw: dict = {}
                if mid_batch:
                    kw["media_id"] = mid_batch
                count_no_ver += api.get_localization_count(project_id, **kw)
            if count_no_ver > 0:
                raise ValueError(
                    f"Version {version_id} has 0 localizations but {count_no_ver} exist across other versions; "
                    f"check that the correct version is specified"
                )
    except ValueError:
        raise
    except Exception as e:
        loc_count = None
        logger.exception(f"get_localization_count failed (will still try list): {e}")

    total = 0
    with open(out_path, "w") as f:
        def _fetch_all_locs() -> int:
            """Fetch localizations across all media_id batches, paginating each. Returns count."""
            fetched = 0
            for bidx, mid_batch in enumerate(media_id_batches):
                after_id = None
                while True:
                    kw = {"stop": batch_size}
                    if mid_batch:
                        kw["media_id"] = mid_batch
                    kw.update(_version_kw())
                    if after_id is not None:
                        kw["after"] = after_id
                    logger.info(
                        f"get_localization_list(project={project_id}, "
                        f"media_batch={bidx + 1}/{len(media_id_batches)}, {kw})"
                    )
                    try:
                        locs = api.get_localization_list(project_id, **kw)
                    except Exception as e:
                        logger.info(f"get_localization_list failed: {e}")
                        return fetched
                    if not locs:
                        logger.info(f"Localizations batch empty (media_batch={bidx + 1}, after={after_id}), moving on")
                        break
                    for loc in locs:
                        try:
                            obj = loc.to_dict() if hasattr(loc, "to_dict") else loc
                            f.write(json.dumps(obj, default=_json_serial) + "\n")
                        except Exception as e:
                            logger.info(f"Skip localization serialization: {e}")
                    fetched += len(locs)
                    after_id = locs[-1].id if locs else None
                    logger.info(f"Localizations batch: count={len(locs)} total_so_far={fetched} last_id={after_id}")
                    if len(locs) < batch_size:
                        break
            return fetched

        total = _fetch_all_locs()

    logger.info(f"Fetched {total} localizations -> {out_path}")
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
        logger.info(f"Crop failed for {out_path.stem}: {e}")
        return False


def crop_localizations_parallel(
    download_dir: str,
    localizations_jsonl_path: str,
    crops_dir: str,
    size: int = 224,
    max_workers: int | None = None,
    locs_to_crop: list[dict] | None = None,
) -> tuple[int, int]:
    """
    Crop localizations from their media in parallel.
    Saves using elemental_id as filestem (e.g. elemental_id.png).

    When locs_to_crop is provided, only those localizations are cropped (cache-miss
    optimization). Otherwise falls back to reading all localizations from the JSONL.

    Returns (num_cropped, num_failed).
    """
    if not os.path.exists(download_dir):
        logger.info("Download dir missing; skipping crops")
        return (0, 0)
    if locs_to_crop is None and not os.path.exists(localizations_jsonl_path):
        logger.info("Localizations JSONL missing and no locs_to_crop provided; skipping crops")
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

    if locs_to_crop is not None:
        loc_list = locs_to_crop
    else:
        loc_list = []
        with open(localizations_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    loc_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    locs_by_media: dict[int, list[dict]] = {}
    for loc in loc_list:
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
        logger.info("No localization crops to process")
        return (0, 0)
    logger.info(f"Cropping {len(tasks)} localizations in parallel (size={size}x{size})")
    workers = max_workers or min(128, (os.cpu_count() or 4) * 2)
    num_ok = 0
    num_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_crop_one, size, imp, loc, op): (imp, loc, op) for imp, loc, op in tasks}
        for fut in as_completed(futures):
            if fut.exception():
                num_fail += 1
                logger.info(f"Crop task failed: {fut.exception()}")
            elif fut.result():
                num_ok += 1
            else:
                num_fail += 1
    logger.info(f"Crops done: {num_ok} saved to {crops_path}, {num_fail} failed")
    return (num_ok, num_fail)


def _load_config(path: str) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _load_localizations_index(jsonl_path: str) -> dict[str, dict]:
    """Load localizations JSONL and index by elemental_id (or id)."""
    index: dict[str, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                loc = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = loc.get("elemental_id") or loc.get("id")
            if eid is not None:
                index[str(eid)] = loc
    return index


def _box_hash(loc: dict | None) -> str:
    """Hash of bounding box (x, y, width, height) for detecting box changes."""
    if not loc:
        return ""
    x = loc.get("x")
    y = loc.get("y")
    w = loc.get("width")
    h = loc.get("height")
    return hashlib.sha256(
        f"{x}|{y}|{w}|{h}".encode()
    ).hexdigest()


def _crop_manifest_path(project_id: int, version_id: int | None) -> str:
    """Path to the crop manifest JSON for a project+version."""
    return os.path.join(_data_dir(project_id, version_id), "crop_manifest.json")


def _load_crop_manifest(project_id: int, version_id: int | None) -> dict[str, dict]:
    """
    Load the crop manifest from disk.
    Returns {elemental_id: {"box_hash": str, "media_id": int, "media_stem": str}} or empty dict.
    """
    path = _crop_manifest_path(project_id, version_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.info(f"Could not load crop manifest {path}: {e}")
        return {}


def _save_crop_manifest(project_id: int, version_id: int | None, manifest: dict[str, dict]) -> None:
    """Atomically write the crop manifest to disk."""
    path = _crop_manifest_path(project_id, version_id)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.info(f"Could not save crop manifest {path}: {e}")


def _cleanup_download_dir(project_id: int) -> None:
    """Remove the download directory to reclaim disk space after crops are produced."""
    dl_dir = _download_dir(project_id)
    if os.path.isdir(dl_dir):
        try:
            shutil.rmtree(dl_dir)
            logger.info(f"Removed download directory: {dl_dir}")
        except OSError as e:
            logger.info(f"Could not remove download directory {dl_dir}: {e}")


def _find_crop_cache_misses(
    localizations_jsonl_path: str,
    crops_dir: str,
    manifest: dict[str, dict],
    download_dir: str | None = None,
) -> tuple[set[int], list[dict], dict[str, dict]]:
    """
    Diff current localizations against the crop manifest and on-disk crop files.
    A localization is a "miss" (needs cropping) when:
      - its elemental_id is absent from the manifest, OR
      - its box_hash differs from the manifest entry, OR
      - the crop file does not exist on disk.

    The media_stem for each localization is resolved in priority order:
    The media_stem for each localization is resolved in priority order:
      1. From the existing manifest entry (survives download-dir cleanup)
      2. From files currently in the download directory
      3. Bare media_id as fallback (new localizations on first encounter)

    Returns:
        media_ids_needed: set of media IDs that must be downloaded (have >= 1 miss)
        locs_to_crop:     list of localization dicts that need cropping
        updated_manifest:  new manifest reflecting current localizations (to be saved after cropping)
    """
    download_stem_map = _media_id_to_stem(download_dir) if download_dir else {}

    manifest_stem_map: dict[int, str] = {}
    for entry in manifest.values():
        mid = entry.get("media_id")
        stem = entry.get("media_stem")
        if mid is not None and stem:
            manifest_stem_map[int(mid)] = stem

    crops_path = Path(crops_dir)

    media_ids_needed: set[int] = set()
    locs_to_crop: list[dict] = []
    updated_manifest: dict[str, dict] = {}

    with open(localizations_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                loc = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = loc.get("elemental_id") or loc.get("id")
            if eid is None:
                continue
            eid = str(eid)
            media_id = loc.get("media")
            if media_id is None:
                continue
            mid = int(media_id)
            current_hash = _box_hash(loc)

            media_stem = (
                manifest_stem_map.get(mid)
                or download_stem_map.get(mid)
                or f"{mid}"
            )

            updated_manifest[eid] = {
                "box_hash": current_hash,
                "media_id": mid,
                "media_stem": media_stem,
            }

            old_entry = manifest.get(eid)
            crop_file = crops_path / media_stem / f"{eid}.png"

            is_miss = (
                old_entry is None
                or old_entry.get("box_hash") != current_hash
                or not crop_file.exists()
            )
            if is_miss:
                media_ids_needed.add(mid)
                locs_to_crop.append(loc)

    total_locs = len(updated_manifest)
    hits = total_locs - len(locs_to_crop)
    logger.info(
        f"Crop cache: {total_locs} localizations, {hits} hits, "
        f"{len(locs_to_crop)} misses across {len(media_ids_needed)} media"
    )
    return media_ids_needed, locs_to_crop, updated_manifest


def _patch_manifest_stems(manifest: dict[str, dict], download_dir: str) -> None:
    """
    After downloading new media, update manifest entries whose media_stem is
    still a bare media_id (fallback) with the real stem from the download directory.
    """
    real_stems = _media_id_to_stem(download_dir)
    if not real_stems:
        return
    for entry in manifest.values():
        mid = entry.get("media_id")
        if mid is None:
            continue
        mid = int(mid)
        current_stem = entry.get("media_stem", "")
        if current_stem == str(mid) and mid in real_stems:
            entry["media_stem"] = real_stems[mid]


def _cleanup_deleted_crops(
    manifest: dict[str, dict],
    updated_manifest: dict[str, dict],
    crops_dir: str,
) -> int:
    """
    Remove crop files for localizations that were deleted in Tator
    (present in old manifest but absent from updated_manifest).
    Returns count of files removed.
    """
    removed = 0
    deleted_eids = set(manifest.keys()) - set(updated_manifest.keys())
    for eid in deleted_eids:
        entry = manifest[eid]
        media_stem = entry.get("media_stem", str(entry.get("media_id", "")))
        crop_file = Path(crops_dir) / media_stem / f"{eid}.png"
        if crop_file.exists():
            try:
                crop_file.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Cleaned up {removed} orphaned crop files ({len(deleted_eids)} deleted localizations)")
    return removed


def _content_hash(label: Any, confidence: Any) -> str:
    """Hash of label+confidence for sync dedup. Must match sync_edits_to_tator."""
    return hashlib.sha256(
        f"{label}|{float(confidence) if confidence is not None else ''}".encode()
    ).hexdigest()


def _tator_localization_url(
    api_url: str,
    project_id: int,
    loc: dict,
    version_id: int | None = None,
) -> str | None:
    """
    Build Tator annotation UI URL that opens the media with this localization selected.
    Format: {base}/{project_id}/annotation/{media_id}?sort_by=$name&selected_entity={elemental_id}&selected_type=...&version=...&lock=0&fill_boxes=1&toggle_text=1
    Uses version_id if provided, else loc['version']. selected_type uses loc['type'] (id or name).
    Returns None if api_url, project_id, or media id is missing.
    """
    if not api_url or project_id is None:
        return None
    base = api_url.rstrip("/")
    vid = version_id if version_id is not None else loc.get("version")
    media_id = loc.get("media")
    if media_id is None or vid is None:
        return None
    path = f"{base}/{project_id}/annotation/{media_id}"
    elemental_id = loc.get("elemental_id") or loc.get("id")
    selected_type = loc.get("type")  # type id (int) or type name (str) if present
    if selected_type is not None:
        selected_type = str(selected_type)
    else:
        selected_type = ""
    params = {
        "sort_by": "$name",
        "selected_entity": elemental_id or "",
        "selected_type": selected_type,
        "version": str(vid),
        "lock": "0",
        "fill_boxes": "1",
        "toggle_text": "1",
    }
    query = urlencode(params, safe="")
    return f"{path}?{query}"


def _media_id_to_stem(download_dir: str) -> dict[int, str]:
    """Map media_id -> file stem for crop path resolution (e.g. 123 -> '123_image')."""
    out: dict[int, str] = {}
    if not download_dir or not os.path.exists(download_dir):
        return out
    for f in Path(download_dir).iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            stem = f.stem
            if "_" in stem:
                try:
                    mid = int(stem.split("_", 1)[0])
                    out[mid] = stem
                except ValueError:
                    pass
    return out


def _media_id_to_stem_from_crops(crops_dir: str) -> dict[int, str]:
    """Fallback: derive media_id -> stem mapping from crops subdirectory names.

    Crops are stored as crops_dir/{media_stem}/{eid}.png where media_stem
    typically starts with the numeric media_id (e.g. '12345' or '12345_image').
    """
    out: dict[int, str] = {}
    if not crops_dir or not os.path.isdir(crops_dir):
        return out
    for d in Path(crops_dir).iterdir():
        if not d.is_dir():
            continue
        stem = d.name
        try:
            mid = int(stem.split("_", 1)[0])
            out[mid] = stem
        except ValueError:
            pass
    return out


def _create_sample_from_loc(
    loc: dict,
    crops_dir: str,
    media_stem: str,
    include_classes: set[str],
    api_url: str | None = None,
    project_id: int | None = None,
    version_id: int | None = None,
) -> fo.Sample | None:
    """Create a FiftyOne sample from a localization (for reconcile add-new)."""
    elemental_id = loc.get("elemental_id") or loc.get("id")
    if elemental_id is None:
        return None
    elemental_id = str(elemental_id)
    label = _get_label_from_loc(loc)
    if include_classes and label not in include_classes:
        return None
    filepath = os.path.abspath(os.path.join(crops_dir, media_stem, f"{elemental_id}.png"))
    if not os.path.exists(filepath):
        return None
    sample = fo.Sample(filepath=filepath)
    sample["ground_truth"] = fo.Classification(label=label, confidence=1.0)
    sample["elemental_id"] = elemental_id
    sample["media_stem"] = media_stem
    sample["box_hash"] = _box_hash(loc)
    if api_url and project_id is not None:
        tator_url = _tator_localization_url(api_url, project_id, loc, version_id)
        if tator_url:
            sample["annotation"] = tator_url
    sample.tags.append(label)
    attrs = loc.get("attributes") or {}
    score = attrs.get("Score") or attrs.get("score")
    if score is not None:
        sample["confidence"] = float(score)
    conf = sample["confidence"] if "confidence" in sample else 1.0
    sample["last_sync_hash"] = _content_hash(label, conf)
    return sample


def reconcile_dataset_with_tator(
    dataset: fo.Dataset,
    loc_index: dict[str, dict],
    crops_dir: str,
    download_dir: str | None,
    config: dict[str, Any],
    max_samples: int | None,
) -> fo.Dataset:
    """
    Reconcile existing dataset with current Tator localizations:
    - Remove samples whose elemental_id was deleted in Tator
    - Update samples whose bounding box changed (crop file already overwritten)
    - Add samples for new elemental_ids in Tator
    """
    tator_eids = set(loc_index.keys())
    include_classes = set(config.get("include_classes") or [])
    media_id_to_stem = _media_id_to_stem(download_dir) if download_dir else {}
    if not media_id_to_stem:
        media_id_to_stem = _media_id_to_stem_from_crops(crops_dir)

    # 1. Remove samples deleted in Tator (only when we have a non-empty localization set from Tator)
    # If Tator returns 0 localizations (e.g. wrong version or API filter), do not remove existing samples
    if tator_eids:
        to_remove = []
        for s in dataset:
            eid = s["elemental_id"] if "elemental_id" in s else None
            if eid is not None and str(eid) not in tator_eids:
                to_remove.append(s.id)
        if to_remove:
            dataset.delete_samples(to_remove)
            logger.info(f"Reconcile: removed {len(to_remove)} samples (deleted in Tator)")
    else:
        logger.info(f"Reconcile: 0 localizations from Tator; skipping delete step (keeping existing samples)")

    # 2. Update samples with changed box (crop already overwritten by crop_localizations_parallel)
    updated = 0
    for sample in dataset.iter_samples(autosave=False):
        elemental_id = sample["elemental_id"] if "elemental_id" in sample else None
        if not elemental_id or str(elemental_id) not in loc_index:
            continue
        loc = loc_index[str(elemental_id)]
        new_hash = _box_hash(loc)
        old_hash = sample["box_hash"] if "box_hash" in sample else None
        if old_hash != new_hash:
            sample["box_hash"] = new_hash
            sample.save()
            updated += 1
    if updated:
        logger.info(f"Reconcile: updated {updated} samples (box changed)")

    # 3. Add new samples (elemental_id in Tator but not in dataset)
    dataset_eids = {str(s["elemental_id"]) for s in dataset if "elemental_id" in s}
    new_eids = [eid for eid in tator_eids if eid not in dataset_eids]
    if max_samples:
        cap = max_samples - len(dataset)
        if cap <= 0:
            new_eids = []
        else:
            new_eids = new_eids[:cap]
    added = 0
    for eid in new_eids:
        loc = loc_index[eid]
        media_id = loc.get("media")
        if media_id is None:
            continue
        media_stem = media_id_to_stem.get(int(media_id))
        if not media_stem:
            continue
        api_url = config.get("api_url")
        project_id = config.get("project_id")
        version_id = config.get("version_id")
        sample = _create_sample_from_loc(
            loc, crops_dir, media_stem, include_classes,
            api_url=api_url, project_id=project_id, version_id=version_id,
        )
        if sample:
            dataset.add_samples([sample])
            added += 1
    if added:
        logger.info(f"Reconcile: added {added} new samples")

    return dataset


def _get_label_from_loc(loc: dict) -> str:
    """Extract label from localization attributes (Label, label) or fallback."""
    attrs = loc.get("attributes") or {}
    label = attrs.get("Label") or attrs.get("label")
    if label:
        return str(label)
    media_stem = None
    media_id = loc.get("media")
    if media_id is not None:
        media_stem = str(media_id)
    return media_stem or "unknown"


def build_fiftyone_dataset_from_crops(
    crops_dir: str,
    localizations_jsonl_path: str,
    dataset_name: str,
    config: dict[str, Any] | None = None,
    download_dir: str | None = None,
) -> Any:
    """
    Build a FiftyOne dataset from crop images and localizations JSONL.

    Crops layout: crops/{media_file_stem}/{elemental_id}.png
    JSONL: one JSON per line with elemental_id, media, x, y, width, height, attributes, etc.

    Config keys (optional):
        include_classes: list of labels to include (None = all)
        image_extensions: glob patterns (default: ["*.png", "*.jpg", ...])
        max_samples: max samples to load (None = no limit)

    Returns the FiftyOne dataset.
    """
    config = config or {}
    include_classes = set(config.get("include_classes") or [])
    image_extensions = config.get("image_extensions") or ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    max_samples = config.get("max_samples")

    # Load localizations index by elemental_id
    loc_index = _load_localizations_index(localizations_jsonl_path)
    logger.info(f"Loaded {len(loc_index)} localizations from JSONL")

    # Collect crop filepaths
    samples: list = []
    seen = 0
    for ext in image_extensions:
        pat = os.path.join(crops_dir, "**", ext)
        for filepath in glob.glob(pat):
            seen += 1
            if max_samples and len(samples) >= max_samples:
                break
            rel = os.path.relpath(filepath, crops_dir)
            parts = Path(rel).parts
            if len(parts) < 2:
                continue
            media_stem = parts[0]
            elemental_id = Path(filepath).stem

            loc = loc_index.get(elemental_id)
            label = _get_label_from_loc(loc) if loc else media_stem or "unknown"

            if include_classes and label not in include_classes:
                continue

            sample = fo.Sample(filepath=os.path.abspath(filepath))
            sample["ground_truth"] = fo.Classification(label=label, confidence=1.0)
            sample["elemental_id"] = elemental_id
            sample["media_stem"] = media_stem
            sample["box_hash"] = _box_hash(loc)
            api_url = config.get("source_url")
            project_id = config.get("project_id")
            version_id = config.get("version_id")
            if loc and api_url and project_id is not None:
                tator_url = _tator_localization_url(api_url, project_id, loc, version_id)
                if tator_url:
                    sample["annotation"] = tator_url
            sample.tags.append(label)
            if loc:
                attrs = loc.get("attributes") or {}
                score = attrs.get("Score") or attrs.get("score")
                if score is not None:
                    sample["confidence"] = float(score)
            conf = sample["confidence"] if "confidence" in sample else 1.0
            sample["last_sync_hash"] = _content_hash(label, conf)
            samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError(f"No crops found in {crops_dir} (checked {seen} files)")

    logger.info(f"Collected {len(samples)} samples for dataset")

    # Handle existing dataset: always reconcile, never delete
    if dataset_name in fo.list_datasets(): 
        logger.info(f"Reconcile: loading dataset {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
        dataset.persistent = True  # Ensure dataset persists in MongoDB after session ends
        dataset = reconcile_dataset_with_tator(
            dataset=dataset,
            loc_index=loc_index,
            crops_dir=crops_dir,
            download_dir=download_dir,
            config=config,
            max_samples=max_samples,
        )
        logger.info(f"Reconcile: dataset {dataset_name} loaded")
        return dataset

    logger.info(f"Reconcile: creating new dataset {dataset_name} in database {fo.config.database_name}")
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True  # Persist dataset in MongoDB after session ends
    dataset.add_samples(samples)
    logger.info(f"Created dataset '{dataset_name}' with {len(samples)} samples")
    return dataset


DEFAULT_LABEL_ATTR = "Label"
DEFAULT_SCORE_ATTR = "score"


def _sanitize_dataset_name(name: str) -> str:
    """Make a string safe for use as a FiftyOne/MongoDB dataset name."""
    if not name:
        return "default"
    # Replace anything that isn't alphanumeric, underscore, or hyphen with underscore
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip())
    return re.sub(r"_+", "_", s).strip("_") or "default"


def _default_dataset_name(api: Any, project_id: int, version_id: int | None) -> str:
    """Default FiftyOne dataset name: project.name + '_' + version_name (from Tator API)."""
    try:
        project = api.get_project(project_id)
        project_name = _sanitize_dataset_name(project.name) if project.name else f"project_{project_id}"
    except Exception:
        project_name = f"project_{project_id}"
    if version_id is not None:
        try:
            version = api.get_version(version_id)
            version_name = _sanitize_dataset_name(version.name) if version.name else f"v{version_id}"
        except Exception:
            version_name = f"v{version_id}"
    else:
        version_name = "default"
    return f"{project_name}_{version_name}"


def _dataset_name_with_port(dataset_name: str, port: int) -> str:
    """Append port to dataset name if not already present (e.g. 901902-uavs_Baseline -> 901902-uavs_Baseline_5151)."""
    name = (dataset_name or "").strip()
    if not name:
        return name
    suffix = f"_{port}"
    return name if name.endswith(suffix) else f"{name}{suffix}"


def _update_localization_attributes(
    api: Any,
    project_id: int,
    version_id: int,
    elemental_id: str,
    attrs: dict[str, Any],
) -> None:
    """
    Update a localization's attributes by elemental_id.
    Uses get_localization_list to resolve elemental_id to id, then
    api.update_localization(id, localization_update) per Tator API.
    """
    locs = api.get_localization_list(
        project_id, version=[version_id], elemental_id=elemental_id
    )
    locs_list = list(locs) if not isinstance(locs, list) else locs
    if not locs_list:
        raise ValueError(f"No localization found for elemental_id={elemental_id}")
    loc_id = locs_list[0].id
    api.update_localization(loc_id, localization_update={"attributes": attrs})


def sync_edits_to_tator(
    project_id: int,
    version_id: int,
    port: int,
    api_url: str,
    token: str,
    dataset_name: str | None = None,
    label_attr: str | None = DEFAULT_LABEL_ATTR,
    score_attr: str | None = DEFAULT_SCORE_ATTR,
    debug: bool = False,
    project_name: str | None = None,
) -> dict[str, Any]:
    """
    Push FiftyOne dataset edits (labels, confidence) back to Tator localizations.
    Matches samples by elemental_id; looks up localization id via get_localization_list,
    then updates attributes via update_localization(id, localization_update).
    Returns {"status": "ok", "updated": int, "failed": int, "errors": list} or raises.
    """
    db_entry = get_database_entry(project_id, port, project_name=project_name)
    if db_entry is None:
        raise ValueError(f"No database entry found for project_id={project_id} and port={port}")
    db_name = database_name_from_uri(db_entry.uri)
    fo.config.database_uri = db_entry.uri
    fo.config.database_name = db_name
    os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
    os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name

    _test_mongodb_connection(fo.config.database_uri)

    host = api_url.rstrip("/")
    api = tator.get_api(host, token)
    ds_name = dataset_name or _default_dataset_name(api, project_id, version_id)

    # Resolve dataset by project name + port (datasets may have been created
    # with a version component or port suffix that differs from _default_dataset_name).
    try:
        _proj = api.get_project(project_id)
        project_prefix = _sanitize_dataset_name(_proj.name) if _proj.name else f"project_{project_id}"
    except Exception:
        project_prefix = f"project_{project_id}"
    port_suffix = f"_{port}"

    def _resolve_dataset(requested: str) -> str | None:
        """Return the actual dataset name: exact match first, then project+port match."""
        available = fo.list_datasets()
        if requested in available:
            return requested
        matches = [d for d in available
                   if d.startswith(project_prefix) and d.endswith(port_suffix)]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        for candidate in matches:
            if candidate == f"{project_prefix}{port_suffix}":
                return candidate
        return matches[0]

    fallback_db = f"{os.environ.get('FIFTYONE_DATABASE_DEFAULT', 'fiftyone_project')}_{project_id}"
    resolved = _resolve_dataset(ds_name)
    if resolved is None and db_name != fallback_db:
        fo.config.database_name = fallback_db
        os.environ["FIFTYONE_DATABASE_NAME"] = fallback_db
        resolved = _resolve_dataset(ds_name)
    if resolved is None:
        fo.config.database_name = db_name
        raise ValueError(
            f"No dataset matching project '{project_prefix}' with port {port} found in database '{db_name}' (or '{fallback_db}'). "
            "Run POST /sync first. Ensure FIFTYONE_DATABASE_URI and FIFTYONE_DATABASE_NAME match the sync process."
        )
    ds_name = resolved

    dataset = fo.load_dataset(ds_name)

    updated = 0
    failed = 0
    skipped = 0
    errors: list[str] = []
    _debug = debug or os.environ.get("FIFTYONE_SYNC_DEBUG", "").lower() in ("1", "true", "yes")

    for sample in dataset.iter_samples(autosave=False):
        elemental_id = sample["elemental_id"] if "elemental_id" in sample else None
        if not elemental_id:
            failed += 1
            errors.append(f"Sample {sample.id}: missing elemental_id")
            continue
        gt = sample["ground_truth"] if "ground_truth" in sample else None
        label = gt.label if gt else None
        confidence = sample["confidence"] if "confidence" in sample else None
        if confidence is None and gt:
            confidence = getattr(gt, "confidence", None)
        attrs: dict[str, Any] = {}
        if label is not None and label_attr:
            attrs[label_attr] = str(label)
        if confidence is not None and score_attr:
            attrs[score_attr] = float(confidence)
        if not attrs:
            continue
        # Content-based skip: only push if label/confidence changed since last sync.
        # Build sets last_sync_hash from Tator; first sync skips all if no user edits.
        current_hash = _content_hash(label, confidence)
        last_sync_hash = sample["last_sync_hash"] if "last_sync_hash" in sample else None
        if current_hash == last_sync_hash:
            skipped += 1
            if _debug:
                logger.info(f"SKIP elem={elemental_id} hash={current_hash[:8]}")
            continue
        try:
            _update_localization_attributes(
                api, project_id, version_id, str(elemental_id), attrs
            )
            sample["last_sync_hash"] = current_hash
            sample.save()
            updated += 1
            if _debug:
                logger.info(f"UPDATE elem={elemental_id}")
        except Exception as e:
            failed += 1
            errors.append(f"{elemental_id}: {e}")
            logger.info(f"Failed to update localization {elemental_id}: {e}")

    logger.info(f"sync_edits_to_tator: updated={updated} skipped={skipped} failed={failed}")
    return {"status": "ok", "updated": updated, "skipped": skipped, "failed": failed, "errors": errors[:20]}


def run_sync_job(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
    project_name: str,
    database_uri: str | None = None,
    database_name: str | None = None,
    config_path: str | None = None,
    launch_app: bool = True,
) -> dict[str, Any]:
    """
    Entrypoint for RQ worker: all args are serializable. Calls sync_project_to_fiftyone.
    """
    from database_manager import register_project_id_name
    register_project_id_name(project_id, project_name)
    return sync_project_to_fiftyone(
        project_id=project_id,
        version_id=version_id,
        api_url=api_url,
        token=token,
        port=port,
        project_name=project_name,
        database_uri=database_uri,
        database_name=database_name,
        config_path=config_path,
        launch_app=launch_app,
    )


def sync_project_to_fiftyone(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
    project_name: str | None = None,
    database_uri: str | None = None,
    database_name: str | None = None,
    config_path: str | None = None,
    launch_app: bool = True,
) -> dict[str, Any]:
    """
    Fetch Tator media and localizations, build FiftyOne dataset, launch App on given port.
    Uses per-project MongoDB database (database_uri when provided, else resolved via config; database_name override or get_database_name).
    Returns {"status": "ok", "dataset_name": str, "database_name": str} or raises.
    """
    logger.info(f"sync_project_to_fiftyone CALLED: project_id={project_id} version_id={version_id} api_url={api_url} port={port}")
    sess = get_session(project_id, port)
    resolved_db = (
        (database_name.strip() if database_name and database_name.strip() else None)
        or (sess.get("database_name") if sess else None)
    )
    fo.config.database_uri = (database_uri.strip() if database_uri and database_uri.strip() else None) or get_database_uri(project_id, port, project_name=project_name)
    fo.config.database_name = resolved_db
    logger.info(f"database_uri={fo.config.database_uri} database_name={resolved_db}")

    try:
        _test_mongodb_connection(fo.config.database_uri)
        logger.info("MongoDB connection OK")
    except ConnectionError as exc:
        logger.error(f"MongoDB pre-flight check failed: {exc}")
        raise RuntimeError(f"MongoDB pre-flight check failed: {exc}") from exc

    lock_key = get_sync_lock_key(resolved_db, project_id, version_id)
    if not try_acquire_sync_lock(lock_key):
        return {
            "status": "busy",
            "message": "This dataset is being updated by another sync. Please try again in a few minutes.",
            "database_name": resolved_db,
        }

    try:
        dl_dir = _download_dir(project_id)
        localizations_path = ""
        crops = _crops_dir(project_id, version_id)
        try:
            host = api_url.rstrip("/")
            api = tator.get_api(host, token)

            # 1. Fetch media IDs (lightweight metadata, needed for localization query)
            logger.info(f"Fetching media IDs... host={host} project_id={project_id} api_url={api_url}")
            media_ids_list = fetch_project_media_ids(api_url, token, project_id)

            # 2. Fetch localizations first (cheap metadata)
            logger.info("Fetching localizations...")
            localizations_path = fetch_and_save_localizations(
                api, project_id, version_id=version_id, media_ids=media_ids_list or None
            )
            if localizations_path:
                logger.info(f"saved_localizations_path (JSONL): {localizations_path}")

            # 3. Determine which crops are missing or stale (cache miss)
            old_manifest = _load_crop_manifest(project_id, version_id)
            media_ids_needed, locs_to_crop, updated_manifest = _find_crop_cache_misses(
                localizations_jsonl_path=localizations_path,
                crops_dir=crops,
                manifest=old_manifest,
                download_dir=dl_dir,
            )

            # 4. Clean up orphaned crop files for deleted localizations
            _cleanup_deleted_crops(old_manifest, updated_manifest, crops)

            # 5. Download only media that have cache misses
            if not media_ids_list:
                logger.info(f"No media IDs for project {project_id}; skipping download")
            elif not media_ids_needed:
                logger.info(f"All {len(updated_manifest)} crops are cached; skipping media download")
            else:
                needed_ids = [mid for mid in media_ids_list if mid in media_ids_needed]
                logger.info(f"Getting {len(needed_ids)}/{len(media_ids_list)} media objects (cache misses)...")
                all_media = get_media_chunked(api, project_id, needed_ids)
                if not all_media:
                    logger.info(f"No Media objects returned for {len(needed_ids)} ids; skipping download")
                else:
                    logger.info(f"Downloading {len(all_media)} media to tmp...")
                    dl_dir = save_media_to_tmp(api, project_id, all_media, media_ids_filter=media_ids_needed)

            # 6. Crop only the cache-miss localizations
            if locs_to_crop and localizations_path:
                crop_localizations_parallel(
                    dl_dir, localizations_path, crops, size=224, locs_to_crop=locs_to_crop,
                )
            elif not locs_to_crop:
                logger.info("No crop cache misses; skipping crop step")

            # 7. Patch manifest stems from actual downloaded filenames
            _patch_manifest_stems(updated_manifest, dl_dir)

            # 8. Persist the updated manifest
            _save_crop_manifest(project_id, version_id, updated_manifest)

            # 9. Remove downloaded media to reclaim disk space
            _cleanup_download_dir(project_id)

        except Exception as e:
            logger.info(f"fetch/save media or localizations failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "database_name": resolved_db,
                "saved_media_dir": dl_dir or None,
                "saved_localizations_path": localizations_path or None,
                "saved_crops_dir": crops or None,
            }

        if not localizations_path:
            logger.info("No localizations; skipping dataset build")
            return {
                "status": "ok",
                "message": "No crops to load; media/localizations missing or empty",
                "database_name": resolved_db,
                "dataset_name": None,
                "saved_media_dir": dl_dir or None,
                "saved_localizations_path": localizations_path or None,
                "saved_crops_dir": crops or None,
            }

        # Load config from YAML/JSON if provided
        config: dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            try:
                config = _load_config(config_path)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.info(f"Failed to load config {config_path}: {e}")

        # Inject Tator base URL and ids so sample "url" can link to the localization's media page
        config["source_url"] = api_url.rstrip("/")
        config["project_id"] = project_id
        config["version_id"] = version_id

        dataset_name = config.get("dataset_name") or _default_dataset_name(api, project_id, version_id)
        dataset_name = _dataset_name_with_port(dataset_name, port)

        # Set env so FiftyOne app subprocess uses the same database
        os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
        os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name

        try:
            logger.info(f"Building dataset {dataset_name}")
            dataset = build_fiftyone_dataset_from_crops(
                crops_dir=crops,
                localizations_jsonl_path=localizations_path,
                dataset_name=dataset_name,
                config=config,
                download_dir=dl_dir or None,
            )
        except Exception as e:
            logger.info(f"Dataset build failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "database_name": resolved_db,
                "dataset_name": None,
                "saved_media_dir": dl_dir or None,
                "saved_localizations_path": localizations_path or None,
                "saved_crops_dir": crops or None,
            }

        logger.info(f"sync_project_to_fiftyone done: dataset={dataset_name}")
        logger.info(
            "sync_project_to_fiftyone: project=%s port=%s database=%s dataset=%s",
            project_id, port, resolved_db, dataset_name,
        )

        sample_count = len(dataset)
        logger.info(f"Dataset '{dataset_name}' has {sample_count} samples")

        # Always compute embeddings (from service) and UMAP; config.embeddings overrides defaults
        embeddings_config = config.get("embeddings") or {}
        if not isinstance(embeddings_config, dict):
            embeddings_config = {}
        try:
            proj = api.get_project(project_id)
            project_name_for_config = getattr(proj, "name", None) or str(project_id)
        except Exception:
            project_name_for_config = str(project_id)
        vss_project = get_vss_project(project_name_for_config, port) if project_name_for_config else None
        if vss_project is None or not str(vss_project).strip():
            vss_project = embeddings_config.get("project_name")
        if vss_project is None or not str(vss_project).strip():
            vss_project = str(project_id)
        else:
            vss_project = str(vss_project).strip()
        if vss_project:
            model_info = {
                "embeddings_field": embeddings_config.get("embeddings_field", "embeddings"),
                "brain_key": embeddings_config.get("brain_key", "umap_viz"),
            }
            try:
                from embedding_service import is_embedding_service_available
                from embeddings_viz import compute_embeddings_and_viz

                if not is_embedding_service_available():
                    logger.info(
                        "Embedding service unavailable; skipping embeddings/UMAP (dataset still available)"
                    )
                else:
                    compute_embeddings_and_viz(
                        dataset,
                        model_info,
                        umap_seed=int(embeddings_config.get("umap_seed", 51)),
                        force_embeddings=bool(embeddings_config.get("force_embeddings", False)),
                        force_umap=bool(embeddings_config.get("force_umap", False)),
                        batch_size=embeddings_config.get("batch_size"),
                        project_name=vss_project,
                        service_url=embeddings_config.get("service_url") or os.environ.get("FASTVSS_API_URL"),
                    )
                    logger.info(f"Embeddings and UMAP completed for dataset '{dataset_name}'")
            except ImportError as e:
                logger.info(f"Skipping embeddings/UMAP (missing deps): {e}")
            except Exception as e:
                logger.info(f"Embeddings/UMAP failed (dataset still available): {e}")
                logging.getLogger(__name__).exception("Embeddings/UMAP failed")
        else:
            logger.info("No vss_project; skipping embeddings/UMAP")

        # URL for the launcher: must use FIFTYONE_APP_PUBLIC_BASE_URL so the dashboard
        # opens the correct FiftyOne app (e.g. maximilian.shore.mbari.org), always http.
        public_url = os.environ.get("FIFTYONE_APP_PUBLIC_BASE_URL", "http://localhost").strip()
        app_url = f"{public_url.rstrip('/')}:{port}"
        logger.info(f"FiftyOne app URL (FIFTYONE_APP_PUBLIC_BASE_URL): {app_url}")

        result = {
            "status": "ok",
            "dataset_name": dataset_name,
            "database_name": resolved_db,
            "sample_count": sample_count,
            "saved_media_dir": dl_dir or None,
            "saved_localizations_path": localizations_path or None,
            "saved_crops_dir": crops or None,
        }
        if app_url is not None:
            result["app_url"] = app_url
        result["port"] = port
        return result
    finally:
        release_sync_lock(lock_key)


def main() -> None:
    """Read env (HOST, TOKEN, PROJECT_ID, optional MEDIA_IDS, VERSION_ID) and fetch media + localizations."""
    host = os.getenv("HOST", "").rstrip("/")
    token = os.getenv("TOKEN")
    project_id_str = os.getenv("PROJECT_ID")
    media_ids_str = os.getenv("MEDIA_IDS", "").strip()
    logger.info(f"main: HOST={'<set>' if host else '<unset>'} PROJECT_ID={project_id_str or '<unset>'} MEDIA_IDS={'<set>' if media_ids_str else '<unset>'}")

    if not host or not token or not project_id_str:
        logger.info("Set HOST, TOKEN, and PROJECT_ID environment variables.")
        return
    project_id = int(project_id_str)
    media_ids_filter: list[int] | None = None
    if media_ids_str:
        media_ids_filter = [int(id_.strip()) for id_ in media_ids_str.split(",") if id_.strip()]

    api = tator.get_api(host, token)
    version_id_str = os.getenv("VERSION_ID", "").strip()
    version_id = int(version_id_str) if version_id_str else None

    try:
        project_name_cli = getattr(api.get_project(project_id), "name", None) or str(project_id)
    except Exception:
        project_name_cli = str(project_id)
    port = get_port_for_project(project_id, project_name=project_name_cli)

    # Fetch media IDs (lightweight)
    media_ids = fetch_project_media_ids(host, token, project_id, media_ids_filter=media_ids_filter)
    logger.info("media_ids: %s", media_ids)

    # Fetch localizations first (cheap metadata)
    localizations_path = fetch_and_save_localizations(
        api, project_id, version_id=version_id, media_ids=media_ids if media_ids else None
    )
    if localizations_path:
        logger.info("saved_localizations_path (JSONL): %s", localizations_path)

    dl_dir = _download_dir(project_id)
    crops = _crops_dir(project_id, version_id)

    # Determine cache misses
    if localizations_path:
        old_manifest = _load_crop_manifest(project_id, version_id)
        media_ids_needed, locs_to_crop, updated_manifest = _find_crop_cache_misses(
            localizations_jsonl_path=localizations_path,
            crops_dir=crops,
            manifest=old_manifest,
            download_dir=dl_dir,
        )
        _cleanup_deleted_crops(old_manifest, updated_manifest, crops)

        # Download only media with cache misses
        if media_ids and media_ids_needed:
            needed_ids = [mid for mid in media_ids if mid in media_ids_needed]
            all_media = get_media_chunked(api, project_id, needed_ids)
            if all_media:
                save_media_to_tmp(api, project_id, all_media, media_ids_filter=media_ids_needed)
            else:
                logger.info("No Media objects returned; download skipped.")
        elif not media_ids_needed:
            logger.info("All crops cached; skipping media download")

        # Crop only cache misses
        if locs_to_crop:
            crop_localizations_parallel(
                dl_dir, localizations_path, crops, size=224, locs_to_crop=locs_to_crop,
            )

        # Patch manifest stems from actual downloaded filenames
        _patch_manifest_stems(updated_manifest, dl_dir)

        # Save updated manifest
        _save_crop_manifest(project_id, version_id, updated_manifest)

        # Remove downloaded media to reclaim disk space
        _cleanup_download_dir(project_id)

    if crops and localizations_path and os.path.isdir(crops):
        fo.config.database_uri = get_database_uri(project_id, port, project_name=project_name_cli)
        fo.config.database_name = get_database_name(project_id, port, project_name=project_name_cli)
        os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
        os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name
        config_path = os.getenv("CONFIG_PATH")
        config = _load_config(config_path) if config_path and os.path.exists(config_path) else {}
        dataset_name = config.get("dataset_name") or _default_dataset_name(api, project_id, version_id)
        dataset_name = _dataset_name_with_port(dataset_name, port)
        dataset = build_fiftyone_dataset_from_crops(
            crops_dir=crops,
            localizations_jsonl_path=localizations_path,
            dataset_name=dataset_name,
            config=config,
            download_dir=dl_dir,
        )
        logger.info(f"Dataset built. FiftyOne app should be running in another container on port {port}.")


if __name__ == "__main__":
    main()
