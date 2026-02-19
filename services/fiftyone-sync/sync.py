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
from port_manager import get_database_name, get_database_uri, get_port_for_project, get_session

CHUNK_SIZE = 200


def _stop_process_on_port(port: int) -> None:
    """
    Try to stop any process listening on the given port so that launch_app
    starts a fresh FiftyOne server with the newly synced dataset. If an old
    server is left running, FiftyOne connects to it and never loads our dataset.
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
            print(f"[sync] Stopped existing process(es) on port {port} (PIDs: {pids})", flush=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logging.getLogger(__name__).debug("Could not stop process on port %s: %s", port, e)


def _launch_app_embedded(dataset: fo.Dataset, port: int):
    """
    Launch FiftyOne app for embedded/iframe use (no browser tab, no SSH instructions).
    Uses remote=True so FiftyOne does not open a browser; we suppress FiftyOne's
    SSH port-forwarding message and print a short note for the Tator dashboard instead.
    See https://docs.voxel51.com/installation/environments.html#remote-data
    """
    fo_session_logger = logging.getLogger("fiftyone.core.session.session")
    old_level = fo_session_logger.level
    fo_session_logger.setLevel(logging.WARNING)
    try:
        session = fo.launch_app(dataset, port=port, address="0.0.0.0", remote=True)
    finally:
        fo_session_logger.setLevel(old_level)
    print(
        f"[sync] FiftyOne app is running on port {port}. "
        "Open it in the Tator dashboard iframe (or at http://<host>:{port})".format(port=port),
        flush=True,
    )
    return session
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
    host = api_url.rstrip("/")
    api = tator.get_api(host, token)
    if media_ids_filter:
        media_list = api.get_media_list(project_id, media_id=media_ids_filter)
    else:
        media_list = api.get_media_list(project_id)
    media_ids = [m.id for m in media_list]
    print(f"[sync] fetch_project_media_ids: got {len(media_ids)} ids", flush=True)
    print("Project %s media count: %s; ids: %s", project_id, len(media_ids), media_ids)
    return media_ids


def get_media_chunked(api: Any, project_id: int, media_ids: list[int]) -> list[Any]:
    """
    Get media objects in chunks of CHUNK_SIZE. Uses get_media_list_by_id for reliable Media objects.
    Filters out non-Media responses (API quirk). Returns list of tator.models.Media.
    """
    print(f"[sync] get_media_chunked: project_id={project_id} num_ids={len(media_ids)} chunk_size={CHUNK_SIZE}", flush=True)
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
    print("get_media_chunked: %s ids -> %s Media objects", len(media_ids), len(all_media))
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
    base_dir = _project_tmp_dir(project_id)
    out_dir = os.path.join(base_dir, "download")
    os.makedirs(out_dir, exist_ok=True)
    valid = [m for m in media_objects if isinstance(m, tator.models.Media)]
    total = len(valid)
    print("Saving %s media files to %s", total, out_dir)
    print(f"[sync] Saving {total} media files to {out_dir}", flush=True)
    for idx, m in enumerate(valid, 1):
        safe_name = f"{m.id}_{m.name}"
        out_path = os.path.join(out_dir, safe_name)
        if _is_video_name(m.name):
            print(f"[sync] Skipping video (not supported): {safe_name}", flush=True)
            print("Skipping video %s", m.id)
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[sync] Skipping existing: {safe_name}", flush=True)
            print("Skipping existing %s", out_path)
            continue
        print(f"[sync] Downloading {m.name} to {out_path}", flush=True)
        num_tries = 0
        success = False
        while num_tries < 3 and not success:
            try:
                for _ in tator.util.download_media(api, m, out_path):
                    pass
                success = True
                print("Saved %s -> %s", m.id, out_path)
                print(f"[sync] Saved media {m.id} -> {out_path}", flush=True)
            except Exception as e:
                print(f"[sync] Download attempt {num_tries + 1}/3 failed for {m.id}: {e}", flush=True)
                print(f"[sync] Attempt {num_tries + 1}/3 failed for {m.id}: {e}", flush=True)
                num_tries += 1
        if not success:
            print(f"[sync] Could not download {m.name} after 3 tries", flush=True)
    print("Saved media files to %s", out_dir)
    print(f"[sync] Done. Files in {out_dir}", flush=True)
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
    """
    base_dir = _project_tmp_dir(project_id)
    out_path = os.path.join(base_dir, "localizations.jsonl")
    print(f"[sync] Localizations JSONL will be saved to: {out_path}", flush=True)
    batch_size = LOCALIZATION_BATCH_SIZE

    def _count_kwargs(use_version: bool) -> dict:
        kwargs: dict = {}
        if media_ids:
            kwargs["media_id"] = media_ids
        if use_version and version_id is not None:
            kwargs["version"] = [version_id]
        return kwargs

    def _list_kwargs(after: int | None, use_version: bool) -> dict:
        kwargs: dict = {"stop": batch_size}
        if media_ids:
            kwargs["media_id"] = media_ids
        if use_version and version_id is not None:
            kwargs["version"] = [version_id]
        if after is not None:
            kwargs["after"] = after
        return kwargs

    use_version = True
    try:
        count_kwargs = _count_kwargs(use_version=True)
        loc_count = api.get_localization_count(project_id, **count_kwargs)
        print(f"[sync] get_localization_count(project_id={project_id}, media_ids={bool(media_ids)}, version={version_id}) = {loc_count}", flush=True)
        if loc_count == 0 and version_id is not None:
            # Fallback: requested version may have no localizations; try without version filter
            count_kwargs_no_ver = _count_kwargs(use_version=False)
            count_no_ver = api.get_localization_count(project_id, **count_kwargs_no_ver)
            if count_no_ver > 0:
                print(f"[sync] Version {version_id} has 0 localizations; falling back to all versions (count={count_no_ver})", flush=True)
                use_version = False
    except Exception as e:
        loc_count = None
        print(f"[sync] get_localization_count failed (will still try list): {e}", flush=True)

    total = 0
    after_id = None
    with open(out_path, "w") as f:
        while True:
            kwargs = _list_kwargs(after_id, use_version)
            print(f"[sync] get_localization_list(project_id={project_id}, {kwargs})", flush=True)
            try:
                batch = api.get_localization_list(project_id, **kwargs)
            except Exception as e:
                print(f"[sync] get_localization_list failed: {e}", flush=True)
                print("get_localization_list failed", flush=True)
                break
            if not batch:
                if total == 0 and version_id is not None and use_version:
                    # First batch empty with version filter; retry without version
                    print(f"[sync] First batch empty with version={version_id}; retrying without version filter", flush=True)
                    use_version = False
                    after_id = None
                    continue
                print(f"[sync] Localizations batch empty (after={after_id}), done", flush=True)
                break
            for loc in batch:
                try:
                    obj = loc.to_dict() if hasattr(loc, "to_dict") else loc
                    f.write(json.dumps(obj, default=_json_serial) + "\n")
                except Exception as e:
                    print("Skip localization serialization: %s", e)
                    continue
            total += len(batch)
            after_id = batch[-1].id if batch else None
            print(f"[sync] Localizations batch: count={len(batch)} total_so_far={total} last_id={after_id}", flush=True)
            if len(batch) < batch_size:
                break
    print("Fetched %s localizations -> %s", total, out_path)
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
        print("Crop failed for %s: %s", out_path.stem, e)
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
        print("Download dir or localizations JSONL missing; skipping crops")
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
                print("Crop task failed: %s", fut.exception())
            elif fut.result():
                num_ok += 1
            else:
                num_fail += 1
    print(f"[sync] Crops done: {num_ok} saved to {crops_path}, {num_fail} failed", flush=True)
    return (num_ok, num_fail)


def _load_config(path: str) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _sanitize_field_name(name: str) -> str:
    """Convert to a valid FiftyOne field name."""
    base = re.sub(r"[^a-zA-Z0-9]", "_", str(name))
    base = re.sub(r"_+", "_", base).strip("_").lower()
    return base or "unknown"


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
            sample["url"] = tator_url
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
    image_extensions: list[str],
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
            print(f"[sync] Reconcile: removed {len(to_remove)} samples (deleted in Tator)", flush=True)
    else:
        print(f"[sync] Reconcile: 0 localizations from Tator; skipping delete step (keeping existing samples)", flush=True)

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
        print(f"[sync] Reconcile: updated {updated} samples (box changed)", flush=True)

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
        print(f"[sync] Reconcile: added {added} new samples", flush=True)

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
    print(f"[sync] ----->loc_index={loc_index}", flush=True)
    print("Loaded %s localizations from JSONL", len(loc_index))
    print(f"[sync] Loaded {len(loc_index)} localizations from JSONL", flush=True)

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
            print(f"[sync] ----->loc={loc} api_url={api_url} project_id={project_id} version_id={version_id}", flush=True)
            if loc and api_url and project_id is not None:
                tator_url = _tator_localization_url(api_url, project_id, loc, version_id)
                if tator_url:
                    sample["url"] = tator_url
                    print(f"[sync] ----->tator_url={tator_url}", flush=True)
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

    print(f"[sync] Collected {len(samples)} samples for dataset", flush=True)

    # Handle existing dataset: always reconcile, never delete
    if dataset_name in fo.list_datasets():
        # Delete the dataset if it exists
        fo.delete_dataset(dataset_name)
        # print(f"[sync] --------->Reconcile: loading dataset {dataset_name}", flush=True)
        # dataset = fo.load_dataset(dataset_name)
        # dataset.persistent = True  # Ensure dataset persists in MongoDB after session ends
        # dataset = reconcile_dataset_with_tator(
        #     dataset=dataset,
        #     loc_index=loc_index,
        #     crops_dir=crops_dir,
        #     download_dir=download_dir,
        #     config=config,
        #     image_extensions=image_extensions,
        #     max_samples=max_samples,
        # )
        # print(f"[sync] --------->Reconcile: dataset {dataset_name} loaded", flush=True)
        # return dataset

    print(f"[sync] --------->Reconcile: creating new dataset {dataset_name}", flush=True)
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True  # Persist dataset in MongoDB after session ends
    dataset.add_samples(samples)
    print(f"[sync] Created dataset '{dataset_name}' with {len(samples)} samples", flush=True)
    return dataset


DEFAULT_LABEL_ATTR = "Label"
DEFAULT_SCORE_ATTR = "score"


def sync_edits_to_tator(
    project_id: int,
    version_id: int,
    api_url: str,
    token: str,
    dataset_name: str | None = None,
    database_name: str | None = None,
    label_attr: str | None = DEFAULT_LABEL_ATTR,
    score_attr: str | None = DEFAULT_SCORE_ATTR,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Push FiftyOne dataset edits (labels, confidence) back to Tator localizations.
    Uses elemental_id to match samples to Tator localizations; updates attributes
    via update_localization_by_elemental_id.
    Returns {"status": "ok", "updated": int, "failed": int, "errors": list} or raises.
    """
    fo.config.database_uri = get_database_uri()
    resolved_db = database_name or get_database_name(project_id, None)
    fo.config.database_name = resolved_db
    os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
    os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name

    ds_name = dataset_name or f"tator_project_{project_id}"
    fallback_db = f"{os.environ.get('FIFTYONE_DATABASE_DEFAULT', 'fiftyone_project')}_{project_id}"
    if ds_name not in fo.list_datasets():
        if resolved_db != fallback_db:
            fo.config.database_name = fallback_db
            os.environ["FIFTYONE_DATABASE_NAME"] = fallback_db
        if ds_name not in fo.list_datasets():
            fo.config.database_name = resolved_db
            raise ValueError(
                f"Dataset '{ds_name}' not found in database '{resolved_db}' (or '{fallback_db}'). "
                "Run POST /sync first. Ensure FIFTYONE_DATABASE_URI and FIFTYONE_DATABASE_NAME match the sync process."
            )

    dataset = fo.load_dataset(ds_name)
    host = api_url.rstrip("/")
    api = tator.get_api(host, token)

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
                print(f"[sync] SKIP elem={elemental_id} hash={current_hash[:8]}", flush=True)
            continue
        try:
            api.update_localization_by_elemental_id(
                version_id, str(elemental_id), localization_update={"attributes": attrs}
            )
            sample["last_sync_hash"] = current_hash
            sample.save()
            updated += 1
            if _debug:
                print(f"[sync] UPDATE elem={elemental_id}", flush=True)
        except Exception as e:
            failed += 1
            errors.append(f"{elemental_id}: {e}")
            print("Failed to update localization %s: %s", elemental_id, e)

    print(f"[sync] sync_edits_to_tator: updated={updated} skipped={skipped} failed={failed}", flush=True)
    return {"status": "ok", "updated": updated, "skipped": skipped, "failed": failed, "errors": errors[:20]}


def sync_project_to_fiftyone(
    project_id: int,
    version_id: int | None,
    api_url: str,
    token: str,
    port: int,
    database_name: str | None = None,
    config_path: str | None = None,
    launch_app: bool = True,
) -> dict[str, Any]:
    """
    Fetch Tator media and localizations, build FiftyOne dataset, launch App on given port.
    Uses per-project MongoDB database (resolved via database_name override or get_database_name).
    Returns {"status": "ok", "dataset_name": str, "database_name": str} or raises.
    """
    print(f"[sync] sync_project_to_fiftyone CALLED: project_id={project_id} version_id={version_id} api_url={api_url} port={port}", flush=True)
    sess = get_session(project_id)
    resolved_db = (
        (database_name.strip() if database_name and database_name.strip() else None)
        or (sess.get("database_name") if sess else None)
        or get_database_name(project_id, None)
    )
    fo.config.database_uri = get_database_uri()
    fo.config.database_name = resolved_db
    print(f"[sync] database_uri={fo.config.database_uri} database_name={resolved_db}", flush=True)

    media_ids_list: list[int] = []
    tmp_dir = ""
    localizations_path = ""
    crops_dir = ""
    try:
        host = api_url.rstrip("/")
        api = tator.get_api(host, token)
        print(f"[sync] Fetching media IDs... {host} {token} {project_id} {api_url}", flush=True)
        media_ids_list = fetch_project_media_ids(api_url, token, project_id)
        if not media_ids_list:
            print(f"[sync] No media IDs for project {project_id}", flush=True)
            print("No media IDs found for project %s; skipping download", project_id)
        else:
            print("media_ids:", media_ids_list, flush=True)
            print(f"[sync] Getting media objects in chunks...", flush=True)
            all_media = get_media_chunked(api, project_id, media_ids_list)
            if not all_media:
                print(f"[sync] No Media objects for {len(media_ids_list)} ids", flush=True)
                print("No Media objects returned for %s ids; skipping download", len(media_ids_list))
            else:
                print(f"[sync] Saving {len(all_media)} media to tmp...", flush=True)
                tmp_dir = save_media_to_tmp(api, project_id, all_media)
                if tmp_dir:
                    print("saved_media_dir:", tmp_dir, flush=True)
        print(f"[sync] Fetching localizations...", flush=True)
        localizations_path = fetch_and_save_localizations(
            api, project_id, version_id=version_id, media_ids=media_ids_list or None
        )
        if localizations_path:
            print(f"[sync] saved_localizations_path (JSONL): {localizations_path}", flush=True)
        if tmp_dir and localizations_path:
            crops_dir = os.path.join(_project_tmp_dir(project_id), "crops")
            crop_localizations_parallel(tmp_dir, localizations_path, crops_dir, size=224)
    except Exception as e:
        print("fetch/save media or localizations failed: %s", e)
        print(f"[sync] Error: {e}", flush=True)
        return {
            "status": "error",
            "message": str(e),
            "database_name": resolved_db,
            "saved_media_dir": tmp_dir or None,
            "saved_localizations_path": localizations_path or None,
            "saved_crops_dir": crops_dir or None,
        }

    if not crops_dir or not localizations_path:
        print(f"[sync] No crops or localizations; skipping dataset build", flush=True)
        return {
            "status": "ok",
            "message": "No crops to load; media/localizations missing or empty",
            "database_name": resolved_db,
            "dataset_name": None,
            "saved_media_dir": tmp_dir or None,
            "saved_localizations_path": localizations_path or None,
            "saved_crops_dir": crops_dir or None,
        }

    # Load config from YAML/JSON if provided
    config: dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            config = _load_config(config_path)
            print(f"[sync] Loaded config from {config_path}", flush=True)
        except Exception as e:
            print("Failed to load config %s: %s", config_path, e)
            print(f"[sync] Config load failed: {e}", flush=True)

    # Inject Tator base URL and ids so sample "url" can link to the localization's media page
    config["source_url"] = api_url.rstrip("/")
    config["project_id"] = project_id
    config["version_id"] = version_id

    dataset_name = config.get("dataset_name") or f"tator_project_{project_id}" 

    # Set env so FiftyOne app subprocess uses the same database
    os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
    os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name

    try:
        print(f"[sync] --------->Building dataset {dataset_name}", flush=True)
        dataset = build_fiftyone_dataset_from_crops(
            crops_dir=crops_dir,
            localizations_jsonl_path=localizations_path,
            dataset_name=dataset_name,
            config=config,
            download_dir=tmp_dir or None,
        )
    except Exception as e:
        print("Dataset build failed: %s", e)
        print(f"[sync] Dataset build failed: {e}", flush=True)
        return {
            "status": "error",
            "message": str(e),
            "database_name": resolved_db,
            "dataset_name": None,
            "saved_media_dir": tmp_dir or None,
            "saved_localizations_path": localizations_path or None,
            "saved_crops_dir": crops_dir or None,
        }

    print(f"[sync] sync_project_to_fiftyone done: dataset={dataset_name}", flush=True)
    print(
        "sync_project_to_fiftyone: project=%s port=%s database=%s dataset=%s",
        project_id, port, resolved_db, dataset_name,
    )

    sample_count = len(dataset)
    print(f"[sync] Dataset '{dataset_name}' has {sample_count} samples", flush=True)

    if launch_app:
        # Reload dataset from MongoDB so the server (which loads from DB) sees the same state
        try:
            dataset.reload()
            sample_count = len(dataset)
            print(f"[sync] Reloaded dataset from DB: {sample_count} samples", flush=True)
        except Exception as e:
            logging.getLogger(__name__).debug("Dataset reload before launch: %s", e)
        print(f"[sync] Launching FiftyOne app on port {port}...", flush=True)
        # _stop_process_on_port(port)
        session = _launch_app_embedded(dataset, port)
        # Push state immediately so the server has the correct dataset
        try:
            session.refresh()
        except Exception as e:
            logging.getLogger(__name__).debug("Session refresh after launch: %s", e)
        # Give the FiftyOne server time to start and accept state before the browser connects
        time.sleep(10)
        try:
            session.refresh()
        except Exception as e:
            logging.getLogger(__name__).debug("Session refresh (delayed): %s", e)

    return {
        "status": "ok",
        "dataset_name": dataset_name,
        "database_name": resolved_db,
        "sample_count": sample_count,
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
    localizations_path = fetch_and_save_localizations(
        api, project_id, version_id=version_id, media_ids=media_ids if media_ids else None
    )
    if localizations_path:
        print("saved_localizations_path (JSONL):", localizations_path)
    base_dir = _project_tmp_dir(project_id)
    download_dir = os.path.join(base_dir, "download")
    crops_dir = os.path.join(base_dir, "crops")
    if os.path.isdir(download_dir) and localizations_path:
        crop_localizations_parallel(download_dir, localizations_path, crops_dir, size=224)

    if crops_dir and localizations_path and os.path.isdir(crops_dir):
        fo.config.database_uri = get_database_uri()
        fo.config.database_name = get_database_name(project_id, None)
        os.environ["FIFTYONE_DATABASE_URI"] = fo.config.database_uri
        os.environ["FIFTYONE_DATABASE_NAME"] = fo.config.database_name
        config_path = os.getenv("CONFIG_PATH")
        config = _load_config(config_path) if config_path and os.path.exists(config_path) else {}
        dataset_name = config.get("dataset_name") or f"tator_project_{project_id}"
        dataset = build_fiftyone_dataset_from_crops(
            crops_dir=crops_dir,
            localizations_jsonl_path=localizations_path,
            dataset_name=dataset_name,
            config=config,
            download_dir=download_dir,
        )
        port = get_port_for_project(project_id)
        print(f"[sync] Launching FiftyOne app on port {port}...", flush=True)
        _stop_process_on_port(port)
        session = _launch_app_embedded(dataset, port)
        print(f"[sync] Session: {session}", flush=True)


if __name__ == "__main__":
    main()
