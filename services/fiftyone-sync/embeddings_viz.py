"""
Compute embeddings and UMAP visualization for FiftyOne datasets with caching.

Embeddings are fetched from the embed service at {base}/embed/{project}
where project is typically the Tator project ID (sync passes str(project_id) by default; config can override).
UMAP requires umap-learn (see requirements.txt).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__) 

# Base URL for embed service (POST /embed/{project} no trailing slash, GET /predict/job/{job_id}/{project})
EMBED_SERVICE_BASE_URL = os.environ.get("FASTVSS_API_URL", "http://cortext.shore.mbari.org/vss").rstrip("/")

# Stop embedding run after this many failed fetch attempts
EMBEDDING_FETCH_MAX_RETRIES = 3


def has_embeddings(dataset: "fo.Dataset", embeddings_field: str) -> bool:
    """Return True if the dataset has the embeddings field and at least one sample has embeddings."""
    if not dataset.has_field(embeddings_field):
        return False
    return dataset.exists(embeddings_field).count() > 0


def has_brain_run(dataset: "fo.Dataset", brain_key: str) -> bool:
    """Return True if the dataset has a brain run with the given key."""
    return brain_key in dataset.list_brain_runs()


def _compute_embeddings_via_service(
    dataset: "fo.Dataset",
    project_name: str,
    embeddings_field: str,
    service_url: str,
    batch_size: int = 32,
    poll_interval: float = 2.0,
    poll_timeout: float = 300.0
) -> None:
    """
    Compute embeddings by sending sample images to the embed service and writing results to the dataset.

    Service: POST {service_url}/embed/{project_name} (no trailing slash) with files -> job_id
             then GET {service_url}/predict/job/{job_id}/{project_name} until embeddings returned.
    """
    import fiftyone as fo
    import httpx
    import numpy as np

    base = service_url.rstrip("/")
    samples = list(dataset.iter_samples(autosave=False))
    if not samples:
        return
 
    path_pairs = []
    for s in samples:
        if "local_filepath" in s:
            path_to_open = s["local_filepath"]
        else:
            continue
        if os.path.isfile(path_to_open):
            path_pairs.append((path_to_open, s["local_filepath"]))
    paths_to_open = [p for p, _ in path_pairs]
    sample_filepaths = [fp for _, fp in path_pairs]

    # Submit all batches, then poll all jobs
    num_batches = (len(paths_to_open) + batch_size - 1) // batch_size
    jobs: list[tuple[int, str]] = []

    logger.info(f"Num batches {num_batches}")
    with httpx.Client(timeout=5.0) as client:
        # Phase 1: submit every batch and collect job IDs (retry each batch up to EMBEDDING_FETCH_MAX_RETRIES)
        for start in range(0, len(paths_to_open), batch_size):
            batch_idx = start // batch_size
            batch_paths = paths_to_open[start : start + batch_size]
            files = []
            for fp in batch_paths:
                with open(fp, "rb") as f:
                    files.append(("files", (os.path.basename(fp), f.read())))
            url = f"{base}/embed/{project_name}"
            last_error = None
            for attempt in range(EMBEDDING_FETCH_MAX_RETRIES):
                try:
                    logger.info(f"Submitting batch {batch_idx + 1}/{num_batches}" + (f" (attempt {attempt + 1}/{EMBEDDING_FETCH_MAX_RETRIES})" if attempt else ""))
                    resp = client.post(url, files=files)
                    resp.raise_for_status()
                    data = resp.json()
                    err = data.get("error")
                    if err:
                        raise RuntimeError(f"Embed service error: {err}")
                    job_id = data.get("job_id")
                    if not job_id:
                        raise RuntimeError(f"No job_id in response: {data}")
                    jobs.append((batch_idx, job_id))
                    logger.info(f"Batch {batch_idx + 1} submitted -> job {job_id}")
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Batch {batch_idx + 1} submit attempt {attempt + 1}/{EMBEDDING_FETCH_MAX_RETRIES} failed: {e}")
            if last_error is not None:
                logger.error(
                    f"Embedding service failed after {EMBEDDING_FETCH_MAX_RETRIES} retries; stopping to avoid continuing for thousands of images. Last error: {last_error}"
                )
                raise RuntimeError(
                    f"Embedding fetch failed after {EMBEDDING_FETCH_MAX_RETRIES} retries: {last_error}"
                ) from last_error

        for batch_idx, job_id in jobs:
            poll_url = f"{base}/predict/job/{job_id}/{project_name}"
            for attempt in range(EMBEDDING_FETCH_MAX_RETRIES):
                try:
                    r = client.get(poll_url)
                    r.raise_for_status()
                    out = r.json()
                    status = out.get("status")
                    if status == "failed":
                        err = out.get("error", "unknown")
                        raise RuntimeError(f"Job failed: {err}")
                    if status == "done":
                        raw_result = out.get("result")
                        if not isinstance(raw_result, dict):
                            logger.warning(
                                f"Poll response 'result' is not a dict (type={type(raw_result).__name__}); {raw_result}"
                            )
                            break
                        emb = raw_result.get("embeddings")
                        start = batch_idx * batch_size
                        end = start + batch_size
                        for s, emb in zip(samples[start:end], emb):
                            s[embeddings_field] = emb
                        break
                    else:
                        time.sleep(poll_interval)
                except Exception as e:
                    logger.warning(
                        f"Poll batch {batch_idx + 1} attempt {attempt + 1}/{EMBEDDING_FETCH_MAX_RETRIES} failed: {e}"
                    )
                    break
 
        dataset.save()
    logger.info(f"Embeddings stored in: {embeddings_field} ({len(samples)} samples)")


def compute_embeddings_and_viz(
    dataset: "fo.Dataset",
    model_info: dict,
    umap_seed: int = 51,
    force_embeddings: bool = False,
    force_umap: bool = False,
    batch_size: Optional[int] = None,
    project_name: Optional[str] = None,
    service_url: Optional[str] = None
) -> None:
    """
    Compute embeddings and UMAP visualization with caching.

    Embeddings are fetched from the embed service at {service_url}/embed/{project_name},
    where project_name is the Tator project name (get_project(project_id).name).
    UMAP is computed locally and stored under brain_key.

    When is_enterprise is True, only local_filepath is passed to the embed service (sample filepath may be S3).

    Args:
        dataset: FiftyOne dataset
        model_info: Dict with embeddings_field, brain_key (and optionally path for local model; if
            project_name is set, the service is used instead).
        umap_seed: Random seed for UMAP
        force_embeddings: If True, recompute embeddings even if they exist
        force_umap: If True, recompute UMAP even if it exists
        batch_size: Batch size for embed service requests (default 32)
        project_name: Project key for embed service URL path (usually project ID; required when using service)
        service_url: Base URL for embed service (default FASTVSS_API_URL or http://localhost:8000)
    """
    import fiftyone as fo
    import fiftyone.brain as fob

    embeddings_field = model_info["embeddings_field"]
    brain_key = model_info["brain_key"]
    base_url = (service_url or EMBED_SERVICE_BASE_URL).rstrip("/")

    logger.info(f"Embeddings from service: {base_url}/embed/ | project={project_name} field={embeddings_field} brain_key={brain_key} batch_size={batch_size}")

    # --- Embeddings (from service) ---
    embeddings_exist = has_embeddings(dataset, embeddings_field)
    if embeddings_exist and not force_embeddings:
        logger.info(f"Embeddings already cached in '{embeddings_field}' - skipping computation (use force_embeddings to recompute)")
    else:
        if not project_name:
            raise ValueError(
                "Embeddings from service require project_name (Tator project name from get_project(project_id).name)"
            )
        if embeddings_exist and force_embeddings:
            logger.info("Force recomputing embeddings (cached embeddings will be overwritten)")

        _compute_embeddings_via_service(
            dataset,
            project_name=project_name,
            embeddings_field=embeddings_field,
            service_url=base_url,
            batch_size=batch_size or 32
        )

    # Reload so exists() and brain see the persisted embeddings
    dataset.reload()

    # --- UMAP (local) ---
    try:
        import umap  # noqa: F401
    except ImportError:
        logger.warning(
            "UMAP visualization skipped (install umap-learn). Embeddings are stored."
        )
        return

    # Only run UMAP on samples that have embeddings (avoid empty array error)
    view_with_emb = dataset.exists(embeddings_field)
    n_with_emb = view_with_emb.count()
    if n_with_emb == 0:
        logger.warning(
            "UMAP skipped: no samples have embeddings (need at least 1). Embeddings may be missing or failed."
        )
        return

    brain_run_exists = has_brain_run(dataset, brain_key)
    if brain_run_exists and not force_umap:
        logger.info(f"UMAP visualization already cached with brain key '{brain_key}' - skipping computation (use force_umap to recompute)")
    else:
        if brain_run_exists and force_umap:
            logger.info("Force recomputing UMAP (deleting existing brain run)")
            dataset.delete_brain_run(brain_key)

        logger.info(f"Computing UMAP visualization ({n_with_emb} samples with embeddings)...")
        fob.compute_visualization(
            view_with_emb,
            embeddings=embeddings_field,
            brain_key=brain_key,
            method="umap",
            verbose=True,
            seed=umap_seed,
        )
        logger.info(f"Visualization stored with brain key: {brain_key}")
