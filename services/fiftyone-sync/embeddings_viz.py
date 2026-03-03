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
    poll_timeout: float = 300.0,
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

    # Use local_filepath when present (is_enterprise/S3 mode) so the embed service can open files locally.
    # Skip samples with no path or remote URIs (s3://, http) since the embed service expects local files.
    path_pairs = []
    for s in samples:
        path_to_open = s["local_filepath"] if "local_filepath" in s else s.get("filepath")
        if path_to_open is None or not isinstance(path_to_open, (str, bytes, os.PathLike)):
            continue
        path_str = path_to_open if isinstance(path_to_open, str) else str(path_to_open)
        if path_str.startswith("s3://") or path_str.startswith("http://") or path_str.startswith("https://"):
            continue
        if os.path.isfile(path_to_open):
            path_pairs.append((path_to_open, s.get("filepath") or path_to_open))
    paths_to_open = [p for p, _ in path_pairs]
    sample_filepaths = [fp for _, fp in path_pairs]
    skipped = len(samples) - len(paths_to_open)
    if skipped:
        logger.warning(f"Skipping {skipped} missing file(s)")

    if not paths_to_open:
        logger.warning("No valid image files to embed")
        return

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
            for i, fp in enumerate(batch_paths):
                with open(fp, "rb") as f:
                    files.append(("files", (f"img_{i}.jpg", f.read())))
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

        # Phase 2: poll all jobs until every one is done (retry each poll up to EMBEDDING_FETCH_MAX_RETRIES)
        all_embeddings: list[list] = [[] for _ in range(num_batches)]
        pending = dict(jobs)  # batch_idx -> job_id
        deadline = time.monotonic() + poll_timeout

        while pending and time.monotonic() < deadline:
            still_pending = {}
            for batch_idx, job_id in pending.items():
                poll_url = f"{base}/predict/job/{job_id}/{project_name}"
                last_poll_error = None
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
                                if raw_result is not None:
                                    logger.warning(
                                        f"Poll response 'result' is not a dict (type={type(raw_result).__name__}); using top-level 'embeddings' if present"
                                    )
                                raw_result = {}
                            result = raw_result
                            emb = result.get("embeddings") or out.get("embeddings")
                            if isinstance(emb, list) and len(emb) > 0:
                                all_embeddings[batch_idx] = (
                                    emb if isinstance(emb[0], (list, tuple)) else [emb]
                                )
                            logger.info(f"Batch {batch_idx + 1} done ({len(all_embeddings[batch_idx])} vectors)")
                        else:
                            still_pending[batch_idx] = job_id
                        last_poll_error = None
                        break
                    except Exception as e:
                        last_poll_error = e
                        logger.warning(
                            f"Poll batch {batch_idx + 1} attempt {attempt + 1}/{EMBEDDING_FETCH_MAX_RETRIES} failed: {e}"
                        )
                if last_poll_error is not None:
                    logger.error(
                        f"Embedding service poll failed after {EMBEDDING_FETCH_MAX_RETRIES} retries; stopping to avoid continuing for thousands of images. Last error: {last_poll_error}"
                    )
                    raise RuntimeError(
                        f"Embedding fetch failed after {EMBEDDING_FETCH_MAX_RETRIES} poll retries: {last_poll_error}"
                    ) from last_poll_error
            pending = still_pending
            if pending:
                logger.info(f"{len(pending)}/{num_batches} batches still pending…")
                time.sleep(poll_interval)

        if pending:
            raise TimeoutError(
                f"Embed service: {len(pending)} batch(es) did not finish within {poll_timeout}s"
            )

    all_embeddings = [vec for batch_embs in all_embeddings for vec in batch_embs]

    if len(all_embeddings) != len(sample_filepaths):
        logger.warning(f"Got {len(all_embeddings)} embeddings for {len(sample_filepaths)} images")

    # Map back to samples by filepath (canonical sample filepath, not path_to_open); store as 1D numpy arrays so FiftyOne infers VectorField
    # (list of floats infers ListField, which brain/UMAP may not treat as embeddings)
    fp_to_emb = dict(zip(sample_filepaths, all_embeddings))
    if np is not None:
        # Bulk set via set_values so schema expands to VectorField; use sample id as key
        values_by_id = {}
        for s in samples:
            fp = s["filepath"]
            if fp in fp_to_emb:
                vec = fp_to_emb[fp]
                values_by_id[s.id] = np.asarray(vec, dtype=np.float64)
        if values_by_id:
            dataset.set_values(embeddings_field, values_by_id, key_field="id", expand_schema=True)
    else:
        for s in samples:
            fp = s["filepath"]
            if fp in fp_to_emb:
                s[embeddings_field] = fp_to_emb[fp]
            else:
                s[embeddings_field] = None
        dataset.save()
    logger.info(f"Embeddings stored in: {embeddings_field} ({len(fp_to_emb)} samples)")


def compute_embeddings_and_viz(
    dataset: "fo.Dataset",
    model_info: dict,
    umap_seed: int = 51,
    force_embeddings: bool = False,
    force_umap: bool = False,
    batch_size: Optional[int] = None,
    project_name: Optional[str] = None,
    service_url: Optional[str] = None,
) -> None:
    """
    Compute embeddings and UMAP visualization with caching.

    Embeddings are fetched from the embed service at {service_url}/embed/{project_name},
    where project_name is the Tator project name (get_project(project_id).name).
    UMAP is computed locally and stored under brain_key.

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
            batch_size=batch_size or 32,
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
