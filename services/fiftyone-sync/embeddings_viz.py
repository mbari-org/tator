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
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import fiftyone as fo

# Base URL for embed service (POST /embed/{project} no trailing slash, GET /predict/job/{job_id}/{project})
EMBED_SERVICE_BASE_URL = os.environ.get("FASTVSS_API_URL", "http://cortext.shore.mbari.org/vss").rstrip("/")


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

    try:
        import httpx
    except ImportError:
        raise ImportError("Embeddings via service require httpx. Install with: pip install httpx") from None

    base = service_url.rstrip("/")
    samples = list(dataset.iter_samples(autosave=False))
    if not samples:
        return

    # Collect image bytes for all samples (filepath -> image bytes)
    filepaths = []
    bytes_list = []
    for s in samples:
        fp = s["filepath"]
        if not os.path.isfile(fp):
            logger.warning(f"Skipping missing file: {fp}")
            continue
        with open(fp, "rb") as f:
            data = f.read()
        filepaths.append(fp)
        bytes_list.append(data)

    if not bytes_list:
        logger.warning("No valid image files to embed")
        return

    # Submit all batches, then poll all jobs
    num_batches = (len(bytes_list) + batch_size - 1) // batch_size
    jobs: list[tuple[int, str]] = []  # (batch_index, job_id)

    with httpx.Client(timeout=5.0) as client:
        # Phase 1: submit every batch and collect job IDs
        for start in range(0, len(bytes_list), batch_size):
            batch_idx = start // batch_size
            logger.info(f"Submitting batch {batch_idx + 1}/{num_batches}")
            batch = bytes_list[start : start + batch_size]
            files = [("files", (f"img_{i}.jpg", data)) for i, data in enumerate(batch)]
            url = f"{base}/embed/{project_name}"
            resp = client.post(url, files=files)
            resp.raise_for_status()
            data = resp.json()

            err = data.get("error")
            if err:
                logger.error(f"Embed service returned error for batch {batch_idx + 1}: {err}")
                return

            job_id = data.get("job_id")
            if not job_id:
                logger.error(f"No job_id in response for batch {batch_idx + 1}: {data}")
                return
            jobs.append((batch_idx, job_id))
            logger.info(f"Batch {batch_idx + 1} submitted -> job {job_id}")

        # Phase 2: poll all jobs until every one is done
        all_embeddings: list[list] = [[] for _ in range(num_batches)]
        pending = dict(jobs)  # batch_idx -> job_id
        deadline = time.monotonic() + poll_timeout

        while pending and time.monotonic() < deadline:
            still_pending = {}
            for batch_idx, job_id in pending.items():
                poll_url = f"{base}/predict/job/{job_id}/{project_name}"
                r = client.get(poll_url)
                r.raise_for_status()
                out = r.json()
                status = out.get("status")
                if status == "done":
                    result = out.get("result") or {}
                    emb = result.get("embeddings") or out.get("embeddings")
                    if isinstance(emb, list) and len(emb) > 0:
                        all_embeddings[batch_idx] = (
                            emb if isinstance(emb[0], (list, tuple)) else [emb]
                        )
                    logger.info(f"Batch {batch_idx + 1} done ({len(all_embeddings[batch_idx])} vectors)")
                else:
                    still_pending[batch_idx] = job_id
            pending = still_pending
            if pending:
                logger.info(f"{len(pending)}/{num_batches} batches still pending…")
                time.sleep(poll_interval)

        if pending:
            raise TimeoutError(
                f"Embed service: {len(pending)} batch(es) did not finish within {poll_timeout}s"
            )

    all_embeddings = [vec for batch_embs in all_embeddings for vec in batch_embs]

    if len(all_embeddings) != len(filepaths):
        logger.warning(f"Got {len(all_embeddings)} embeddings for {len(filepaths)} images")

    # Map back to samples by filepath; store as 1D numpy arrays so FiftyOne infers VectorField
    # (list of floats infers ListField, which brain/UMAP may not treat as embeddings)
    try:
        import numpy as np
    except ImportError:
        np = None
    fp_to_emb = dict(zip(filepaths, all_embeddings))
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