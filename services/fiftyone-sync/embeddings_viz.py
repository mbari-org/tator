"""
Compute embeddings and UMAP visualization for FiftyOne datasets with caching.

Embeddings are fetched from the embed service at {base}/embed/{project}
where project is typically the Tator project ID (sync passes str(project_id) by default; config can override).
UMAP requires umap-learn (see requirements.txt).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import fiftyone as fo

# Base URL for embed service (POST /embed/{project} no trailing slash, GET /predict/job/{job_id}/{project})
EMBED_SERVICE_BASE_URL = os.environ.get(
    "FASTVSS_API_URL", "http://doris.shore.mbari.org:8000"
).rstrip("/")


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
            print(f"[embeddings] Skipping missing file: {fp}", flush=True)
            continue
        with open(fp, "rb") as f:
            data = f.read()
        filepaths.append(fp)
        bytes_list.append(data)

    if not bytes_list:
        print("[embeddings] No valid image files to embed", flush=True)
        return

    # Process in batches
    all_embeddings = []
    with httpx.Client(timeout=60.0) as client:
        for start in range(0, len(bytes_list), batch_size):
            print(f"[embeddings] Processing batch {start // batch_size + 1} of {len(bytes_list) // batch_size}")
            batch = bytes_list[start : start + batch_size]
            files = [("files", (f"img_{i}.jpg", data)) for i, data in enumerate(batch)]
            url = f"{base}/embed/{project_name}"
            resp = client.post(url, files=files)
            resp.raise_for_status()
            data = resp.json()
            print(f"[embeddings] Response: {data}")

            job_id = data.get("job_id")
            if job_id:
                print(f"[embeddings] Polling Job ID: {job_id}")
                # Poll GET /predict/job/{job_id}/{project_name} until status == "done", then use result.embeddings
                poll_url = f"{base}/predict/job/{job_id}/{project_name}"
                deadline = time.monotonic() + poll_timeout
                while time.monotonic() < deadline:
                    r = client.get(poll_url)
                    r.raise_for_status()
                    out = r.json()
                    print(f'[embeddings] Poll response: {out.get("status")}')
                    if out.get("status") == "done":
                        result = out.get("result") or {}
                        emb = result.get("embeddings") or out.get("embeddings")
                        if isinstance(emb, list) and len(emb) > 0:
                            print(f"[embeddings] Embeddings: {emb}")
                            all_embeddings.extend(
                                emb if isinstance(emb[0], (list, tuple)) else [emb]
                            )
                            break
                    time.sleep(poll_interval)
                else:
                    raise TimeoutError(
                        f"Embed service did not return status=done within {poll_timeout}s"
                    ) 

    if len(all_embeddings) != len(filepaths):
        print(
            f"[embeddings] Warning: got {len(all_embeddings)} embeddings for {len(filepaths)} images",
            flush=True,
        )

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
    print(f"✓ Embeddings stored in: {embeddings_field} ({len(fp_to_emb)} samples)", flush=True)


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

    print(f"\n{'='*80}")
    print("Embeddings from service:", f"{base_url}/embed/{{project_name}}")
    print(f"  Project name: {project_name}")
    print(f"  Embeddings field: {embeddings_field}")
    print(f"  Brain key: {brain_key}")
    print(f"{'='*80}")

    # --- Embeddings (from service) ---
    embeddings_exist = has_embeddings(dataset, embeddings_field)
    if embeddings_exist and not force_embeddings:
        print(f"\n✓ Embeddings already cached in '{embeddings_field}' - skipping computation")
        print("  (use force_embeddings in config to recompute)")
    else:
        if not project_name:
            raise ValueError(
                "Embeddings from service require project_name (Tator project name from get_project(project_id).name)"
            )
        if embeddings_exist and force_embeddings:
            print("\n⟳ Force recomputing embeddings (cached embeddings will be overwritten)")

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
        print(
            "\n⚠ UMAP visualization skipped (install umap-learn). Embeddings are stored.",
            flush=True,
        )
        return

    # Only run UMAP on samples that have embeddings (avoid empty array error)
    view_with_emb = dataset.exists(embeddings_field)
    n_with_emb = view_with_emb.count()
    if n_with_emb == 0:
        print(
            "\n⚠ UMAP skipped: no samples have embeddings (need at least 1). Embeddings may be missing or failed.",
            flush=True,
        )
        return

    brain_run_exists = has_brain_run(dataset, brain_key)
    if brain_run_exists and not force_umap:
        print(
            f"\n✓ UMAP visualization already cached with brain key '{brain_key}' - skipping computation"
        )
        print("  (use force_umap in config to recompute)")
    else:
        if brain_run_exists and force_umap:
            print("\n⟳ Force recomputing UMAP (deleting existing brain run)")
            dataset.delete_brain_run(brain_key)

        print(f"\nComputing UMAP visualization ({n_with_emb} samples with embeddings)...")
        fob.compute_visualization(
            view_with_emb,
            embeddings=embeddings_field,
            brain_key=brain_key,
            method="umap",
            verbose=True,
            seed=umap_seed,
        )
        print(f"✓ Visualization stored with brain key: {brain_key}")