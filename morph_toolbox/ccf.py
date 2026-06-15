"""Allen Mouse Common Coordinate Framework (CCFv3) brain-region annotation.

SWC reconstructions registered to the CCFv3 carry XYZ coordinates in microns,
so each node can be looked up against the Allen annotation volume to find which
brain region it sits in.  This module downloads and caches the annotation volume
and structure ontology, maps coordinates to region ids/acronyms/names, and
builds region "projection vectors" from a neuron's axon terminals.

This is an **optional** part of the toolbox: it needs the ``pynrrd`` package to
read the annotation volume.  Install it with ``pip install morph_toolbox[ccf]``
(or ``pip install pynrrd``).  The volume and ontology are downloaded once (from
the Allen Institute) and cached on disk.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CCF_EXTENT_UM  # noqa: F401  (re-exported for convenience)
from .core import Morphology

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allen Institute download URLs.  ``annotation_<res>.nrrd`` is the CCFv3 2017
# annotation volume; the structure graph is the region ontology.
CCF_ANNOTATION_URL = (
    "http://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/annotation_{res}.nrrd"
)
CCF_STRUCTURE_GRAPH_URL = (
    "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
)

# The annotation array axes are (anterior-posterior, dorsal-ventral,
# medial-lateral) == (x, y, z) of the registered SWC coordinates, so a micron
# coordinate maps to a voxel by ``floor(coord / resolution)`` with no axis
# swapping (x/y/z extents match 13200/8000/11400).

# Module-level caches so repeated lookups don't reload the volume / ontology.
_CCF_VOLUME_CACHE: dict = {}
_CCF_ONTOLOGY_CACHE: dict = {}


def _resolve_cache_dir(cache_dir) -> Path:
    if cache_dir is None:
        raise ValueError(
            "cache_dir is required: pass a directory to cache the CCF "
            "annotation volume and ontology (e.g. data/ccf_cache).")
    return Path(cache_dir)


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (atomic), creating parent dirs."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def load_ccf_ontology(cache_dir) -> dict:
    """Load the Allen structure ontology as a flat ``{id: node}`` dict.

    Each ``node`` is a dict with ``id``, ``acronym``, ``name``,
    ``parent_structure_id`` and ``structure_id_path`` (ancestor ids from root
    down to the structure).  Downloaded once and cached on disk
    (``structure_graph.json``) and in memory.
    """
    cache_dir = _resolve_cache_dir(cache_dir)
    key = str(cache_dir)
    if key in _CCF_ONTOLOGY_CACHE:
        return _CCF_ONTOLOGY_CACHE[key]

    path = cache_dir / "structure_graph.json"
    if not path.exists():
        _download(CCF_STRUCTURE_GRAPH_URL, path)

    with open(path) as fh:
        graph = json.load(fh)

    flat: dict[int, dict] = {}

    def _walk(node: dict, ancestors: list[int]) -> None:
        sid = int(node["id"])
        path_ids = ancestors + [sid]
        flat[sid] = {
            "id": sid,
            "acronym": node.get("acronym"),
            "name": node.get("name"),
            "parent_structure_id": (
                int(node["parent_structure_id"])
                if node.get("parent_structure_id") is not None
                else None
            ),
            "structure_id_path": path_ids,
        }
        for child in node.get("children", []) or []:
            _walk(child, path_ids)

    for root in graph["msg"]:
        _walk(root, [])

    _CCF_ONTOLOGY_CACHE[key] = flat
    return flat


def load_ccf_annotation(cache_dir, resolution: int = 25) -> np.ndarray:
    """Load the CCFv3 annotation volume as a ``uint32`` voxel->structure-id array.

    Parameters
    ----------
    cache_dir :
        Directory for the cached ``annotation_<res>.nrrd``.
    resolution :
        Voxel size in microns (10, 25, 50, or 100).  25 is a good accuracy/size
        trade-off (~4 MB); 10 is most precise (~80 MB).

    The array is indexed ``vol[i, j, k]`` with ``i = x // res`` (AP),
    ``j = y // res`` (DV), ``k = z // res`` (ML); value 0 means unannotated.
    Requires the ``pynrrd`` package (``pip install morph_toolbox[ccf]``).
    """
    cache_dir = _resolve_cache_dir(cache_dir)
    key = (str(cache_dir), int(resolution))
    if key in _CCF_VOLUME_CACHE:
        return _CCF_VOLUME_CACHE[key]

    try:
        import nrrd  # pynrrd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CCF annotation requires the 'pynrrd' package. "
            "Install it with: pip install morph_toolbox[ccf]"
        ) from exc

    path = cache_dir / f"annotation_{int(resolution)}.nrrd"
    if not path.exists():
        _download(CCF_ANNOTATION_URL.format(res=int(resolution)), path)

    vol, _ = nrrd.read(str(path))
    vol = np.asarray(vol)
    _CCF_VOLUME_CACHE[key] = vol
    return vol


def annotate_points(points, cache_dir, resolution: int = 25) -> pd.DataFrame:
    """Map XYZ coordinates (microns, CCFv3) to Allen brain regions.

    Parameters
    ----------
    points :
        ``(N, 3)`` array-like of ``[x, y, z]`` in microns, or a DataFrame with
        ``x``/``y``/``z`` columns (e.g. ``morph.nodes`` or its tip subset).
    cache_dir :
        Where the CCF files are cached.
    resolution :
        Annotation voxel size in microns (default 25).

    Returns
    -------
    DataFrame with one row per input point and columns ``structure_id`` (Allen
    id; 0 = unannotated / outside brain), ``acronym`` and ``name`` (None where
    unannotated).  Out-of-volume coordinates are treated as unannotated rather
    than raising, so whole neurons grazing the volume edge are safe to pass.
    """
    if isinstance(points, pd.DataFrame):
        xyz = points[["x", "y", "z"]].to_numpy(dtype=float)
    else:
        xyz = np.asarray(points, dtype=float)
        if xyz.ndim == 1:
            xyz = xyz[None, :]
    if xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3): [x, y, z] in microns")

    vol = load_ccf_annotation(cache_dir, resolution)
    onto = load_ccf_ontology(cache_dir)

    idx = np.floor(xyz / float(resolution)).astype(np.int64)
    in_bounds = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)

    sid = np.zeros(len(xyz), dtype=np.int64)
    ib = idx[in_bounds]
    sid[in_bounds] = vol[ib[:, 0], ib[:, 1], ib[:, 2]].astype(np.int64)

    acronyms = [onto.get(int(s), {}).get("acronym") for s in sid]
    names = [onto.get(int(s), {}).get("name") for s in sid]

    out = pd.DataFrame({"structure_id": sid, "acronym": acronyms, "name": names})
    if isinstance(points, pd.DataFrame):
        out.index = points.index
    return out


def annotate_region(x: float, y: float, z: float, cache_dir,
                    resolution: int = 25) -> dict:
    """Return the Allen brain region for a single ``(x, y, z)`` point (microns).

    Convenience wrapper around :func:`annotate_points`.  Returns a dict with
    ``structure_id``, ``acronym`` and ``name`` (the latter two None if the point
    is unannotated / outside the brain).
    """
    row = annotate_points([[x, y, z]], cache_dir, resolution).iloc[0]
    return {
        "structure_id": int(row["structure_id"]),
        "acronym": row["acronym"],
        "name": row["name"],
    }


def annotate_morphology(morph: Morphology, cache_dir,
                        resolution: int = 25) -> pd.DataFrame:
    """Return a copy of ``morph.nodes`` with brain-region columns added.

    Adds ``structure_id``, ``acronym`` and ``name`` columns by looking up every
    node's ``(x, y, z)`` in the CCF annotation -- useful for asking which region
    a neuron's soma sits in or how its axon is distributed across areas.
    """
    regions = annotate_points(morph.nodes, cache_dir, resolution)
    out = morph.nodes.copy()
    out["structure_id"] = regions["structure_id"].to_numpy()
    out["acronym"] = regions["acronym"].to_numpy()
    out["name"] = regions["name"].to_numpy()
    return out


def projection_vector(morph: Morphology, cache_dir, types: int = 2,
                      resolution: int = 25) -> pd.Series:
    """Build a normalized region histogram from a neuron's terminal tips.

    Maps the terminal nodes of the given SWC ``type`` (default ``2`` = axon) to
    Allen regions, drops tips that fall outside the annotated volume
    (``structure_id == 0``), and returns the fraction of tips terminating in
    each region.

    Returns
    -------
    A ``pandas.Series`` indexed by region acronym, summing to 1.0 (empty if the
    neuron has no annotated tips of the requested type).
    """
    tips = morph.terminal_nodes(types=types)
    regions = annotate_points(tips, cache_dir, resolution)
    labeled = regions[regions["structure_id"] != 0]
    counts = labeled["acronym"].value_counts()
    total = counts.sum()
    if total == 0:
        return pd.Series(dtype=float)
    return counts / total
