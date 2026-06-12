"""
morpho_utils.py
===============
Reusable helpers for exploring the ION x CTX neuronal morphology dataset
(`data/ion_ctx/data_v2/<sample_id>/<NNN>.swc`).

The SWC files are single-neuron reconstructions registered to the Allen Mouse
Common Coordinate Framework (CCFv3), so XYZ coordinates are in microns within
the ~13200 x 8000 x 11400 um CCF volume.

Key utilities
-------------
- ``find_swc_files`` / ``build_file_index`` : discover files on disk.
- ``load_swc``                              : read one SWC into a tidy DataFrame.
- ``reindex_swc``                           : fix the non-consecutive PointNo
                                              indices (e.g. 1,2,3,...,16,214,...)
                                              by renumbering to 1..N and remapping
                                              the parent column to match.
- ``compute_morphometrics``                 : per-neuron summary features.
- ``plot_neuron_2d`` / ``plot_neuron_3d``   : visualization.

All functions take/return plain pandas DataFrames so they compose easily in
notebooks.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# Resolve the data root relative to this file so notebooks work regardless of
# the current working directory.
_THIS_DIR = Path(__file__).resolve().parent
# code/notebooks/ -> capsule root -> data/...
DATA_ROOT = (_THIS_DIR / ".." / ".." / "data" / "ion_ctx" / "data_v2").resolve()

# SWC point-type codes (Neuronland / Allen convention).
SWC_TYPE_NAMES = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal dendrite",
    4: "apical dendrite",
    5: "custom",
    6: "neurite",
    7: "glia",
}

# Standard color per type for consistent plots.
SWC_TYPE_COLORS = {
    0: "#7f7f7f",  # undefined - grey
    1: "#000000",  # soma      - black
    2: "#1f77b4",  # axon      - blue
    3: "#d62728",  # basal dend- red
    4: "#ff7f0e",  # apical    - orange
    5: "#9467bd",
    6: "#2ca02c",
    7: "#8c564b",
}

# Approximate CCFv3 full-volume extent in microns (for plot bounds / sanity).
CCF_EXTENT_UM = {"x": (0, 13200), "y": (0, 8000), "z": (0, 11400)}

SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_swc_files(data_root: os.PathLike | str | None = None) -> list[str]:
    """Return a sorted list of all .swc file paths under ``data_root``."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return sorted(glob.glob(str(root / "*" / "*.swc")))


def build_file_index(data_root: os.PathLike | str | None = None) -> pd.DataFrame:
    """Build a DataFrame index of every SWC file.

    Columns: ``sample_id`` (parent folder), ``neuron_id`` (file stem, e.g.
    "041"), ``filename``, ``path``, ``size_bytes``.
    """
    paths = find_swc_files(data_root)
    rows = []
    for p in paths:
        p = Path(p)
        rows.append(
            {
                "sample_id": p.parent.name,
                "neuron_id": p.stem,
                "filename": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sample_id", "neuron_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Loading & index repair
# ---------------------------------------------------------------------------

def load_swc(path: os.PathLike | str, reindex: bool = True) -> pd.DataFrame:
    """Load a single SWC file into a tidy DataFrame.

    Parameters
    ----------
    path : path to the .swc file.
    reindex : if True (default), renumber PointNo to consecutive 1..N and remap
        the parent column. The dataset's raw files have gapped indices
        (e.g. ...,15,16,214,...) which this repairs. See :func:`reindex_swc`.

    Returns
    -------
    DataFrame with columns id, type, x, y, z, radius, parent.
    A boolean attr ``df.attrs['was_reindexed']`` records whether the raw file
    had non-consecutive indices.
    """
    # SWC is whitespace-delimited; comments start with '#'.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=SWC_COLUMNS,
        dtype={"id": np.int64, "type": np.int64, "parent": np.int64},
    )
    df["type"] = df["type"].astype(np.int64)

    had_gaps = not _is_consecutive(df["id"].to_numpy())
    if reindex and had_gaps:
        df = reindex_swc(df)
    df.attrs["was_reindexed"] = bool(reindex and had_gaps)
    df.attrs["had_gaps"] = bool(had_gaps)
    df.attrs["source_path"] = str(path)
    return df


def _is_consecutive(ids: np.ndarray) -> bool:
    """True if ids are exactly 1, 2, ..., len(ids)."""
    n = len(ids)
    if n == 0:
        return True
    return ids[0] == 1 and np.array_equal(ids, np.arange(1, n + 1))


def reindex_swc(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber node ids to consecutive 1..N and remap parents accordingly.

    The raw SWC files keep the original (gapped) PointNo values after some
    points were pruned, so ids may look like 1,2,3,...,16,214,215,...  The
    parent column references these original ids, so we must remap *both* the
    id and parent columns through the same old->new mapping to preserve the
    tree topology.  Root nodes keep parent == -1.

    Row order is preserved (parents already precede children in these files),
    so the remap is a pure relabeling and does not reorder the tree.
    """
    df = df.reset_index(drop=True).copy()
    old_ids = df["id"].to_numpy()
    new_ids = np.arange(1, len(df) + 1, dtype=np.int64)
    id_map = dict(zip(old_ids.tolist(), new_ids.tolist()))
    id_map[-1] = -1  # parent sentinel for roots

    df["id"] = new_ids
    df["parent"] = df["parent"].map(id_map).astype(np.int64)
    return df


def write_swc(df: pd.DataFrame, path: os.PathLike | str,
              header: str | None = None) -> None:
    """Write a neuron DataFrame back out as a standard SWC file.

    Useful for persisting the index-repaired version to disk (e.g. into
    ``results/``). Expects columns id, type, x, y, z, radius, parent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = SWC_COLUMNS
    with open(path, "w") as fh:
        fh.write("# SWC format file\n")
        fh.write("# PointNo Label X Y Z Radius Parent\n")
        if header:
            for line in header.splitlines():
                fh.write(f"# {line}\n")
        for rec in df[cols].itertuples(index=False):
            fh.write(
                f"{int(rec.id)} {int(rec.type)} {rec.x:g} {rec.y:g} "
                f"{rec.z:g} {rec.radius:g} {int(rec.parent)}\n"
            )


def load_many(
    paths, reindex: bool = True, add_source: bool = True
) -> pd.DataFrame:
    """Load multiple SWC files and concatenate into one long DataFrame.

    Adds ``sample_id`` and ``neuron_id`` columns identifying each file. Node
    ids are local to each neuron (after reindexing), so always group by
    (sample_id, neuron_id) before doing per-node graph work.
    """
    frames = []
    for p in paths:
        p = Path(p)
        d = load_swc(p, reindex=reindex)
        if add_source:
            d.insert(0, "neuron_id", p.stem)
            d.insert(0, "sample_id", p.parent.name)
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["sample_id", "neuron_id", *SWC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Morphometrics
# ---------------------------------------------------------------------------

def _parent_row_index(df: pd.DataFrame) -> np.ndarray:
    """Return an array mapping each row to its parent's row index (-1 for roots)."""
    ids = df["id"].to_numpy()
    # Fast path: ids are consecutive 1..N (true after reindex), so the parent's
    # row index is simply parent-1; roots (-1) map to -1.
    if len(ids) and ids[0] == 1 and ids[-1] == len(ids):
        prow = df["parent"].to_numpy() - 1
        prow[df["parent"].to_numpy() == -1] = -1
        return prow.astype(np.int64)
    # General path via id->row mapping.
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    prow = df["parent"].map(row_of).to_numpy()
    prow = np.where(np.isnan(prow), -1, prow).astype(np.int64)
    return prow


def _path_length(df: pd.DataFrame) -> float:
    """Total cable length: sum of euclidean distances from each node to its parent."""
    coords = df[["x", "y", "z"]].to_numpy()
    prow = _parent_row_index(df)
    mask = prow != -1
    if not mask.any():
        return 0.0
    diffs = coords[mask] - coords[prow[mask]]
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def compute_morphometrics(df: pd.DataFrame) -> dict:
    """Compute summary morphometric features for a single neuron DataFrame.

    Returns a dict of scalar features:
      n_nodes, n_branch_points, n_tips, n_soma_nodes,
      total_length_um, max_path_radius_um (max dist of any node from soma),
      bbox_x/y/z_um (bounding-box extents), soma_x/y/z (CCF coords),
      n_axon_nodes, n_dend_nodes, n_undef_nodes.
    """
    n = len(df)
    parent = df["parent"].to_numpy()
    ids = df["id"].to_numpy()

    # Branch points = nodes appearing >1 time as a parent; tips = nodes that are
    # never a parent.
    parent_counts = pd.Series(parent[parent != -1]).value_counts()
    n_branch = int((parent_counts > 1).sum())
    n_tips = int((~np.isin(ids, parent_counts.index.to_numpy())).sum())

    types = df["type"].to_numpy()
    coords = df[["x", "y", "z"]].to_numpy()

    soma_mask = types == 1
    if soma_mask.any():
        soma_xyz = coords[soma_mask].mean(axis=0)
    else:
        soma_xyz = coords.mean(axis=0)

    dists = np.linalg.norm(coords - soma_xyz, axis=1)
    bbox = coords.max(axis=0) - coords.min(axis=0)

    return {
        "n_nodes": n,
        "n_branch_points": n_branch,
        "n_tips": n_tips,
        "n_soma_nodes": int(soma_mask.sum()),
        "n_axon_nodes": int((types == 2).sum()),
        "n_dend_nodes": int(np.isin(types, [3, 4]).sum()),
        "n_undef_nodes": int((types == 0).sum()),
        "total_length_um": _path_length(df),
        "max_path_radius_um": float(dists.max()),
        "mean_radius_um": float(df["radius"].mean()),
        "bbox_x_um": float(bbox[0]),
        "bbox_y_um": float(bbox[1]),
        "bbox_z_um": float(bbox[2]),
        "soma_x": float(soma_xyz[0]),
        "soma_y": float(soma_xyz[1]),
        "soma_z": float(soma_xyz[2]),
    }


def morphometrics_table(file_index: pd.DataFrame, reindex: bool = True,
                        progress: bool = True) -> pd.DataFrame:
    """Compute morphometrics for every file in a file index DataFrame.

    ``file_index`` must have columns sample_id, neuron_id, path (see
    :func:`build_file_index`). Returns one row per neuron.
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    rows = []
    it = file_index.itertuples(index=False)
    if progress:
        it = tqdm(it, total=len(file_index), desc="morphometrics")
    for rec in it:
        df = load_swc(rec.path, reindex=reindex)
        feats = compute_morphometrics(df)
        feats["sample_id"] = rec.sample_id
        feats["neuron_id"] = rec.neuron_id
        rows.append(feats)
    out = pd.DataFrame(rows)
    front = ["sample_id", "neuron_id"]
    return out[front + [c for c in out.columns if c not in front]]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _segments_by_type(df: pd.DataFrame, axes: tuple[int, int]):
    """Yield (type, line_xs, line_ys) parent->child segments for a 2D projection.

    ``axes`` selects coordinate columns, e.g. (0, 1) for an XY projection.
    """
    coords = df[["x", "y", "z"]].to_numpy()
    id_to_row = {int(i): r for r, i in enumerate(df["id"].to_numpy())}
    parent = df["parent"].to_numpy()
    types = df["type"].to_numpy()
    a, b = axes
    by_type: dict[int, list] = {}
    for r in range(len(df)):
        p = int(parent[r])
        if p == -1:
            continue
        pr = id_to_row.get(p)
        if pr is None:
            continue
        seg = [(coords[pr, a], coords[pr, b]), (coords[r, a], coords[r, b])]
        by_type.setdefault(int(types[r]), []).append(seg)
    return by_type


def plot_neuron_2d(df: pd.DataFrame, ax=None, projection: str = "xy",
                   linewidth: float = 0.4, mark_soma: bool = True,
                   title: str | None = None, equal: bool = True):
    """Plot a single neuron as a 2D projection colored by point type.

    projection : one of 'xy', 'xz', 'yz'.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in axis_map:
        raise ValueError(f"projection must be one of {list(axis_map)}")
    a, b = axis_map[projection]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    by_type = _segments_by_type(df, (a, b))
    for t, segs in by_type.items():
        lc = LineCollection(
            segs, colors=SWC_TYPE_COLORS.get(t, "#333333"),
            linewidths=linewidth,
            label=f"{SWC_TYPE_NAMES.get(t, t)} (n={len(segs)})",
        )
        ax.add_collection(lc)

    if mark_soma:
        soma = df[df["type"] == 1]
        if len(soma):
            ax.scatter(soma.iloc[:, :][["x", "y", "z"]].to_numpy()[:, a],
                       soma[["x", "y", "z"]].to_numpy()[:, b],
                       s=40, c="black", marker="o", zorder=5, label="soma")

    ax.autoscale()
    if equal:
        ax.set_aspect("equal")
    labels = ["x", "y", "z"]
    ax.set_xlabel(f"{labels[a]} (um, CCF)")
    ax.set_ylabel(f"{labels[b]} (um, CCF)")
    ax.set_title(title or "")
    ax.legend(fontsize=7, loc="best", framealpha=0.6)
    return ax


def plot_neuron_3d(df: pd.DataFrame, ax=None, linewidth: float = 0.4,
                   title: str | None = None):
    """Plot a single neuron in 3D (matplotlib), colored by point type."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

    coords = df[["x", "y", "z"]].to_numpy()
    id_to_row = {int(i): r for r, i in enumerate(df["id"].to_numpy())}
    parent = df["parent"].to_numpy()
    types = df["type"].to_numpy()
    by_type: dict[int, list] = {}
    for r in range(len(df)):
        p = int(parent[r])
        if p == -1:
            continue
        pr = id_to_row.get(p)
        if pr is None:
            continue
        by_type.setdefault(int(types[r]), []).append([coords[pr], coords[r]])

    for t, segs in by_type.items():
        lc = Line3DCollection(segs, colors=SWC_TYPE_COLORS.get(t, "#333333"),
                              linewidths=linewidth,
                              label=SWC_TYPE_NAMES.get(t, str(t)))
        ax.add_collection3d(lc)

    soma = df[df["type"] == 1]
    if len(soma):
        s = soma[["x", "y", "z"]].to_numpy()
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], c="black", s=40, label="soma")

    mn = coords.min(0)
    mx = coords.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    try:
        ax.set_box_aspect(mx - mn)
    except Exception:
        pass
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)"); ax.set_zlabel("z (um)")
    ax.set_title(title or "")
    ax.legend(fontsize=7)
    return ax





# Read SWC -> build (Indexed, Root, Succeeding) + ancestor matrix
# Quantize parent-relative Δ to bins + residuals
from typing import Dict, List, Tuple
import numpy as np
import torch

SWC_TYPE_SOMA = 1
SWC_TYPE_AXON = 2
SWC_TYPE_DEND = 3  # (3 basal / 4 apical → merge)

def parse_swc(path: str):
    """
    Returns dict:
      nodes: List[dict(id,type,x,y,z,r,parent_id)]
      id2idx: map from original id to 0..N-1 in input order
      parent: list of parent indices (-1 for soma)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            toks = line.split()
            if len(toks) < 7: continue
            nid = int(toks[0]); tp = int(toks[1])
            x, y, z = map(float, toks[2:5])
            r = float(toks[5]); pid = int(toks[6])
            rows.append((nid, tp, x, y, z, r, pid))
    rows.sort(key=lambda t: t[0])
    id2row = {r[0]: r for r in rows}
    # assume root has parent -1 
    N = len(rows)
    id2idx = {r[0]: i for i, r in enumerate(rows)}
    parent = [-1]*N
    for i, r in enumerate(rows):
        pid = r[6]
        parent[i] = -1 if pid == -1 else id2idx[pid]
    return rows, id2idx, parent

def map_type(tp: int) -> int:
    if tp == SWC_TYPE_SOMA: return 0
    if tp == SWC_TYPE_AXON: return 2
    return 1  # dendrite (merge 3/4)

# ---------------------------------------------------------------------------
# Tree downsampling utilities
# ---------------------------------------------------------------------------

def _rebuild_tree(rows, keep_set, old_parent):
    """
    Given original *rows* and *old_parent* (index-based), keep only indices in
    *keep_set* and return new (rows, id2idx, parent) with contiguous 0..M-1
    indexing.  Parent pointers are re-wired: for every kept node, walk up
    old_parent until we hit another kept node (or -1).
    """
    keep = sorted(keep_set)
    old2new = {old: new for new, old in enumerate(keep)}
    new_rows = []
    new_parent = []
    for old_idx in keep:
        r = rows[old_idx]
        # re-wire parent to nearest kept ancestor
        p = old_parent[old_idx]
        while p != -1 and p not in old2new:
            p = old_parent[p]
        new_pid_idx = -1 if p == -1 else old2new[p]
        # build a new row with a fresh sequential id (1-based) and updated parent id
        new_id = len(new_rows) + 1
        new_pid = -1 if new_pid_idx == -1 else new_pid_idx + 1
        new_rows.append((new_id, r[1], r[2], r[3], r[4], r[5], new_pid))
        new_parent.append(new_pid_idx)
    new_id2idx = {r[0]: i for i, r in enumerate(new_rows)}
    return new_rows, new_id2idx, new_parent


def collapse_extra_soma(rows, id2idx, parent):
    """
    Remove non-root soma nodes (SWC type == 1) that are direct children of
    another soma node.  Their children are re-parented to the root soma.
    Returns (rows, id2idx, parent) in the same format as parse_swc.
    """
    N = len(rows)
    # build children
    children = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            children[parent[i]].append(i)

    root = next(i for i in range(N) if parent[i] == -1)

    # find redundant soma nodes: type==1, not root, parent is also type==1
    remove = set()
    for i in range(N):
        if i == root:
            continue
        if rows[i][1] == SWC_TYPE_SOMA and parent[i] >= 0 and rows[parent[i]][1] == SWC_TYPE_SOMA:
            remove.add(i)

    if not remove:
        return rows, id2idx, parent

    # re-parent children of removed nodes to their (kept) parent
    # (walk up until we find a non-removed ancestor)
    new_parent = list(parent)
    for rm in remove:
        for ch in children[rm]:
            p = parent[rm]
            while p in remove:
                p = parent[p]
            new_parent[ch] = p

    keep = set(range(N)) - remove
    return _rebuild_tree(rows, keep, new_parent)


def downsample_tree(rows, id2idx, parent, factor: int = 3):
    """
    Down-sample chain (pass-through) nodes by keeping every *factor*-th node.
    Always preserves:
      - root node (parent == -1)
      - branch nodes (>= 2 children)
      - leaf nodes (0 children)
      - first and last node of every chain (to keep junction geometry accurate)
    A chain is a maximal sequence of 1-child nodes whose first member's parent
    is the root or a branch node.

    factor=1 means no downsampling (all nodes kept).
    Returns (rows, id2idx, parent) in the same format as parse_swc.
    """
    if factor <= 1:
        return rows, id2idx, parent

    N = len(rows)
    children = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            children[parent[i]].append(i)

    n_children = [len(ch) for ch in children]
    root = next(i for i in range(N) if parent[i] == -1)

    # classify nodes
    is_chain = [False] * N
    for i in range(N):
        if i == root:
            continue
        if n_children[i] == 1:
            is_chain[i] = True

    # extract maximal chains
    # a chain starts at the first 1-child node after a root/branch node
    visited = [False] * N
    chains = []  # list of lists of node indices
    for i in range(N):
        if not is_chain[i] or visited[i]:
            continue
        # check that this node's parent is root or a branch (>= 2 children) or is root
        p = parent[i]
        if p == -1 or not is_chain[p]:
            # start of a new chain
            chain = []
            cur = i
            while cur != -1 and is_chain[cur] and not visited[cur]:
                visited[cur] = True
                chain.append(cur)
                # move to the single child
                cur = children[cur][0] if n_children[cur] == 1 else -1
            chains.append(chain)

    # decide which chain nodes to keep
    keep = set()
    # always keep root, branch, leaf nodes
    for i in range(N):
        if not is_chain[i]:
            keep.add(i)

    for chain in chains:
        L = len(chain)
        if L <= factor:
            # short chain: keep endpoints only
            keep.add(chain[0])
            keep.add(chain[-1])
        else:
            # keep first, last, and every factor-th from the start
            keep.add(chain[0])
            keep.add(chain[-1])
            for idx in range(factor, L - 1, factor):
                keep.add(chain[idx])

    return _rebuild_tree(rows, keep, parent)


def downsample_swc(rows, id2idx, parent, factor: int = 3, collapse_soma: bool = True):
    """
    Convenience wrapper: optionally collapse extra soma nodes, then
    down-sample chain nodes by *factor*.
    Returns (rows, id2idx, parent).
    """
    if collapse_soma:
        rows, id2idx, parent = collapse_extra_soma(rows, id2idx, parent)
    if factor > 1:
        rows, id2idx, parent = downsample_tree(rows, id2idx, parent, factor=factor)
    return rows, id2idx, parent


# Maps the model's internal type codes (0=soma, 1=dendrite, 2=axon, 3=other)
# back to standard SWC types.
_INTERNAL_TO_SWC_TYPE = {0: SWC_TYPE_SOMA, 1: SWC_TYPE_DEND, 2: SWC_TYPE_AXON, 3: 6}
_DEFAULT_RADIUS_TABLE = (0.1, 0.3, 0.5, 0.8, 1.2)


def write_swc(xyz, types, radii, parent_indices, filename,
              radius_table=_DEFAULT_RADIUS_TABLE,
              type_map=_INTERNAL_TO_SWC_TYPE, comment="Generated by TreeFormer"):
    """Write a generated tree to an SWC file.

    Args:
        xyz: (N, 3) array-like of node coordinates.
        types: length-N array of internal type codes.
        radii: length-N array of radius bin indices (looked up in `radius_table`).
        parent_indices: length-N list; -1 for root, else index of parent.
    """
    import numpy as _np

    xyz_np = _np.asarray(xyz, dtype=float)
    types_np = _np.asarray(types, dtype=int)
    radii_np = _np.asarray(radii, dtype=int)
    radius_lut = _np.asarray(radius_table, dtype=float)
    radius_values = radius_lut[_np.clip(radii_np, 0, len(radius_lut) - 1)]

    with open(filename, "w") as f:
        f.write("# id type x y z radius parent\n")
        f.write(f"# {comment}\n")
        for i in range(len(xyz_np)):
            parent = parent_indices[i] + 1 if parent_indices[i] >= 0 else -1
            swc_type = type_map.get(int(types_np[i]), 0)
            f.write(
                f"{i + 1} {swc_type} "
                f"{xyz_np[i, 0]:.6f} {xyz_np[i, 1]:.6f} {xyz_np[i, 2]:.6f} "
                f"{radius_values[i]:.6f} {parent}\n"
            )
    return filename


# ---------------------------------------------------------------------------
# Training-time augmentations
# ---------------------------------------------------------------------------

def augment_rows(rows, parent, *, rotate_z: bool = False, mirror_xy: bool = False,
                 prune_p: float = 0.0, prune_max_branch_len: int = 0, rng=None):
    """Apply geometric + topological augmentations to (rows, parent).

    - rotate_z: random rotation of (x, y) about the soma's (x, y) by θ ~ U[0, 2π).
    - mirror_xy: independently flip x and y about the soma with prob 0.5 each.
    - prune_p / prune_max_branch_len: drop terminal branches (leaf -> nearest
      bifurcation/soma) of length <= prune_max_branch_len, each with prob prune_p.
      Never drops the only child of the soma.

    Returns (rows, id2idx, parent) in the same format as parse_swc.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(rows)
    if N == 0:
        return rows, {r[0]: i for i, r in enumerate(rows)}, parent

    # locate soma (root)
    soma = next(i for i in range(N) if parent[i] == -1)
    cx, cy = rows[soma][2], rows[soma][3]

    # --- geometric transforms (build new rows) ---
    if rotate_z:
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
    else:
        cos_t, sin_t = 1.0, 0.0
    flip_x = -1.0 if (mirror_xy and rng.random() < 0.5) else 1.0
    flip_y = -1.0 if (mirror_xy and rng.random() < 0.5) else 1.0

    new_rows = []
    for r in rows:
        x, y, z = r[2], r[3], r[4]
        # translate so soma is at origin, transform, translate back
        dx, dy = x - cx, y - cy
        dx *= flip_x
        dy *= flip_y
        rx = cos_t * dx - sin_t * dy + cx
        ry = sin_t * dx + cos_t * dy + cy
        new_rows.append((r[0], r[1], rx, ry, z, r[5], r[6]))
    rows = new_rows

    # --- topological pruning of small terminal branches ---
    if prune_p > 0.0 and prune_max_branch_len > 0:
        children = [[] for _ in range(N)]
        for i in range(N):
            if parent[i] >= 0:
                children[parent[i]].append(i)

        leaves = [i for i in range(N) if not children[i] and i != soma]
        remove = set()
        for leaf in leaves:
            # walk up until we reach a node with >1 child (bifurcation) or the soma
            branch_nodes = [leaf]
            cur = leaf
            while True:
                p = parent[cur]
                if p == -1 or p == soma or len(children[p]) > 1:
                    break
                branch_nodes.append(p)
                cur = p
            if len(branch_nodes) > prune_max_branch_len:
                continue
            # don't make the soma childless
            top = branch_nodes[-1]
            if parent[top] == soma and len([c for c in children[soma] if c not in remove]) <= 1:
                continue
            if rng.random() < prune_p:
                remove.update(branch_nodes)

        if remove:
            keep = set(range(N)) - remove
            rows, id2idx, parent = _rebuild_tree(rows, keep, parent)
            return rows, id2idx, parent

    id2idx = {r[0]: i for i, r in enumerate(rows)}
    return rows, id2idx, parent


def build_sequences_from_swc(path: str, phys_bin_size: float = 2.0,
                             bins: int = 21, delta_min: float = -10.0, bin_width: float = 1.0,
                             order: str = "rand_index", rng_seed: int = 42, bias_p: float = 0.7,
                             downsample_factor: int = 1, collapse_soma: bool = False,
                             augment: dict | None = None):
    """
    Build (Indexed, Root, Succeeding) sequences and ancestor matrix from an SWC file.

    order:
      - "rand_index": stepwise random-indexing (ONE child per step) with depth-first bias `bias_p`
      - "bfs":        original grouped-BFS (emit ALL children of a popped root)
    """
    rows, id2idx, parent = parse_swc(path)

    # Some SWC files contain disconnected components (multiple root nodes).
    # Keep only the largest connected component so the tree invariant holds.
    _roots = [i for i in range(len(rows)) if parent[i] == -1]
    if len(_roots) > 1:
        _ch = [[] for _ in range(len(rows))]
        for i in range(len(rows)):
            if parent[i] >= 0:
                _ch[parent[i]].append(i)
        _best_keep, _best_size = set(), 0
        for _r in _roots:
            _q = [_r]; _vis = {_r}
            while _q:
                _cur = _q.pop()
                for _c in _ch[_cur]:
                    _vis.add(_c); _q.append(_c)
            if len(_vis) > _best_size:
                _best_keep, _best_size = _vis, len(_vis)
        rows, id2idx, parent = _rebuild_tree(rows, _best_keep, parent)

    # optional downsampling
    if collapse_soma or downsample_factor > 1:
        rows, id2idx, parent = downsample_swc(
            rows, id2idx, parent,
            factor=downsample_factor, collapse_soma=collapse_soma)

    # optional augmentation (rotation, mirror, terminal-branch pruning)
    if augment:
        rows, id2idx, parent = augment_rows(
            rows, parent,
            rotate_z=bool(augment.get("rotate_z", False)),
            mirror_xy=bool(augment.get("mirror_xy", False)),
            prune_p=float(augment.get("prune_p", 0.0)),
            prune_max_branch_len=int(augment.get("prune_max_branch_len", 0)),
            rng=np.random.default_rng(),
        )

    N = len(rows)

    # children list
    children = [[] for _ in range(N)]
    for i in range(N):
        p = parent[i]
        if p >= 0:
            children[p].append(i)

    # soma index (root)
    soma = next(i for i, _ in enumerate(rows) if parent[i] == -1)

    rng = np.random.default_rng(rng_seed)

    # ---- step 1: create X (indexed order) and RS (root, succ) pairs ----
    if order == "rand_index":
        # Stepwise emission; maintain remaining children per node
        remaining = [list(ch) for ch in children]

        X = [soma]
        RS = []  # list of (root_node_index, succ_node_index)
        avail = [soma] if remaining[soma] else []
        recent = soma if remaining[soma] else None
        p_recent = float(bias_p)

        while avail:
            # choose a root r
            if recent is not None and remaining[recent] and rng.random() < p_recent:
                r = recent
            else:
                if recent is not None:
                    candidates = [a for a in avail if a != recent]
                    if candidates:
                        r = candidates[int(rng.integers(0, len(candidates)))]
                    else:
                        r = recent
                else:
                    r = avail[int(rng.integers(0, len(avail)))]

            # emit exactly ONE child of r
            c_list = remaining[r]
            ci = int(rng.integers(0, len(c_list)))
            s = c_list.pop(ci)

            RS.append((r, s))
            X.append(s)

            # update avail/recent
            if not remaining[r]:
                avail.remove(r)
                if recent == r:
                    recent = None
            if remaining[s]:
                avail.append(s)
                recent = s

        # sanity: tree edges = nodes - 1
        assert len(RS) == len(X) - 1 == N - 1

    elif order == "bfs":
        # original grouped-BFS behavior (all children at once)
        X = [soma]
        queue = [soma]
        RS = []
        while queue:
            r = queue.pop(0)
            for s in children[r]:
                RS.append((r, s))
                X.append(s)
                queue.append(s)
        assert len(RS) == N - 1
    else:
        raise ValueError(f"Unknown order='{order}'. Use 'rand_index' or 'bfs'.")

    # ---- step 2: ancestor matrix (self-included) ----
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        j = i
        while j != -1:
            A[i, j] = True
            j = parent[j]

    # ---- step 3: helpers and attributes ----
    def quantize(delta_phys: float):
        # parent-relative delta in "bin units"
        delta_units = delta_phys / phys_bin_size
        c = np.clip(np.round((delta_units - delta_min) / bin_width).astype(int), 0, bins - 1)
        center = delta_min + c * bin_width
        half = 0.5 - 1e-3 # this 1e-3 is also used in the TokenHead_TypeRad_XYZ function (in heads.py) as "residual_eps"
        r = np.clip((delta_units - center)/bin_width, -half, half)
        #r = np.clip(delta_units - center, -0.5 * bin_width, 0.5 * bin_width) / bin_width
        return int(c), float(r)

    # absolute coordinates and attributes
    xyz = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
    typ = np.array([map_type(r[1]) for r in rows], dtype=int)
    rad = np.array([r[5] for r in rows], dtype=float)

    # simple radius bins (replace with dataset quantiles if desired)
    thr = np.array([0.5, 1.0, 1.5, 2.0])  # μm
    rad_bin = np.digitize(rad, thr).clip(0, 4)

    # ---- step 4: build token sequences ----
    center_bin = int(np.clip(np.round((0 - delta_min) / bin_width), 0, bins - 1))

    indexed = {k: [] for k in ["dx", "dy", "dz", "type", "radius", "branch"]}
    roots   = {k: [] for k in ["dx", "dy", "dz", "type", "radius", "branch"]}
    succ    = {k: [] for k in ["dx", "dy", "dz", "type", "radius", "branch", "dx_res", "dy_res", "dz_res"]}

    # branch capacity = #children capped at 5
    cap = np.array([min(5, len(children[i])) for i in range(N)], dtype=int)

    # fill Indexed (in X order)
    for node in X:
        if node == soma:
            dx = dy = dz = center_bin
        else:
            p = parent[node]
            d = xyz[node] - xyz[p]
            dx, _ = quantize(d[0]); dy, _ = quantize(d[1]); dz, _ = quantize(d[2])
        indexed["dx"].append(int(dx)); indexed["dy"].append(int(dy)); indexed["dz"].append(int(dz))
        indexed["type"].append(int(typ[node]))
        indexed["radius"].append(int(rad_bin[node]))
        indexed["branch"].append(int(cap[node]))

    # precompute node -> position in X for O(1) lookup
    pos_in_X = {node: i for i, node in enumerate(X)}

    # fill aligned (R,S)
    for (r, s) in RS:
        rpos = pos_in_X[r]
        roots["dx"].append(indexed["dx"][rpos])
        roots["dy"].append(indexed["dy"][rpos])
        roots["dz"].append(indexed["dz"][rpos])
        roots["type"].append(int(typ[r]))
        roots["radius"].append(int(rad_bin[r]))
        roots["branch"].append(int(cap[r]))

        p = parent[s]
        d = xyz[s] - xyz[p]
        cx, rx = quantize(d[0]); cy, ry = quantize(d[1]); cz, rz = quantize(d[2])
        succ["dx"].append(int(cx)); succ["dy"].append(int(cy)); succ["dz"].append(int(cz))
        succ["dx_res"].append(float(rx)); succ["dy_res"].append(float(ry)); succ["dz_res"].append(float(rz))
        succ["type"].append(int(typ[s])); succ["radius"].append(int(rad_bin[s])); succ["branch"].append(int(cap[s]))

    # ---- step 5: to tensors ----
    def to_long(d):  return {k: torch.tensor(v, dtype=torch.long) for k, v in d.items()}
    indexed = to_long(indexed)
    roots   = to_long(roots)

    succ_long = {k: torch.tensor(succ[k], dtype=torch.long)
                 for k in ["dx", "dy", "dz", "type", "radius", "branch"]}
    succ_res  = {k: torch.tensor(succ[k], dtype=torch.float32)
                 for k in ["dx_res", "dy_res", "dz_res"]}
    succ = {**succ_long, **succ_res}

    adj = torch.tensor(A, dtype=torch.bool)
    return indexed, roots, succ, adj




from typing import Dict, List
import torch
from torch.utils.data import Dataset
from data.swc_to_lists import build_sequences_from_swc

PAD = -100

class SWCDataset(Dataset):
    def __init__(self, swc_paths: List[str], bins: int = 21, delta_min: float = -10.0, bin_width: float = 1.0,
                 phys_bin_size: float = 2.0, order: str = "bfs",
                 downsample_factor: int = 1, collapse_soma: bool = False,
                 augment: dict | None = None):
        self.paths = swc_paths
        self.kw = dict(bins=bins, delta_min=delta_min, bin_width=bin_width,
                       phys_bin_size=phys_bin_size, order=order,
                       downsample_factor=downsample_factor, collapse_soma=collapse_soma,
                       augment=augment)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        indexed, roots, succ, adj = build_sequences_from_swc(self.paths[i], **self.kw)
        return {"indexed": indexed, "roots": roots, "succ": succ, "adj": adj}

def _pad_field(seq_list, key, dtype=torch.long, pad=PAD):
    max_len = max(x[key].shape[0] for x in seq_list)
    out = torch.full((len(seq_list), max_len), pad if dtype==torch.long else 0.0, dtype=dtype)
    for b, x in enumerate(seq_list):
        L = x[key].shape[0]
        out[b, :L] = x[key]
    return out

def collate_batch(batch: List[Dict]):
    # pad indexed, roots, succ (long), succ residuals (float), adj to same N
    idxs = [b["indexed"] for b in batch]
    rts  = [b["roots"] for b in batch]
    sucl = [{k:v for k,v in b["succ"].items() if v.dtype==torch.long} for b in batch]
    sucf = [{k:v for k,v in b["succ"].items() if v.dtype==torch.float32} for b in batch]
    adjs = [b["adj"] for b in batch]

    indexed = {k: _pad_field(idxs, k, dtype=torch.long) for k in idxs[0].keys()}
    roots   = {k: _pad_field(rts,  k, dtype=torch.long) for k in rts[0].keys()}
    succ    = {k: _pad_field(sucl, k, dtype=torch.long) for k in sucl[0].keys()}
    # float residuals
    for k in sucf[0].keys():
        succ[k] = _pad_field(sucf, k, dtype=torch.float32, pad=0.0)

    # pad adj to same N (right/bottom with False)
    maxN = max(a.shape[0] for a in adjs)
    B = len(adjs)
    adj = torch.zeros((B, maxN, maxN), dtype=torch.bool)
    for b, a in enumerate(adjs):
        n = a.shape[0]
        adj[b, :n, :n] = a

    return {"indexed": indexed, "roots": roots, "succ": succ, "adj": adj}


def collate_fn_padded(batch, max_seq_len=None):
    """Collate function for SWCDataset with padding and optional truncation.

    Pads `indexed` to max_T, and `roots`/`succ` to max_T - 1 across the batch.
    If `max_seq_len` is set, samples longer than it are truncated first.
    Returns CPU tensors; the training loop is responsible for moving to device.
    """
    PAD_TOKEN = 0  # 0 is a valid embedding index; do not use -100 here.

    if max_seq_len is not None:
        for sample in batch:
            T_i = sample['indexed']['dx'].shape[0]
            if T_i > max_seq_len:
                for k in list(sample['indexed'].keys()):
                    sample['indexed'][k] = sample['indexed'][k][:max_seq_len]
                for k in list(sample['roots'].keys()):
                    sample['roots'][k] = sample['roots'][k][:max_seq_len - 1]
                for k in list(sample['succ'].keys()):
                    sample['succ'][k] = sample['succ'][k][:max_seq_len - 1]
                sample['adj'] = sample['adj'][:max_seq_len, :max_seq_len]

    seq_lengths = [sample['indexed']['dx'].shape[0] for sample in batch]
    max_T = max(seq_lengths)
    bsz = len(batch)
    token_keys = ["dx", "dy", "dz", "type", "radius", "branch"]

    residual_keys = [k for k in batch[0]['succ'].keys() if k not in token_keys]

    indexed_padded = {k: torch.full((bsz, max_T), PAD_TOKEN, dtype=torch.long) for k in token_keys}
    roots_padded = {k: torch.full((bsz, max_T - 1), PAD_TOKEN, dtype=torch.long) for k in token_keys}
    succ_padded = {k: torch.full((bsz, max_T - 1), PAD_TOKEN, dtype=torch.long) for k in token_keys}
    for k in residual_keys:
        succ_padded[k] = torch.zeros((bsz, max_T - 1), dtype=torch.float)

    adj_padded = torch.zeros((bsz, max_T, max_T), dtype=torch.bool)

    for i, sample in enumerate(batch):
        T_i = seq_lengths[i]
        for k in token_keys:
            indexed_padded[k][i, :T_i] = sample['indexed'][k]
        if T_i > 1:
            for k in token_keys:
                roots_padded[k][i, :T_i - 1] = sample['roots'][k]
                succ_padded[k][i, :T_i - 1] = sample['succ'][k]
            for k in residual_keys:
                succ_padded[k][i, :T_i - 1] = sample['succ'][k]
        adj_padded[i, :T_i, :T_i] = sample['adj']

    return {
        'indexed': indexed_padded,
        'roots': roots_padded,
        'succ': succ_padded,
        'adj': adj_padded,
    }    