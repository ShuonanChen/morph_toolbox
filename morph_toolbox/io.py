"""Reading and writing SWC files.

SWC is a whitespace-delimited text format with one node per line and seven
columns: ``id type x y z radius parent``.  Lines beginning with ``#`` are
comments.  Real-world files often have *gapped* ids (e.g. after pruning), so
:func:`load_swc` can optionally renumber them to a contiguous 1..N range while
preserving topology.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ROOT_PARENT, SWC_COLUMNS
from .core import Morphology

PathLike = "os.PathLike | str"


def find_swc_files(data_root) -> list[str]:
    """Return a sorted list of all ``.swc`` file paths under ``data_root``.

    Files are matched one directory deep (``<data_root>/<sample>/<neuron>.swc``),
    the common one-folder-per-sample layout.  ``data_root`` is required -- the
    toolbox does not assume any particular dataset location.
    """
    root = Path(data_root)
    return sorted(glob.glob(str(root / "*" / "*.swc")))


def build_file_index(data_root) -> pd.DataFrame:
    """Build a DataFrame index of every SWC file under ``data_root``.

    Columns: ``sample_id`` (parent folder), ``neuron_id`` (file stem, e.g.
    "041"), ``filename``, ``path``, ``size_bytes``.
    """
    rows = []
    for p in find_swc_files(data_root):
        p = Path(p)
        rows.append({
            "sample_id": p.parent.name,
            "neuron_id": p.stem,
            "filename": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sample_id", "neuron_id"]).reset_index(drop=True)
    return df


def load_many(paths, reindex: bool = True) -> list[Morphology]:
    """Load multiple SWC files into a list of :class:`Morphology` objects.

    Each morphology's ``name`` is set to its file stem and its ``metadata``
    carries ``sample_id`` (parent folder) and ``neuron_id`` (stem) identifying
    the source file.
    """
    out = []
    for p in paths:
        p = Path(p)
        morph = load_swc(p, reindex=reindex)
        morph.metadata.update({"sample_id": p.parent.name, "neuron_id": p.stem})
        out.append(morph)
    return out


def _is_consecutive(ids: np.ndarray) -> bool:
    """True if ids are exactly 1, 2, ..., len(ids)."""
    n = len(ids)
    if n == 0:
        return True
    return ids[0] == 1 and np.array_equal(ids, np.arange(1, n + 1))


def reindex_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Renumber node ids to a contiguous 1..N and remap the parent column.

    Row order is preserved.  Parent references are remapped through the same
    old->new mapping so the tree topology is unchanged; roots keep parent -1.
    """
    nodes = nodes.reset_index(drop=True).copy()
    new_ids = np.arange(1, len(nodes) + 1, dtype=np.int64)
    id_map = dict(zip(nodes["id"].to_numpy().tolist(), new_ids.tolist()))
    id_map[ROOT_PARENT] = ROOT_PARENT
    nodes["id"] = new_ids
    nodes["parent"] = nodes["parent"].map(id_map).astype(np.int64)
    return nodes


def load_swc(path, reindex: bool = False, name: str | None = None) -> Morphology:
    """Load an SWC file into a :class:`Morphology`.

    Parameters
    ----------
    path :
        Path to the ``.swc`` file.
    reindex :
        If True, renumber gapped ids to a contiguous 1..N (see
        :func:`reindex_nodes`).  Defaults to False so the returned ids match
        the file; set True when downstream code assumes contiguous indices.
    name :
        Label for the morphology; defaults to the file stem.

    Returns
    -------
    Morphology
    """
    path = Path(path)
    nodes = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=SWC_COLUMNS,
        dtype={"id": np.int64, "type": np.int64, "parent": np.int64,
               "x": float, "y": float, "z": float, "radius": float},
    )

    had_gaps = not _is_consecutive(nodes["id"].to_numpy())
    if reindex and had_gaps:
        nodes = reindex_nodes(nodes)

    morph = Morphology(nodes, name=name or path.stem)
    morph.metadata.update({
        "source_path": str(path),
        "had_gaps": bool(had_gaps),
        "was_reindexed": bool(reindex and had_gaps),
    })
    return morph


def save_swc(morph: Morphology, path, header: str | None = None) -> Path:
    """Write a :class:`Morphology` to an SWC file.

    Parameters
    ----------
    morph : the morphology to write.
    path : destination path (parent directories are created).
    header : optional extra comment text (each line is prefixed with ``# ``).

    Returns the destination :class:`Path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = morph.nodes
    with open(path, "w") as fh:
        fh.write("# SWC file written by morph_toolbox\n")
        fh.write("# id type x y z radius parent\n")
        if morph.name:
            fh.write(f"# name: {morph.name}\n")
        if header:
            for line in str(header).splitlines():
                fh.write(f"# {line}\n")
        for rec in nodes[SWC_COLUMNS].itertuples(index=False):
            fh.write(
                f"{int(rec.id)} {int(rec.type)} {rec.x:g} {rec.y:g} "
                f"{rec.z:g} {rec.radius:g} {int(rec.parent)}\n"
            )
    return path
