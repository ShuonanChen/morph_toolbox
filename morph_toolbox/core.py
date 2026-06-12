"""The :class:`Morphology` data structure.

A ``Morphology`` is a thin, convenient wrapper around the SWC node table
(a :class:`pandas.DataFrame` with columns ``id, type, x, y, z, radius,
parent``).  It validates the table, exposes the tree topology, and provides
a few lightweight summary statistics.  I/O lives in :mod:`morph_toolbox.io`
and :mod:`morph_toolbox.convert`; plotting lives in :mod:`morph_toolbox.viz`.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .constants import ROOT_PARENT, SWC_COLUMNS, type_name


class Morphology:
    """A single neuron reconstruction backed by an SWC node table.

    Parameters
    ----------
    nodes :
        A DataFrame with (at least) the columns ``id, type, x, y, z, radius,
        parent``.  Extra columns are preserved but ignored.
    name :
        Optional label (e.g. the source filename stem) for plots and repr.
    metadata :
        Optional free-form dict carried alongside the data (e.g. source path,
        coordinate space).

    The node table is stored on :attr:`nodes`.  It is *not* deep-copied, so
    pass a copy if you need to keep the original untouched.
    """

    def __init__(self, nodes: pd.DataFrame, name: str | None = None,
                 metadata: dict | None = None):
        missing = [c for c in SWC_COLUMNS if c not in nodes.columns]
        if missing:
            raise ValueError(
                f"node table is missing required SWC columns: {missing}")
        self.nodes = nodes.reset_index(drop=True)
        self.name = name
        self.metadata: dict = dict(metadata or {})
        self.validate()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_arrays(cls, id, type, x, y, z, radius, parent, **kw) -> "Morphology":
        """Build a Morphology from seven array-likes (one per SWC column)."""
        df = pd.DataFrame({
            "id": np.asarray(id, dtype=np.int64),
            "type": np.asarray(type, dtype=np.int64),
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "z": np.asarray(z, dtype=float),
            "radius": np.asarray(radius, dtype=float),
            "parent": np.asarray(parent, dtype=np.int64),
        })
        return cls(df, **kw)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Check basic SWC invariants; raise ``ValueError`` on violation.

        Verifies unique ids, that every non-root parent references an existing
        node, and that there is at least one root.  Does not require a single
        connected component (disconnected reconstructions are common).
        """
        ids = self.nodes["id"].to_numpy()
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("node ids are not unique")

        parents = self.nodes["parent"].to_numpy()
        id_set = set(ids.tolist())
        bad = [int(p) for p in parents if p != ROOT_PARENT and p not in id_set]
        if bad:
            raise ValueError(
                f"{len(bad)} node(s) reference a non-existent parent id, "
                f"e.g. {bad[:5]}")

        if not (parents == ROOT_PARENT).any():
            raise ValueError("no root node (parent == -1) found")

    # -- size / dunder ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return (f"<Morphology{label}: {len(self)} nodes, "
                f"{self.num_roots} root(s), {self.num_branch_points} branches, "
                f"{self.num_tips} tips>")

    # -- topology -----------------------------------------------------------

    @property
    def roots(self) -> np.ndarray:
        """Ids of all root nodes (parent == -1)."""
        return self.nodes.loc[self.nodes["parent"] == ROOT_PARENT, "id"].to_numpy()

    @property
    def num_roots(self) -> int:
        return int((self.nodes["parent"] == ROOT_PARENT).sum())

    def _parent_counts(self) -> pd.Series:
        p = self.nodes["parent"]
        return p[p != ROOT_PARENT].value_counts()

    @property
    def num_branch_points(self) -> int:
        """Number of nodes that are a parent to more than one node."""
        return int((self._parent_counts() > 1).sum())

    @property
    def num_tips(self) -> int:
        """Number of terminal nodes (never referenced as a parent)."""
        parents = set(self._parent_counts().index.tolist())
        return int(sum(1 for i in self.nodes["id"].to_numpy()
                       if i not in parents))

    def children_map(self) -> dict[int, list[int]]:
        """Map each node id to the list of its children's ids."""
        out: dict[int, list[int]] = {int(i): [] for i in self.nodes["id"]}
        for nid, pid in zip(self.nodes["id"].to_numpy(),
                            self.nodes["parent"].to_numpy()):
            if pid != ROOT_PARENT:
                out[int(pid)].append(int(nid))
        return out

    def _parent_row_index(self) -> np.ndarray:
        """Row index of each node's parent (-1 for roots)."""
        row_of = pd.Series(np.arange(len(self.nodes)),
                           index=self.nodes["id"].to_numpy())
        prow = self.nodes["parent"].map(row_of).to_numpy()
        prow = np.where(np.isnan(prow), -1, prow).astype(np.int64)
        return prow

    # -- geometry / summary -------------------------------------------------

    @property
    def coords(self) -> np.ndarray:
        """``(N, 3)`` array of XYZ coordinates."""
        return self.nodes[["x", "y", "z"]].to_numpy()

    @property
    def soma_coord(self) -> np.ndarray:
        """Mean coordinate of soma (type 1) nodes, or centroid if none."""
        soma = self.nodes[self.nodes["type"] == 1]
        if len(soma):
            return soma[["x", "y", "z"]].to_numpy().mean(axis=0)
        return self.coords.mean(axis=0)

    def total_length(self) -> float:
        """Total cable length: sum of parent->child segment lengths."""
        coords = self.coords
        prow = self._parent_row_index()
        mask = prow != -1
        if not mask.any():
            return 0.0
        diffs = coords[mask] - coords[prow[mask]]
        return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(min_xyz, max_xyz)`` of all node coordinates."""
        coords = self.coords
        return coords.min(axis=0), coords.max(axis=0)

    def type_counts(self) -> dict[str, int]:
        """Count of nodes per (named) SWC type."""
        counts = self.nodes["type"].value_counts().to_dict()
        return {type_name(t): int(n) for t, n in sorted(counts.items())}

    def summary(self) -> dict:
        """A dict of headline morphometrics for this neuron."""
        mn, mx = self.bounding_box()
        return {
            "name": self.name,
            "n_nodes": len(self),
            "n_roots": self.num_roots,
            "n_branch_points": self.num_branch_points,
            "n_tips": self.num_tips,
            "total_length": self.total_length(),
            "bbox_size": (mx - mn).tolist(),
            "soma_coord": self.soma_coord.tolist(),
            "type_counts": self.type_counts(),
        }

    def copy(self) -> "Morphology":
        """Return a deep copy of this morphology."""
        return Morphology(self.nodes.copy(deep=True), name=self.name,
                          metadata=dict(self.metadata))

    def iter_segments(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(type, parent_xyz, child_xyz)`` for every edge in the tree."""
        coords = self.coords
        prow = self._parent_row_index()
        types = self.nodes["type"].to_numpy()
        for r in range(len(self.nodes)):
            if prow[r] == -1:
                continue
            yield int(types[r]), coords[prow[r]], coords[r]
