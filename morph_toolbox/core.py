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


def _subtree(root: int, children: dict[int, list[int]]) -> list[int]:
    """Return all node ids in the subtree rooted at ``root`` (root included)."""
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children[n])
    return out


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

    def terminal_nodes(self, types: "int | list[int] | None" = None) -> pd.DataFrame:
        """Return the terminal tips (leaf nodes / branch ends) of the neuron.

        A tip is a node that is never referenced as anyone's parent -- i.e. the
        end of a branch.  This matches the ``num_tips`` count.

        Parameters
        ----------
        types :
            Optional SWC ``type`` code(s) to keep.  Pass ``2`` for axon
            terminals (the relevant tips for a projection-target map), ``[3, 4]``
            for dendritic tips, etc.

        Returns
        -------
        A copy of the subset of rows that are terminals (in node-table order).
        """
        parent = self.nodes["parent"].to_numpy()
        ids = self.nodes["id"].to_numpy()
        is_tip = ~np.isin(ids, parent[parent != ROOT_PARENT])
        out = self.nodes[is_tip]
        if types is not None:
            type_list = [types] if np.isscalar(types) else list(types)
            out = out[out["type"].isin(type_list)]
        return out.copy()

    def morphometrics(self) -> dict:
        """Compute a flat dict of per-neuron summary morphometric features.

        Unlike :meth:`summary` (which returns nested/structured values), this
        returns only scalar features convenient for tabulating across a
        population::

            n_nodes, n_branch_points, n_tips, n_soma_nodes, n_axon_nodes,
            n_dend_nodes, n_undef_nodes, total_length_um, max_path_radius_um
            (max distance of any node from the soma), mean_radius_um,
            bbox_x/y/z_um (bounding-box extents), soma_x/y/z (CCF coords).
        """
        nodes = self.nodes
        n = len(nodes)
        parent = nodes["parent"].to_numpy()
        ids = nodes["id"].to_numpy()

        # Branch points = nodes appearing >1 time as a parent; tips = nodes that
        # are never a parent.
        parent_counts = pd.Series(parent[parent != ROOT_PARENT]).value_counts()
        n_branch = int((parent_counts > 1).sum())
        n_tips = int((~np.isin(ids, parent_counts.index.to_numpy())).sum())

        types = nodes["type"].to_numpy()
        coords = self.coords
        soma_xyz = self.soma_coord
        dists = np.linalg.norm(coords - soma_xyz, axis=1) if n else np.zeros(0)
        mn, mx = self.bounding_box()
        bbox = mx - mn

        return {
            "n_nodes": n,
            "n_branch_points": n_branch,
            "n_tips": n_tips,
            "n_soma_nodes": int((types == 1).sum()),
            "n_axon_nodes": int((types == 2).sum()),
            "n_dend_nodes": int(np.isin(types, [3, 4]).sum()),
            "n_undef_nodes": int((types == 0).sum()),
            "total_length_um": self.total_length(),
            "max_path_radius_um": float(dists.max()) if n else 0.0,
            "mean_radius_um": float(nodes["radius"].mean()) if n else 0.0,
            "bbox_x_um": float(bbox[0]),
            "bbox_y_um": float(bbox[1]),
            "bbox_z_um": float(bbox[2]),
            "soma_x": float(soma_xyz[0]),
            "soma_y": float(soma_xyz[1]),
            "soma_z": float(soma_xyz[2]),
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

    # -- editing ------------------------------------------------------------

    def prune_branches(self, frac: float, rng=None, seed: int | None = None
                       ) -> "Morphology":
        """Remove whole branches at random until >= ``frac`` of nodes are gone.

        A *branch* is the subtree hanging off a child of a bifurcation (branch
        point).  Removing a branch deletes that whole subtree at once, so the
        remainder stays a valid connected tree -- mimicking how a real partial
        reconstruction is missing entire arbors rather than scattered points.
        The soma and the main trunk (root subtree) are never removed.

        Parameters
        ----------
        frac :
            Target fraction of nodes to remove (0..1).  Branches are removed in
            random order until at least this fraction is gone.
        rng :
            Optional ``numpy.random.Generator`` for reproducibility.  If omitted,
            one is created from ``seed``.
        seed :
            Seed used to build a generator when ``rng`` is not given.

        Returns
        -------
        A new, reindexed :class:`Morphology` of the surviving nodes.  Its
        ``metadata`` carries ``n_removed`` and ``frac_removed`` (the *actual*
        fraction removed, which can exceed ``frac`` because whole subtrees are
        removed at once).
        """
        from .io import reindex_nodes

        if rng is None:
            rng = np.random.default_rng(seed)

        nodes = self.nodes
        N = len(nodes)
        if frac <= 0:
            out = self.copy()
            out.metadata.update({"n_removed": 0, "frac_removed": 0.0})
            return out

        ch = self.children_map()
        soma = set(nodes.loc[nodes["type"] == 1, "id"].astype(int))
        root = int(nodes.loc[nodes["parent"] == ROOT_PARENT, "id"].iloc[0])
        bps = [n for n, c in ch.items() if len(c) > 1]

        cands = []  # (branch_root, subtree_set)
        for b in bps:
            for c in ch[b]:
                st = set(_subtree(c, ch))
                if not (st & soma) and root not in st:
                    cands.append((c, st))

        target = int(frac * N)
        removed: set[int] = set()
        for k in rng.permutation(len(cands)):
            _, st = cands[k]
            removed |= st
            if len(removed) >= target:
                break

        kept = reindex_nodes(nodes.loc[~nodes["id"].isin(removed)].copy())
        out = Morphology(kept, name=self.name, metadata=dict(self.metadata))
        out.metadata.update({
            "n_removed": len(removed),
            "frac_removed": len(removed) / N if N else 0.0,
        })
        return out
