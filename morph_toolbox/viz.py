"""2D and 3D visualization of a :class:`Morphology`.

Both plotters draw the neuron as parent->child line segments colored by SWC
point type (see :data:`morph_toolbox.constants.SWC_TYPE_COLORS`) and mark the
soma.  They accept an existing matplotlib axis so plots can be composed into
multi-panel figures, and otherwise create their own.
"""

from __future__ import annotations

import numpy as np

from .constants import type_color, type_name
from .core import Morphology

_AXIS_MAP = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _segments_by_type(morph: Morphology, axes: tuple[int, ...]):
    """Group projected parent->child segments by SWC type code.

    ``axes`` selects which coordinate columns to keep, e.g. ``(0, 1)`` for an
    XY projection or ``(0, 1, 2)`` for 3D.
    """
    by_type: dict[int, list] = {}
    for t, parent_xyz, child_xyz in morph.iter_segments():
        seg = [tuple(parent_xyz[list(axes)]), tuple(child_xyz[list(axes)])]
        by_type.setdefault(t, []).append(seg)
    return by_type


def plot_2d(morph: Morphology, projection: str = "xy", ax=None,
            linewidth: float = 0.6, mark_soma: bool = True,
            equal: bool = True, legend: bool = True, title: str | None = None):
    """Plot a neuron as a 2D projection colored by point type.

    Parameters
    ----------
    morph : the morphology to draw.
    projection : one of ``'xy'``, ``'xz'``, ``'yz'``.
    ax : an existing matplotlib axis; a new one is created if omitted.
    linewidth, mark_soma, equal, legend, title : styling options.

    Returns the matplotlib axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if projection not in _AXIS_MAP:
        raise ValueError(f"projection must be one of {list(_AXIS_MAP)}")
    a, b = _AXIS_MAP[projection]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for t, segs in sorted(_segments_by_type(morph, (a, b)).items()):
        ax.add_collection(LineCollection(
            segs, colors=type_color(t), linewidths=linewidth,
            label=f"{type_name(t)} (n={len(segs)})"))

    if mark_soma:
        soma = morph.nodes[morph.nodes["type"] == 1]
        if len(soma):
            coords = soma[["x", "y", "z"]].to_numpy()
            ax.scatter(coords[:, a], coords[:, b], s=40, c="black",
                       marker="o", zorder=5, label="soma")

    ax.autoscale()
    if equal:
        ax.set_aspect("equal")
    labels = ["x", "y", "z"]
    ax.set_xlabel(labels[a])
    ax.set_ylabel(labels[b])
    ax.set_title(title if title is not None else (morph.name or ""))
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.6)
    return ax


def plot_3d(morph: Morphology, ax=None, linewidth: float = 0.6,
            mark_soma: bool = True, legend: bool = True,
            title: str | None = None):
    """Plot a neuron in 3D colored by point type.

    Parameters
    ----------
    morph : the morphology to draw.
    ax : an existing 3D axis (``projection='3d'``); created if omitted.
    linewidth, mark_soma, legend, title : styling options.

    Returns the matplotlib 3D axis.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

    for t, segs in sorted(_segments_by_type(morph, (0, 1, 2)).items()):
        ax.add_collection3d(Line3DCollection(
            segs, colors=type_color(t), linewidths=linewidth,
            label=type_name(t)))

    if mark_soma:
        soma = morph.nodes[morph.nodes["type"] == 1]
        if len(soma):
            s = soma[["x", "y", "z"]].to_numpy()
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], c="black", s=40, label="soma")

    mn, mx = morph.bounding_box()
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    try:  # keep real-world proportions when the data has extent on every axis
        span = mx - mn
        if np.all(span > 0):
            ax.set_box_aspect(span)
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title if title is not None else (morph.name or ""))
    if legend:
        ax.legend(fontsize=7)
    return ax
