"""morph_toolbox -- a small toolbox for working with neuron morphology data.

Quick start
-----------
>>> import morph_toolbox as mt
>>> morph = mt.load_swc("neuron.swc")        # load an SWC reconstruction
>>> morph.summary()                           # headline morphometrics
>>> mt.plot_2d(morph, projection="xy")        # 2D projection
>>> mt.plot_3d(morph)                          # 3D view
>>> mt.json_to_swc("neuron.json", "neuron.swc")  # convert JSON -> SWC

The central data structure is :class:`Morphology`, a thin wrapper around the
SWC node table (a pandas DataFrame).  I/O lives in :mod:`morph_toolbox.io`,
JSON conversion in :mod:`morph_toolbox.convert`, and plotting in
:mod:`morph_toolbox.viz`.
"""

from __future__ import annotations

from . import constants
from .constants import SWC_TYPE_COLORS, SWC_TYPE_NAMES, type_color, type_name
from .convert import json_to_morphology, json_to_swc
from .core import Morphology
from .io import load_swc, reindex_nodes, save_swc
from .viz import plot_2d, plot_3d

__version__ = "0.1.0"

__all__ = [
    "Morphology",
    "load_swc",
    "save_swc",
    "reindex_nodes",
    "json_to_morphology",
    "json_to_swc",
    "plot_2d",
    "plot_3d",
    "type_name",
    "type_color",
    "SWC_TYPE_NAMES",
    "SWC_TYPE_COLORS",
    "constants",
]
