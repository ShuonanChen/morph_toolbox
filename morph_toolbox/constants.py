"""Shared constants for the morphology toolbox.

SWC point-type codes follow the standard Neuronland / Allen convention.
Coordinates in SWC files produced for the Allen Mouse Common Coordinate
Framework (CCFv3) are in microns.
"""

from __future__ import annotations

# The seven canonical columns of an SWC node table, in file order.
SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent"]

# Sentinel parent id used by SWC to mark a root (e.g. soma) node.
ROOT_PARENT = -1

# SWC point-type codes -> human-readable names.
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

# A consistent color per point type, used across all plots.
SWC_TYPE_COLORS = {
    0: "#7f7f7f",  # undefined  - grey
    1: "#000000",  # soma       - black
    2: "#1f77b4",  # axon       - blue
    3: "#d62728",  # basal dend - red
    4: "#ff7f0e",  # apical     - orange
    5: "#9467bd",  # custom     - purple
    6: "#2ca02c",  # neurite    - green
    7: "#8c564b",  # glia       - brown
}

# Fallback color for any type code not in SWC_TYPE_COLORS.
DEFAULT_COLOR = "#333333"


def type_name(code: int) -> str:
    """Return the human-readable name for an SWC type code."""
    return SWC_TYPE_NAMES.get(int(code), f"type {int(code)}")


def type_color(code: int) -> str:
    """Return the plotting color for an SWC type code."""
    return SWC_TYPE_COLORS.get(int(code), DEFAULT_COLOR)
