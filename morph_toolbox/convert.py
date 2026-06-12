"""Convert JSON neuron descriptions into SWC / :class:`Morphology`.

There is no single JSON standard for neuron morphology, so this module is
deliberately tolerant.  It accepts:

* a top-level JSON *list* of node objects, or
* a top-level JSON *object* whose node list lives under a common key
  (``nodes``, ``compartments``, ``compartmentList``, ``data``, ``neuron``, ...).

For each node it resolves the seven SWC fields by trying a list of common
key *aliases* (case-insensitive), e.g. ``parent`` may appear as
``parent``, ``parent_id``, ``parentSampleNumber``, or ``pid``.  Unknown but
fixable issues (missing radius, string numbers) are coerced sensibly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ROOT_PARENT, SWC_COLUMNS
from .core import Morphology
from .io import save_swc

# Candidate keys for each SWC field, tried in order (matched case-insensitively).
_FIELD_ALIASES = {
    "id": ["id", "node_id", "sampleNumber", "sample_number", "n", "PointNo"],
    "type": ["type", "node_type", "structureIdentifier", "structure_id",
             "label", "Label", "t"],
    "x": ["x", "X"],
    "y": ["y", "Y"],
    "z": ["z", "Z"],
    "radius": ["radius", "r", "R", "Radius"],
    "parent": ["parent", "parent_id", "parentSampleNumber",
               "parent_sample_number", "pid", "Parent"],
}

# Keys under which a node list might be nested in a top-level object.
_LIST_KEYS = ["nodes", "compartments", "compartmentList", "data", "neuron",
              "neurons", "points", "swc", "tree"]

# Arbor keys in the MouseLight / Janelia neuron JSON format, and the SWC type
# code to fall back to if a node lacks a structureIdentifier.
_MOUSELIGHT_ARBORS = {"soma": 1, "axon": 2, "dendrite": 3, "apical": 4}


def _find_node_list(obj) -> list:
    """Extract the list of node dicts from a parsed JSON object."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in _LIST_KEYS:
            val = obj.get(key)
            if isinstance(val, list):
                return val
            # Allen format: {"neuron": {"compartmentList": [...]}}
            if isinstance(val, dict):
                for k2 in _LIST_KEYS:
                    if isinstance(val.get(k2), list):
                        return val[k2]
        # Fall back to the first list-of-dicts value we find.
        for val in obj.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
    raise ValueError(
        "could not locate a list of node objects in the JSON; "
        f"top-level type was {type(obj).__name__}")


def _build_resolver(sample: dict) -> dict[str, str]:
    """Map each SWC field to the actual key present in a sample node dict."""
    lower = {k.lower(): k for k in sample}
    resolver: dict[str, str] = {}
    for field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                resolver[field] = lower[alias.lower()]
                break
    return resolver


def _is_mouselight(obj) -> bool:
    """True if ``obj`` looks like a MouseLight / Janelia neuron JSON document."""
    neurons = obj.get("neurons") if isinstance(obj, dict) else None
    if not (isinstance(neurons, list) and neurons and isinstance(neurons[0], dict)):
        return False
    return any(k in neurons[0] for k in ("axon", "dendrite", "soma"))


def _mouselight_to_morphology(obj, name=None, neuron_index=0) -> Morphology:
    """Convert one neuron from a MouseLight JSON document into a Morphology.

    The MouseLight format stores a neuron as separate ``soma``/``axon``/
    ``dendrite`` arbors.  Each arbor is its own node list with ``sampleNumber``
    /``parentNumber`` that restart at 1, and each arbor repeats the soma as its
    root (``structureIdentifier == 1``, ``parentNumber == -1``).  We merge them
    into one SWC tree: a single soma root, with every arbor's nodes renumbered
    to globally-unique ids and their roots re-parented onto that soma.
    """
    neurons = obj["neurons"]
    if not 0 <= neuron_index < len(neurons):
        raise IndexError(
            f"neuron_index {neuron_index} out of range (file has {len(neurons)})")
    neuron = neurons[neuron_index]

    # 1. The single soma root (id 1).  Prefer the explicit "soma" point; else
    #    fall back to any arbor's root node.
    soma = neuron.get("soma") or {}
    if not soma:
        for key in _MOUSELIGHT_ARBORS:
            arbor = neuron.get(key)
            if isinstance(arbor, list) and arbor:
                soma = arbor[0]
                break
    rows = [{
        "id": 1, "type": 1,
        "x": float(soma.get("x", 0.0)), "y": float(soma.get("y", 0.0)),
        "z": float(soma.get("z", 0.0)),
        "radius": float(soma.get("radius", 1.0) or 1.0), "parent": ROOT_PARENT,
    }]

    next_id = 2
    for key, fallback_type in _MOUSELIGHT_ARBORS.items():
        if key == "soma":
            continue
        arbor = neuron.get(key)
        if not isinstance(arbor, list) or not arbor:
            continue

        # Map this arbor's local sampleNumber -> global id.  Arbor roots
        # (parentNumber == -1) are the duplicated soma; map them to id 1.
        local2global = {}
        for n in arbor:
            if int(n.get("parentNumber", -1)) == ROOT_PARENT:
                local2global[int(n["sampleNumber"])] = 1
        for n in arbor:
            s = int(n["sampleNumber"])
            if s in local2global:
                continue
            local2global[s] = next_id
            next_id += 1

        for n in arbor:
            if int(n.get("parentNumber", -1)) == ROOT_PARENT:
                continue  # the soma alias, already represented by id 1
            pnum = int(n["parentNumber"])
            rows.append({
                "id": local2global[int(n["sampleNumber"])],
                "type": int(n.get("structureIdentifier", fallback_type)),
                "x": float(n["x"]), "y": float(n["y"]), "z": float(n["z"]),
                "radius": float(n.get("radius", 0.0) or 0.0),
                "parent": local2global.get(pnum, 1),
            })

    df = pd.DataFrame(rows)[SWC_COLUMNS]
    df = df.astype({"id": np.int64, "type": np.int64, "parent": np.int64})
    morph = Morphology(df, name=name or neuron.get("idString"))
    morph.metadata.update({"source_format": "mouselight",
                          "n_neurons_in_file": len(neurons)})
    return morph


def json_to_morphology(source, name: str | None = None,
                       type_map: dict | None = None,
                       neuron_index: int = 0) -> Morphology:
    """Parse JSON (a path, string, dict, or list) into a :class:`Morphology`.

    Parameters
    ----------
    source :
        A path to a ``.json`` file, a JSON string, or an already-parsed
        ``dict``/``list``.
    name :
        Label for the morphology.
    type_map :
        Optional remapping applied to the ``type`` column after parsing
        (e.g. ``{0: 1}`` to relabel undefined nodes as soma).

    Notes
    -----
    Missing ``radius`` defaults to ``0.0``; a missing ``parent`` (or any
    falsy/negative value) becomes the root sentinel ``-1``.  Nodes are
    returned in input order.
    """
    # 1. Load raw JSON.
    if isinstance(source, (dict, list)):
        obj = source
        src_name = None
    else:
        text = str(source)
        looks_like_path = "\n" not in text and (
            text.strip().startswith("{") is False
            and text.strip().startswith("[") is False)
        if looks_like_path and Path(text).exists():
            obj = json.loads(Path(text).read_text())
            src_name = Path(text).stem
        else:
            obj = json.loads(text)
            src_name = None

    # MouseLight / Janelia documents store arbors separately; handle them with
    # a dedicated merger rather than the flat-node-list path below.
    if _is_mouselight(obj):
        morph = _mouselight_to_morphology(obj, name=name or src_name,
                                          neuron_index=neuron_index)
        if type_map:
            morph.nodes["type"] = morph.nodes["type"].map(
                lambda t: type_map.get(t, t)).astype(np.int64)
        return morph

    nodes = _find_node_list(obj)
    if not nodes:
        raise ValueError("JSON contained an empty node list")
    if not isinstance(nodes[0], dict):
        raise ValueError("node list entries are not objects/dicts")

    # 2. Figure out which keys hold which SWC fields.
    resolver = _build_resolver(nodes[0])
    for required in ("id", "x", "y", "z", "parent"):
        if required not in resolver:
            raise ValueError(
                f"could not find a '{required}' field among node keys "
                f"{list(nodes[0].keys())}")

    # 3. Pull each column out.
    def col(field, default=None):
        key = resolver.get(field)
        if key is None:
            return [default] * len(nodes)
        return [n.get(key, default) for n in nodes]

    def as_int(values, default):
        out = []
        for v in values:
            if v is None or v == "":
                out.append(default)
            else:
                out.append(int(float(v)))
        return out

    parents_raw = as_int(col("parent"), ROOT_PARENT)
    # Normalize any non-positive / null parent marker to the SWC root sentinel.
    parents = [p if p is not None and p > 0 else ROOT_PARENT for p in parents_raw]

    df = pd.DataFrame({
        "id": as_int(col("id"), 0),
        "type": as_int(col("type", 0), 0),
        "x": np.asarray(col("x", 0.0), dtype=float),
        "y": np.asarray(col("y", 0.0), dtype=float),
        "z": np.asarray(col("z", 0.0), dtype=float),
        "radius": np.asarray(col("radius", 0.0), dtype=float),
        "parent": parents,
    })

    if type_map:
        df["type"] = df["type"].map(lambda t: type_map.get(t, t)).astype(np.int64)

    morph = Morphology(df[SWC_COLUMNS], name=name or src_name)
    morph.metadata["source_format"] = "json"
    return morph


def json_to_swc(source, out_path, name: str | None = None,
                type_map: dict | None = None, reindex: bool = False,
                neuron_index: int = 0):
    """Convert a JSON neuron description to an SWC file on disk.

    Parameters mirror :func:`json_to_morphology`, plus ``out_path`` (the
    destination ``.swc``) and ``reindex`` (renumber ids to 1..N before
    writing).  Returns the destination :class:`pathlib.Path`.
    """
    from .io import reindex_nodes

    morph = json_to_morphology(source, name=name, type_map=type_map,
                               neuron_index=neuron_index)
    if reindex:
        morph = Morphology(reindex_nodes(morph.nodes), name=morph.name,
                           metadata=morph.metadata)
    return save_swc(morph, out_path)
