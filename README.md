# morph_toolbox

A small, well-organized toolbox for working with neuron **morphology** data.
It centers on the standard [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
reconstruction format and gives you three core capabilities:

- **Load** SWC files into a clean, validated data structure.
- **Convert** JSON neuron descriptions into SWC (tolerant of many JSON shapes).
- **Visualize** a neuron in 2D (XY / XZ / YZ projections) or 3D, colored by point type.

It's intentionally simple and modular so it can grow (morphometrics, batch
processing, downsampling, etc.) without rework.

## Install

```bash
python -m pip install -e .
```

Requires Python ≥ 3.9 with `numpy`, `pandas`, and `matplotlib`.

## Quick start

```python
import morph_toolbox as mt

# 1. Load an SWC reconstruction
morph = mt.load_swc("examples/sample.swc")
print(morph)                    # <Morphology 'sample': 13 nodes, 1 root(s), ...>
print(morph.summary())          # headline morphometrics

# 2. Convert a JSON neuron description to SWC
mt.json_to_swc("examples/sample.json", "out.swc")

# 3. Visualize
mt.plot_2d(morph, projection="xy")
mt.plot_3d(morph)
```

Run the full demo (writes `examples/quickstart_output.png`):

```bash
python examples/quickstart.py
```

## Layout

```
morph_toolbox/
├── morph_toolbox/          # the package
│   ├── __init__.py         # public API (load_swc, json_to_swc, plot_2d, ...)
│   ├── constants.py        # SWC type codes, names, and plot colors
│   ├── core.py             # the Morphology class (wraps the SWC node table)
│   ├── io.py               # load_swc / save_swc / reindex_nodes
│   ├── convert.py          # json_to_morphology / json_to_swc
│   └── viz.py              # plot_2d / plot_3d
├── examples/               # runnable demo + sample SWC and JSON
└── tests/                  # pytest suite
```

## Concepts

### The `Morphology` object

A `Morphology` wraps the SWC node table — a `pandas.DataFrame` with columns
`id, type, x, y, z, radius, parent`, accessible via `morph.nodes`. It validates
SWC invariants on construction (unique ids, valid parent references, ≥1 root)
and exposes topology and geometry helpers:

| Member | Meaning |
| --- | --- |
| `len(morph)` | number of nodes |
| `morph.roots`, `morph.num_roots` | root node ids / count |
| `morph.num_branch_points`, `morph.num_tips` | branch / terminal counts |
| `morph.coords` | `(N, 3)` XYZ array |
| `morph.soma_coord` | mean soma coordinate |
| `morph.total_length()` | total cable length |
| `morph.bounding_box()` | `(min_xyz, max_xyz)` |
| `morph.type_counts()` | nodes per named type |
| `morph.summary()` | dict of the above |

### SWC point types

Standard Neuronland / Allen convention, each with a consistent plot color
(`morph_toolbox.constants.SWC_TYPE_COLORS`):
`0` undefined · `1` soma · `2` axon · `3` basal dendrite · `4` apical dendrite ·
`5` custom · `6` neurite · `7` glia.

### Gapped ids

Some SWC files keep original (non-consecutive) ids after pruning. Pass
`load_swc(path, reindex=True)` (or call `mt.reindex_nodes`) to renumber to a
contiguous `1..N` while preserving topology.

### JSON → SWC

`json_to_swc` / `json_to_morphology` accept a file path, a JSON string, or an
already-parsed `dict`/`list`. The node list may be top-level or nested under a
common key (`nodes`, `compartments`, `compartmentList`, `neuron`, ...), and each
node's fields are matched case-insensitively against common aliases (e.g.
`parent`, `parentSampleNumber`, `pid`). Missing radius defaults to `0`; a
non-positive/missing parent becomes the root sentinel `-1`.

## Testing

```bash
python -m pytest
```

> `example_tools.py` at the repo root is a reference scratchpad of prior
> functions and is **not** part of the package.
