# morph_toolbox

A small, well-organized toolbox for working with neuron **morphology** data.
It centers on the standard [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
reconstruction format and gives you three core capabilities:

- **Load** SWC files into a clean, validated data structure (single files or a
  whole dataset via a file index).
- **Convert** JSON neuron descriptions into SWC (tolerant of many JSON shapes).
- **Analyze** per-neuron morphometrics, terminal tips, and branch pruning, plus
  population-wide morphometric tables.
- **Visualize** a neuron in 2D (XY / XZ / YZ projections) or 3D, colored by point type.
- **Annotate** (optional) reconstructions against the Allen CCFv3 to map nodes /
  axon terminals to brain regions.

## Install

```bash
python -m pip install -e .            # runtime install
python -m pip install -e ".[ccf]"     # + Allen CCF annotation (pynrrd)
python -m pip install -e ".[analysis]" # + tqdm progress bars
python -m pip install -e ".[dev]"     # + test/build tooling (and the above)
```

Requires Python ≥ 3.9 with `numpy`, `pandas`, and `matplotlib`. The package
ships type information (`py.typed`), so type checkers see its annotations.
CCF annotation additionally needs `pynrrd` (the `ccf` extra).

To build a distributable wheel and sdist:

```bash
python -m build      # writes dist/morph_toolbox-*.whl and *.tar.gz
```

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

# 4. Analyze
morph.morphometrics()                  # flat dict of per-neuron features
morph.terminal_nodes(types=2)          # axon tip rows
morph.prune_branches(0.5, seed=0)      # drop ~50% of nodes as whole branches

# 5. Work over a whole dataset
idx = mt.build_file_index("/path/to/data")   # one row per .swc file
feats = mt.morphometrics_table(idx)           # one morphometrics row per neuron

# 6. (optional) Allen CCF brain-region annotation  -- needs morph_toolbox[ccf]
from morph_toolbox import ccf
ccf.annotate_morphology(morph, cache_dir="data/ccf_cache")     # per-node regions
ccf.projection_vector(morph, cache_dir="data/ccf_cache")       # axon-target histogram
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
│   ├── constants.py        # SWC type codes, names, plot colors, CCF extent
│   ├── core.py             # the Morphology class (wraps the SWC node table)
│   ├── io.py               # load_swc / save_swc / reindex_nodes / file index
│   ├── convert.py          # json_to_morphology / json_to_swc
│   ├── analysis.py         # morphometrics_table (population batch)
│   ├── ccf.py              # optional Allen CCF brain-region annotation
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
| `morph.summary()` | structured dict of the above |
| `morph.terminal_nodes(types=None)` | terminal-tip rows (optionally by type) |
| `morph.morphometrics()` | flat dict of scalar features (for tabulating) |
| `morph.prune_branches(frac, seed=...)` | drop whole branches → new Morphology |

### Datasets & analysis

For directory layouts of `<root>/<sample>/<neuron>.swc`, `mt.build_file_index(root)`
returns one row per file (`sample_id, neuron_id, filename, path, size_bytes`),
`mt.find_swc_files(root)` lists the paths, and `mt.load_many(paths)` loads them
into a list of `Morphology`. `mt.morphometrics_table(file_index)` computes
`Morphology.morphometrics()` for every file and returns one row per neuron.

### CCF brain-region annotation (optional)

`morph_toolbox.ccf` maps CCFv3 micron coordinates to Allen brain regions. It
needs `pynrrd` (`pip install morph_toolbox[ccf]`) and a `cache_dir` for the
downloaded annotation volume + ontology: `ccf.annotate_points`,
`ccf.annotate_region`, `ccf.annotate_morphology`, and `ccf.projection_vector`
(a normalized axon-terminal region histogram).

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

## License

[MIT](LICENSE). © 2026 Shuonan Chen.
