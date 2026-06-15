"""Tests for morph_toolbox: loading, conversion, round-trip, and plotting."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import morph_toolbox as mt
from morph_toolbox import ccf

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAMPLE_SWC = EXAMPLES / "sample.swc"
SAMPLE_JSON = EXAMPLES / "sample.json"


# -- loading ---------------------------------------------------------------

def test_load_swc_basic():
    m = mt.load_swc(SAMPLE_SWC)
    assert len(m) == 13
    assert m.num_roots == 1            # node 1 is the only root
    assert m.name == "sample"
    assert set(["soma", "axon"]).issubset(m.type_counts())


def test_summary_fields():
    m = mt.load_swc(SAMPLE_SWC)
    s = m.summary()
    for key in ("n_nodes", "n_branch_points", "n_tips", "total_length",
                "bbox_size", "soma_coord", "type_counts"):
        assert key in s
    assert s["total_length"] > 0
    assert len(s["bbox_size"]) == 3


def test_topology_counts():
    m = mt.load_swc(SAMPLE_SWC)
    # Node 1 has children {2,3,6,8,11} -> a branch point; tips are leaves.
    assert m.num_branch_points >= 1
    assert m.num_tips == 5             # nodes 2,5,7,10,13 are leaves


# -- validation ------------------------------------------------------------

def test_validate_rejects_bad_parent():
    m = mt.load_swc(SAMPLE_SWC)
    bad = m.nodes.copy()
    bad.loc[2, "parent"] = 9999        # dangling parent reference
    with pytest.raises(ValueError):
        mt.Morphology(bad)


def test_validate_rejects_duplicate_ids():
    m = mt.load_swc(SAMPLE_SWC)
    bad = m.nodes.copy()
    bad.loc[1, "id"] = bad.loc[0, "id"]
    with pytest.raises(ValueError):
        mt.Morphology(bad)


# -- reindex ---------------------------------------------------------------

def test_reindex_preserves_topology():
    m = mt.load_swc(SAMPLE_SWC)
    gapped = m.nodes.copy()
    # Renumber ids to a gapped scheme and remap parents to match.
    remap = {old: old * 10 for old in gapped["id"]}
    remap[-1] = -1
    gapped["id"] = gapped["id"].map(remap)
    gapped["parent"] = gapped["parent"].map(remap)

    fixed = mt.reindex_nodes(gapped)
    assert list(fixed["id"]) == list(range(1, len(fixed) + 1))
    # Same number of edges / roots as the original.
    assert (fixed["parent"] == -1).sum() == (m.nodes["parent"] == -1).sum()


# -- JSON conversion -------------------------------------------------------

def test_json_to_morphology_matches_swc():
    swc = mt.load_swc(SAMPLE_SWC)
    js = mt.json_to_morphology(SAMPLE_JSON)
    assert len(js) == len(swc)
    assert js.num_roots == swc.num_roots
    np.testing.assert_allclose(
        np.sort(js.coords, axis=0), np.sort(swc.coords, axis=0))


def test_json_to_swc_roundtrip(tmp_path):
    out = tmp_path / "out.swc"
    mt.json_to_swc(SAMPLE_JSON, out)
    assert out.exists()
    reloaded = mt.load_swc(out)
    assert len(reloaded) == 13


def test_json_from_plain_list():
    nodes = [
        {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0, "radius": 1, "parent": -1},
        {"id": 2, "type": 3, "x": 1, "y": 0, "z": 0, "radius": 1, "parent": 1},
    ]
    m = mt.json_to_morphology(nodes)
    assert len(m) == 2 and m.num_roots == 1


def test_mouselight_format_merges_arbors():
    # Minimal MouseLight-shaped doc: separate soma/axon/dendrite arbors, each
    # restarting sampleNumber at 1 and repeating the soma as its root.
    doc = {
        "neurons": [{
            "idString": "TEST-001",
            "soma": {"x": 0, "y": 0, "z": 0, "radius": 5.0},
            "axon": [
                {"sampleNumber": 1, "structureIdentifier": 1, "x": 0, "y": 0,
                 "z": 0, "radius": 5.0, "parentNumber": -1},
                {"sampleNumber": 2, "structureIdentifier": 2, "x": 1, "y": 0,
                 "z": 0, "radius": 1.0, "parentNumber": 1},
                {"sampleNumber": 3, "structureIdentifier": 2, "x": 2, "y": 0,
                 "z": 0, "radius": 1.0, "parentNumber": 2},
            ],
            "dendrite": [
                {"sampleNumber": 1, "structureIdentifier": 1, "x": 0, "y": 0,
                 "z": 0, "radius": 5.0, "parentNumber": -1},
                {"sampleNumber": 2, "structureIdentifier": 3, "x": 0, "y": 1,
                 "z": 0, "radius": 1.0, "parentNumber": 1},
            ],
        }]
    }
    m = mt.json_to_morphology(doc)
    # 1 soma + 2 axon + 1 dendrite (each arbor's duplicate soma is collapsed).
    assert len(m) == 4
    assert m.num_roots == 1            # single merged soma root
    tc = m.type_counts()
    assert tc.get("soma") == 1 and tc.get("axon") == 2
    assert tc.get("basal dendrite") == 1
    assert m.metadata["source_format"] == "mouselight"


def test_json_normalizes_null_parent():
    nodes = [
        {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0, "parent": 0},   # 0 -> root
        {"id": 2, "type": 3, "x": 1, "y": 0, "z": 0, "parent": 1},
    ]
    m = mt.json_to_morphology(nodes)
    assert m.num_roots == 1            # parent 0 normalized to -1
    assert (m.nodes["radius"] == 0).all()  # missing radius defaults to 0


# -- save ------------------------------------------------------------------

def test_save_swc_roundtrip(tmp_path):
    m = mt.load_swc(SAMPLE_SWC)
    out = tmp_path / "rt.swc"
    mt.save_swc(m, out)
    again = mt.load_swc(out)
    assert len(again) == len(m)
    np.testing.assert_allclose(again.coords, m.coords)


# -- plotting (smoke tests) ------------------------------------------------

def test_plot_2d_runs():
    import matplotlib.pyplot as plt
    m = mt.load_swc(SAMPLE_SWC)
    ax = mt.plot_2d(m, projection="xz")
    assert ax is not None
    plt.close("all")


def test_plot_3d_runs():
    import matplotlib.pyplot as plt
    m = mt.load_swc(SAMPLE_SWC)
    ax = mt.plot_3d(m)
    assert ax is not None
    plt.close("all")


def test_plot_2d_bad_projection():
    m = mt.load_swc(SAMPLE_SWC)
    with pytest.raises(ValueError):
        mt.plot_2d(m, projection="ab")


# -- terminal nodes --------------------------------------------------------

def test_terminal_nodes():
    m = mt.load_swc(SAMPLE_SWC)
    tips = m.terminal_nodes()
    assert len(tips) == m.num_tips == 5
    # Only node 10 is an axon (type 2) terminal in the sample.
    axon_tips = m.terminal_nodes(types=2)
    assert len(axon_tips) == 1
    assert set(axon_tips["id"]) == {10}
    # List of types also works: dendritic tips are nodes 5, 7 (type 3) and 13 (type 4).
    assert set(m.terminal_nodes(types=[3, 4])["id"]) == {5, 7, 13}


# -- morphometrics ---------------------------------------------------------

def test_morphometrics_fields():
    m = mt.load_swc(SAMPLE_SWC)
    feats = m.morphometrics()
    expected = {
        "n_nodes", "n_branch_points", "n_tips", "n_soma_nodes", "n_axon_nodes",
        "n_dend_nodes", "n_undef_nodes", "total_length_um", "max_path_radius_um",
        "mean_radius_um", "bbox_x_um", "bbox_y_um", "bbox_z_um",
        "soma_x", "soma_y", "soma_z",
    }
    assert expected.issubset(feats)
    assert feats["n_nodes"] == 13
    assert feats["n_tips"] == 5
    assert feats["n_soma_nodes"] == 2     # nodes 1 and 2 are type 1
    assert feats["n_axon_nodes"] == 3     # nodes 8, 9, 10
    assert feats["total_length_um"] > 0
    # Matches the cable length exposed by the Morphology directly.
    assert feats["total_length_um"] == pytest.approx(m.total_length())


# -- pruning ---------------------------------------------------------------

def test_prune_branches_reduces_and_stays_valid():
    m = mt.load_swc(SAMPLE_SWC)
    pruned = m.prune_branches(0.3, seed=0)
    assert isinstance(pruned, mt.Morphology)
    assert len(pruned) < len(m)
    # Reindexed to a contiguous 1..N and still a valid tree.
    assert list(pruned.nodes["id"]) == list(range(1, len(pruned) + 1))
    pruned.validate()
    assert pruned.metadata["n_removed"] == len(m) - len(pruned)
    assert 0 < pruned.metadata["frac_removed"] <= 1.0
    # Soma is never removed.
    assert (pruned.nodes["type"] == 1).any()


def test_prune_branches_frac_zero_is_noop():
    m = mt.load_swc(SAMPLE_SWC)
    pruned = m.prune_branches(0.0)
    assert len(pruned) == len(m)
    assert pruned.metadata["n_removed"] == 0
    assert pruned.metadata["frac_removed"] == 0.0


def test_prune_branches_reproducible():
    m = mt.load_swc(SAMPLE_SWC)
    a = m.prune_branches(0.5, seed=7)
    b = m.prune_branches(0.5, seed=7)
    assert a.metadata["n_removed"] == b.metadata["n_removed"]
    assert list(a.nodes["id"]) == list(b.nodes["id"])


# -- file index / batch ----------------------------------------------------

@pytest.fixture
def mini_dataset(tmp_path):
    """A <root>/<sample>/<neuron>.swc layout with two copies of the sample."""
    for sample, neuron in [("s1", "001"), ("s1", "002"), ("s2", "001")]:
        dest = tmp_path / sample / f"{neuron}.swc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SAMPLE_SWC, dest)
    return tmp_path


def test_find_swc_files(mini_dataset):
    files = mt.find_swc_files(mini_dataset)
    assert len(files) == 3
    assert files == sorted(files)


def test_build_file_index(mini_dataset):
    idx = mt.build_file_index(mini_dataset)
    assert list(idx.columns) == [
        "sample_id", "neuron_id", "filename", "path", "size_bytes"]
    assert len(idx) == 3
    assert set(idx["sample_id"]) == {"s1", "s2"}
    assert (idx["size_bytes"] > 0).all()


def test_load_many(mini_dataset):
    idx = mt.build_file_index(mini_dataset)
    morphs = mt.load_many(idx["path"])
    assert len(morphs) == 3
    assert all(isinstance(m, mt.Morphology) for m in morphs)
    assert all("sample_id" in m.metadata and "neuron_id" in m.metadata
               for m in morphs)


def test_morphometrics_table(mini_dataset):
    idx = mt.build_file_index(mini_dataset)
    table = mt.morphometrics_table(idx, progress=False)
    assert len(table) == 3
    assert list(table.columns[:2]) == ["sample_id", "neuron_id"]
    assert {"n_nodes", "total_length_um", "soma_x"}.issubset(table.columns)
    assert (table["n_nodes"] == 13).all()


# -- CCF annotation (optional dependency) ----------------------------------

def test_ccf_extent_exposed():
    assert ccf.CCF_EXTENT_UM == mt.CCF_EXTENT_UM
    assert ccf.CCF_EXTENT_UM["x"] == (0, 13200)


def test_ccf_requires_cache_dir():
    with pytest.raises(ValueError):
        ccf.load_ccf_ontology(None)


def test_ccf_annotation_without_pynrrd(tmp_path):
    """If pynrrd is unavailable, the volume loader raises a clear ImportError."""
    pytest.importorskip  # keep import side-effect-free
    try:
        import nrrd  # noqa: F401
        has_nrrd = True
    except ImportError:
        has_nrrd = False

    if has_nrrd:
        pytest.skip("pynrrd installed; download-dependent path not exercised")
    with pytest.raises(ImportError, match="pynrrd"):
        ccf.load_ccf_annotation(tmp_path, resolution=25)
