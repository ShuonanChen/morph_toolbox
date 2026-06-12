"""Tests for morph_toolbox: loading, conversion, round-trip, and plotting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import morph_toolbox as mt

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
