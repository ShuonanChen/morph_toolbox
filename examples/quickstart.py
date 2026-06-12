"""End-to-end demo of morph_toolbox.

Run from the repo root:  python examples/quickstart.py

Loads the sample SWC, prints summary morphometrics, converts the sample JSON
to SWC, and saves a 2D + 3D figure to ``examples/quickstart_output.png``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove for an interactive window
import matplotlib.pyplot as plt

import morph_toolbox as mt

HERE = Path(__file__).resolve().parent


def main() -> None:
    # 1. Load an SWC reconstruction.
    morph = mt.load_swc(HERE / "sample.swc")
    print("Loaded:", morph)
    for k, v in morph.summary().items():
        print(f"  {k}: {v}")

    # 2. Convert the JSON version to SWC and confirm it round-trips.
    out_swc = HERE / "sample_from_json.swc"
    mt.json_to_swc(HERE / "sample.json", out_swc)
    from_json = mt.load_swc(out_swc)
    print(f"\nConverted JSON -> {out_swc.name}: {from_json}")

    # 3. Visualize: 2D projection + 3D view side by side.
    fig = plt.figure(figsize=(12, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    mt.plot_2d(morph, projection="xy", ax=ax2d, title="XY projection")
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    mt.plot_3d(morph, ax=ax3d, title="3D view")

    out_png = HERE / "quickstart_output.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved figure -> {out_png}")


if __name__ == "__main__":
    main()
