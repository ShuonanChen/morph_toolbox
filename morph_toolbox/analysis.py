"""Population-level analysis helpers.

These build on :class:`~morph_toolbox.core.Morphology` to summarize many
neurons at once -- e.g. tabulating per-neuron morphometrics across a whole
dataset for downstream clustering or matching.
"""

from __future__ import annotations

import pandas as pd

from .io import load_swc


def morphometrics_table(file_index: pd.DataFrame, reindex: bool = True,
                        progress: bool = True) -> pd.DataFrame:
    """Compute morphometrics for every file in a file index DataFrame.

    Parameters
    ----------
    file_index :
        A DataFrame with (at least) columns ``sample_id``, ``neuron_id`` and
        ``path`` -- e.g. the output of :func:`morph_toolbox.build_file_index`.
    reindex :
        Passed through to :func:`morph_toolbox.load_swc` for each file.
    progress :
        Show a ``tqdm`` progress bar if available.

    Returns
    -------
    One row per neuron: the ``sample_id``/``neuron_id`` columns followed by all
    keys from :meth:`Morphology.morphometrics`.
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is optional
        def tqdm(x, **k):
            return x

    rows = []
    it = file_index.itertuples(index=False)
    if progress:
        it = tqdm(it, total=len(file_index), desc="morphometrics")
    for rec in it:
        feats = load_swc(rec.path, reindex=reindex).morphometrics()
        feats["sample_id"] = rec.sample_id
        feats["neuron_id"] = rec.neuron_id
        rows.append(feats)

    out = pd.DataFrame(rows)
    front = ["sample_id", "neuron_id"]
    return out[front + [c for c in out.columns if c not in front]]
