# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "tiled[server]",
#     "pandas",
#     "h5py",
#     "numpy",
#     "matplotlib",
#     "ruamel.yaml",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Modal Data Broker: Ingestion Showcase

    This notebook demonstrates that **6 heterogeneous datasets** have been
    successfully ingested into a single Tiled catalog and are accessible
    for interactive exploration.

    | Dataset | Domain | Entities | Artifacts | Source |
    |---------|--------|----------|-----------|--------|
    | **VDP** | Spin Hamiltonians | 10,000 | 110,000 | Sunny.jl simulations |
    | **EDRIXS** | RIXS spectra | 10,000 | 10,000 | EDRIXS calculations |
    | **NiPS3 Multimodal** | Magnetism + INS | 7,616 | 45,696 | Synthetic |
    | **RIXS** | Exp. RIXS | 7 | 42 | Converted HDF5 |
    | **Challenge** | Benchmark | 1 | 9 | Converted HDF5 |
    | **SEQUOIA** | Neutron TOF | 3 | 76 | SEQUOIA instrument |
    | **Total** | | **27,627** | **165,823** | |

    **Prerequisites:** Start the Tiled server from `cwang31-data-broker/`:
    ```bash
    export BROKER=/sdf/data/lcls/ds/prj/prjmaiqmag01/results/cwang31/proj-vdp-generic-broker/tiled_poc
    uv run --with $BROKER tiled serve config config.yml --api-key secret
    ```
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import json
    import time
    import os
    from pathlib import Path

    return Path, json, mo, mticker, np, os, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Registration Progression

    The catalog grew across three milestones.  The chart below is rendered
    live from `counts.json` produced during the ingestion pipeline.
    """)
    return


@app.cell
def _(Path, json, mo, mticker, np, plt):
    # Load counts.json (CWD-relative)
    _counts_path = Path("counts.json")
    with open(_counts_path) as _f:
        _counts = json.load(_f)

    _labels = [c["label"] for c in _counts]
    _cum_ent = [c["cumulative_entities"] for c in _counts]
    _cum_art = [c["cumulative_artifacts"] for c in _counts]
    _descriptions = [c["description"] for c in _counts]

    _x = np.arange(len(_labels))
    _width = 0.35

    fig_prog, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(_x - _width / 2, _cum_ent, _width, label="Entities", color="#2563eb")
    _ax.bar(_x + _width / 2, _cum_art, _width, label="Artifacts", color="#f59e0b")

    _ax.set_xlabel("Milestone")
    _ax.set_ylabel("Cumulative Count")
    _ax.set_title("Catalog Growth: Entities & Artifacts Over Time")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels)
    _ax.legend()
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v >= 1000 else f"{v:.0f}"))
    _ax.grid(axis="y", alpha=0.3)

    # Annotate descriptions
    for _i, _desc in enumerate(_descriptions):
        _ax.annotate(
            _desc,
            xy=(_x[_i], max(_cum_ent[_i], _cum_art[_i])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="gray",
        )

    # Annotate artifact bars with cumulative data size (GB)
    _cum_gb = [c.get("cumulative_data_gb", 0) for c in _counts]
    for _i, _gb in enumerate(_cum_gb):
        if _gb > 0:
            _ax.annotate(
                f"{_gb} GB",
                xy=(_x[_i] + _width / 2, _cum_art[_i] / 2),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()
    fig_prog.savefig("catalog_growth.png", dpi=300, bbox_inches="tight")

    mo.md(f"**Current totals:** {_cum_ent[-1]:,} entities, {_cum_art[-1]:,} artifacts, {_cum_gb[-1]} GB")
    return (fig_prog,)


@app.cell
def _(fig_prog):
    fig_prog
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connect to Catalog

    All 6 datasets live in a single Tiled catalog (`catalog.db`) served
    on port 8006.
    """)
    return


@app.cell
def _(mo, os):
    from tiled.client import from_uri

    TILED_URL = os.environ.get("TILED_URL", "http://localhost:8006")
    API_KEY = os.environ.get("TILED_API_KEY", "secret")

    client = from_uri(TILED_URL, api_key=API_KEY)

    mo.md(f"**Connected to `{TILED_URL}`** — catalog contains **{len(client):,}** datasets: `{list(client.keys())}`")
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-Dataset Retrieval (Mode B)

    For each dataset we fetch **one representative entity** via Tiled
    adapters (Mode B), list its children, and read the first array to
    confirm end-to-end retrieval.
    """)
    return


@app.cell
def _(client, mo, time):
    # Pre-selected sample keys (one per dataset, verified during registration)
    SAMPLES = {
        "VDP":       ("VDP", "H_636ce3e4"),
        "EDRIXS":    ("EDRIXS", "H_edx00000"),
        "NiPS3":     ("NiPS3_Multimodal", "H_mm_1"),
        "RIXS":      ("RIXS", "H_rixs_052"),
        "Challenge": ("Challenge", "H_challeng"),
        "SEQUOIA":   ("SEQUOIA", "H_seq_Ei28"),
    }

    _rows = []
    for _label, (_dataset_key, _entity_key) in SAMPLES.items():
        _t0 = time.perf_counter()
        _h = client[_dataset_key][_entity_key]
        _children = list(_h.keys())
        _arr = _h[_children[0]].read()
        _elapsed_ms = (time.perf_counter() - _t0) * 1000

        # Pick a few representative metadata keys (skip path_/dataset_/index_)
        _meta_keys = [
            k for k in _h.metadata.keys()
            if not k.startswith(("path_", "dataset_", "index_", "uid", "key"))
        ][:4]

        _rows.append(
            f"| {_label} | `{_entity_key}` | {len(_children)} | "
            f"`{_children[0]}` | `{_arr.shape}` | "
            f"{', '.join(_meta_keys)} | {_elapsed_ms:.0f} ms |"
        )

    _table = "\n".join(_rows)
    mo.md(f"""
    | Dataset | Sample Key | Children | First Child | Shape | Metadata (sample) | Read Time |
    |---------|-----------|----------|-------------|-------|-------------------|-----------|
    {_table}

    All 6 datasets retrieved successfully.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mode A: Expert Path-Based Access

    The expert workflow uses **physics-based queries** to find matching
    entities, then extracts the **locator tuple** `(file, dataset, index)`
    from metadata and reads directly with h5py — no Tiled adapter overhead.

    The database stores relative paths; `base_dir` comes from the dataset
    config YAML (a deployment concern, not stored in the catalog).

    ```python
    from tiled.queries import Key

    # Scope to a dataset, then query by physics parameters
    vdp = client["VDP"]
    subset = vdp.search(Key("Ja_meV") > 0.5).search(Key("Dc_meV") < -0.5)
    h       = list(subset.values())[0]
    file    = h.metadata["path_mh_powder_30T"]      # relative HDF5 path
    dataset = h.metadata["dataset_mh_powder_30T"]   # HDF5 dataset path
    index   = h.metadata.get("index_mh_powder_30T") # batch index (if any)
    # full_path = base_dir / file
    ```
    """)
    return


@app.cell
def _(Path, client, mo, time):
    import h5py
    from ruamel.yaml import YAML
    from tiled.queries import Key

    # Load base_dir from each dataset config YAML (keyed by config "key")
    _yaml = YAML()
    _base_dirs = {}
    for _cfg_path in sorted(Path("datasets").glob("*.yaml")):
        with open(_cfg_path) as _f:
            _cfg = _yaml.load(_f)
        _base_dirs[_cfg["key"]] = _cfg["base_dir"]

    # Per-dataset: key, query filters, artifact suffix, query description
    # Each filter is (metadata_key, operator, value)
    _SAMPLES = [
        {
            "label": "VDP",
            "key": "VDP",
            "query_desc": "Ja > 0.5, Dc < -0.5",
            "filters": [("Ja_meV", ">", 0.5), ("Dc_meV", "<", -0.5)],
            "suffix": "mh_powder_30T",
        },
        {
            "label": "EDRIXS",
            "key": "EDRIXS",
            "query_desc": "10Dq > 3.0, Gam_c < 0.2",
            "filters": [("tenDq", ">", 3.0), ("Gam_c", "<", 0.2)],
            "suffix": "rixs",
        },
        {
            "label": "NiPS3 Multimodal",
            "key": "NiPS3_Multimodal",
            "query_desc": "J1a < -4.0, Az > 0.3",
            "filters": [("J1a", "<", -4.0), ("Az", ">", 0.3)],
            "suffix": "ins_powder",
        },
        {
            "label": "RIXS",
            "key": "RIXS",
            "query_desc": "experiment_type == energy_scan",
            "filters": [("experiment_type", "==", "energy_scan")],
            "suffix": "S",
        },
        {
            "label": "Challenge",
            "key": "Challenge",
            "query_desc": "material_formula == YbBi2IO4",
            "filters": [("material_formula", "==", "YbBi2IO4")],
            "suffix": "cef_spectrum",
        },
        {
            "label": "SEQUOIA",
            "key": "SEQUOIA",
            "query_desc": "incident_energy_meV == 28",
            "filters": [("incident_energy_meV", "==", 28)],
            "suffix": "path_0001_intensity",
        },
    ]

    def _apply_filters(_client, _filters):
        _sub = _client
        for _key, _op, _val in _filters:
            if _op == ">":
                _sub = _sub.search(Key(_key) > _val)
            elif _op == "<":
                _sub = _sub.search(Key(_key) < _val)
            elif _op == "==":
                _sub = _sub.search(Key(_key) == _val)
            elif _op == ">=":
                _sub = _sub.search(Key(_key) >= _val)
        return _sub

    _rows = []
    for _s in _SAMPLES:
        _t0 = time.perf_counter()

        # Step 1: Query by physics parameters (scoped to dataset)
        _dataset_client = client[_s["key"]]
        _subset = _apply_filters(_dataset_client, _s["filters"])
        _n_hits = len(_subset)

        # Step 2: First matching entity
        _ent_key = list(_subset.keys())[0]
        _h = _subset[_ent_key]
        _meta = _h.metadata

        # Step 3: Extract locator tuple
        _suffix = _s["suffix"]
        _file    = _meta[f"path_{_suffix}"]
        _dataset = _meta[f"dataset_{_suffix}"]
        _index   = _meta.get(f"index_{_suffix}")

        # Step 4: Read via h5py
        _base = _base_dirs[_s["key"]]
        _full_path = f"{_base}/{_file}"

        with h5py.File(_full_path, "r") as _f:
            _raw = _f[_dataset]
            if _index is not None:
                _arr = _raw[int(_index)]
            else:
                _arr = _raw[()]
        _elapsed_ms = (time.perf_counter() - _t0) * 1000

        # Format locator tuple (always 3 elements)
        _loc_str = f"`({_file}, {_dataset}, {_index})`"

        _rows.append(
            f"| {_s['label']} | {_s['query_desc']} | {_n_hits} | "
            f"`{_ent_key}` | {_loc_str} | `{_arr.shape}` | {_elapsed_ms:.0f} ms |"
        )

    _table_a = "\n".join(_rows)
    mo.md(f"""
    | Dataset | Query | Hits | Entity | Locator (file, dataset, index) | Shape | Time |
    |---------|-------|------|--------|-------------------------------|-------|------|
    {_table_a}

    `base_dir` for each dataset loaded from `datasets/*.yaml`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## VDP Deep Dive (Mode B)

    The following sections use **Mode B** (Tiled adapters) to visualize
    representative data from each dataset — chunked HTTP access, no
    file paths needed.

    VDP is the largest dataset (10K Hamiltonians, 11 artifacts each).
    Below we read a sample M(H) powder curve and an INS spectrum.
    """)
    return


@app.cell
def _(client, mo, np, plt):
    _h = client["VDP"]["H_636ce3e4"]
    _children = list(_h.keys())

    # Read M(H) curve
    _mh = _h["mh_powder_30T"].read()
    _h_grid = np.linspace(0, 1, len(_mh))

    # Read INS spectrum
    _ins = _h["ins_12meV"].read()

    fig_vdp, _axes = plt.subplots(1, 2, figsize=(12, 4))

    # M(H) curve
    _axes[0].plot(_h_grid, _mh, color="#2563eb", linewidth=2)
    _axes[0].set_xlabel("Reduced field h = H/Hmax")
    _axes[0].set_ylabel("M (magnetization)")
    _Ja = _h.metadata.get("Ja_meV", 0)
    _Dc = _h.metadata.get("Dc_meV", 0)
    _axes[0].set_title(f"M(H) powder 30T  (Ja={_Ja:.2f}, Dc={_Dc:.2f})")
    _axes[0].grid(True, alpha=0.3)

    # INS spectrum
    _vmax = np.percentile(_ins[_ins > 0], 99) if np.any(_ins > 0) else 1
    _im = _axes[1].imshow(_ins.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
    _axes[1].set_xlabel("Q index")
    _axes[1].set_ylabel("E index")
    _axes[1].set_title("INS Ei=12 meV  S(Q, E)")
    plt.colorbar(_im, ax=_axes[1], label="S(Q,E)")

    plt.tight_layout()

    mo.md(f"**Container `H_636ce3e4`** — {len(_children)} children: `{_children}`")
    return (fig_vdp,)


@app.cell
def _(fig_vdp):
    fig_vdp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDRIXS Deep Dive (Mode B)

    10,000 calculated RIXS spectra.  Each entity has a single `rixs`
    artifact — a 2D intensity map.
    """)
    return


@app.cell
def _(client, mo, np, plt):
    _h = client["EDRIXS"]["H_edx00000"]
    _spec = _h["rixs"].read()

    fig_edrixs, _ax = plt.subplots(figsize=(6, 4))
    _vmax = np.percentile(_spec[_spec > 0], 99) if np.any(_spec > 0) else 1
    _im = _ax.imshow(_spec.T, aspect="auto", origin="lower", cmap="inferno", vmin=0, vmax=_vmax)
    _ax.set_xlabel("Incident energy index")
    _ax.set_ylabel("Energy loss index")
    _ax.set_title(f"EDRIXS spectrum  (shape {_spec.shape})")
    plt.colorbar(_im, ax=_ax, label="Intensity")
    plt.tight_layout()

    _meta = {
        k: v for k, v in _h.metadata.items()
        if not k.startswith(("path_", "dataset_", "index_"))
    }
    _meta_keys = list(_meta.keys())[:6]
    _meta_vals = ", ".join(f"{k}={_meta[k]}" for k in _meta_keys)
    mo.md(f"**H_edx00000** metadata: {_meta_vals}")
    return (fig_edrixs,)


@app.cell
def _(fig_edrixs):
    fig_edrixs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NiPS3 Deep Dive (Mode B)

    7,616 simulated NiPS3 configurations, each with 6 artifacts
    (3 magnetization axes + 3 INS spectra).
    """)
    return


@app.cell
def _(client, mo, np, plt):
    _h = client["NiPS3_Multimodal"]["H_mm_1"]
    _children = list(_h.keys())

    _n_plots = min(3, len(_children))
    fig_nips3, _axes = plt.subplots(1, _n_plots, figsize=(14, 4))
    if not isinstance(_axes, np.ndarray):
        _axes = [_axes]

    for _i, _child_key in enumerate(_children[:3]):
        _arr = _h[_child_key].read()

        if _arr.ndim == 1:
            _axes[_i].plot(_arr, color="#059669", linewidth=2)
            _axes[_i].set_title(f"{_child_key}  ({_arr.shape})")
            _axes[_i].set_ylabel("Value")
            _axes[_i].grid(True, alpha=0.3)
        else:
            _vmax = np.percentile(_arr[_arr > 0], 99) if np.any(_arr > 0) else 1
            _im = _axes[_i].imshow(_arr.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=_vmax)
            _axes[_i].set_title(f"{_child_key}  ({_arr.shape})")
            plt.colorbar(_im, ax=_axes[_i])

    plt.tight_layout()

    _meta = {
        k: v for k, v in _h.metadata.items()
        if not k.startswith(("path_", "dataset_", "index_", "uid", "key"))
    }
    _meta_str = ", ".join(f"{k}={v}" for k, v in list(_meta.items())[:6])
    mo.md(f"**H_mm_1** — {len(_children)} children: `{_children}`\n\nMetadata: {_meta_str}")
    return (fig_nips3,)


@app.cell
def _(fig_nips3):
    fig_nips3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RIXS Deep Dive (Mode B)

    7 experimental RIXS runs, each with 6 artifacts (S, I, dI, Q, E, metadata arrays).
    """)
    return


@app.cell
def _(client, mo, np, plt):
    _h = client["RIXS"]["H_rixs_052"]
    _children = list(_h.keys())

    # Plot the S(Q,E) intensity map
    _s_arr = _h["S"].read()

    fig_rixs, _ax = plt.subplots(figsize=(6, 4))
    _vmax = np.percentile(np.abs(_s_arr[np.isfinite(_s_arr)]), 99) if np.any(np.isfinite(_s_arr)) else 1
    _im = _ax.imshow(_s_arr.T, aspect="auto", origin="lower", cmap="plasma", vmin=0, vmax=_vmax)
    _ax.set_xlabel("Q index")
    _ax.set_ylabel("E index")
    _ax.set_title(f"RIXS S(Q,E)  run 052  ({_s_arr.shape})")
    plt.colorbar(_im, ax=_ax, label="S(Q,E)")
    plt.tight_layout()

    mo.md(f"**H_rixs_052** — {len(_children)} children: `{_children}`")
    return (fig_rixs,)


@app.cell
def _(fig_rixs):
    fig_rixs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge & SEQUOIA Deep Dive (Mode B)

    **Challenge:** 1 benchmark entity with 9 artifacts (CEF + magnon spectra).
    **SEQUOIA:** 3 experimental neutron time-of-flight runs with ~26 artifacts each.
    """)
    return


@app.cell
def _(client, mo, np, plt):
    # Challenge
    _ch_h = client["Challenge"]["H_challeng"]
    _ch_children = list(_ch_h.keys())
    _ch_arr = _ch_h[_ch_children[0]].read()

    # SEQUOIA — pick a Q-path intensity child (small array) rather than MDE (79M rows)
    _seq_h = client["SEQUOIA"]["H_seq_Ei28"]
    _seq_children = list(_seq_h.keys())
    _seq_child = next(k for k in _seq_children if k.startswith("path_"))
    _seq_arr = _seq_h[_seq_child].read()

    fig_misc, _axes = plt.subplots(1, 2, figsize=(12, 4))

    # Challenge: CEF spectrum (2D)
    if _ch_arr.ndim == 2:
        _vmax = np.percentile(_ch_arr[_ch_arr > 0], 99) if np.any(_ch_arr > 0) else 1
        _im = _axes[0].imshow(_ch_arr.T, aspect="auto", origin="lower", cmap="cividis", vmin=0, vmax=_vmax)
        plt.colorbar(_im, ax=_axes[0])
    else:
        _axes[0].plot(_ch_arr, color="#7c3aed")
    _axes[0].set_title(f"Challenge: {_ch_children[0]}  ({_ch_arr.shape})")

    # SEQUOIA
    if _seq_arr.ndim == 2:
        _vmax_s = np.percentile(_seq_arr[_seq_arr > 0], 99) if np.any(_seq_arr > 0) else 1
        _im_s = _axes[1].imshow(_seq_arr.T, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=_vmax_s)
        plt.colorbar(_im_s, ax=_axes[1])
    else:
        _axes[1].plot(_seq_arr, color="#dc2626")
    _axes[1].set_title(f"SEQUOIA: {_seq_child}  ({_seq_arr.shape})")

    plt.tight_layout()

    mo.md(f"""
    - **H_challeng**: {len(_ch_children)} children — `{_ch_children[:4]}` ...
    - **H_seq_Ei28**: {len(_seq_children)} children — `{_seq_children[:4]}` ...
    """)
    return (fig_misc,)


@app.cell
def _(fig_misc):
    fig_misc
    return


@app.cell(hide_code=True)
def _(client, mo):
    mo.md(f"""
    ## Summary

    | Check | Status |
    |-------|--------|
    | Catalog online | **{len(client):,}** datasets |
    | VDP (10K entities, 11 artifacts each) | Read M(H) + INS |
    | EDRIXS (10K entities, 1 artifact each) | Read RIXS spectrum |
    | NiPS3 Multimodal (7.6K entities, 6 artifacts each) | Read magnetization + INS |
    | RIXS experimental (7 entities, 6 artifacts each) | Read S(Q,E) |
    | Challenge (1 entity, 9 artifacts) | Read CEF spectrum |
    | SEQUOIA (3 entities, ~26 artifacts each) | Read neutron TOF |
    | Mode A (query + h5py) | 6 datasets queried and read directly |

    **All 6 datasets ingested, queryable, and retrievable via both
    Mode A (expert h5py) and Mode B (Tiled adapters).**

    ### How to add a new dataset

    ```bash
    # 1. Write a manifest generator
    vi generators/gen_foo_manifest.py

    # 2. Write a dataset config
    echo "key: Foo\\ngenerator: gen_foo_manifest\\nbase_dir: /path/to/data" > datasets/foo.yaml

    # 3. Generate manifests
    uv run --with $BROKER broker-generate datasets/foo.yaml -n 1000

    # 4. Register into catalog
    uv run --with $BROKER broker-ingest datasets/foo.yaml
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
