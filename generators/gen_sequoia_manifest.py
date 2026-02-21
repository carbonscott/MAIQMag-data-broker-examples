"""
Generate SEQUOIA manifests in the generic broker standard.

Reads Q-path data from NIPS_all_Qpaths.h5 (70 groups) and registers
MDE raw data files. Groups are assigned to incident energy entities
by parsing the ``source_txt`` attribute for the Ei value.

Entities: 3 (one per Ei: 28, 60, 100 meV)
Artifacts per entity:
    - Q-path Intensity arrays (~23 each)
    - 1 merged MDE file reference
    - 1 background MDE file reference

Interface:
    generate(output_dir, n_entities=10) → (ent_df, art_df)

Source data:
    /sdf/.../neutrons_SEQUOIA/NIPS_all_Qpaths.h5
    /sdf/.../neutrons_SEQUOIA/merged_mde/*.nxs
"""

import os
import re
from pathlib import Path

import h5py
import pandas as pd


SEQUOIA_DIR = "/sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/data-tlinker/neutrons_SEQUOIA"
QPATHS_H5 = os.path.join(SEQUOIA_DIR, "NIPS_all_Qpaths.h5")
MDE_DIR = os.path.join(SEQUOIA_DIR, "merged_mde")

# MDE files mapped by Ei
MDE_MERGED = {
    28:  "merged_mde/merged_mde_NiPS3_rotation_28meV_4K.nxs",
    60:  "merged_mde/merged_mde_NiPS3_rotation_60meV_4K.nxs",
    100: "merged_mde/merged_mde_NiPS3_rotationHighRes_100meV_4K.nxs",
}
MDE_BACKGROUND = {
    28:  "merged_mde/mde_bkg_28meV_4K.nxs",
    60:  "merged_mde/mde_bkg_60meV_4K.nxs",
    100: "merged_mde/mde_bkg_100meV_4K.nxs",
}

# Crystal parameters for NiPS3
CRYSTAL_PARAMS = {
    "sample": "NiPS3",
    "temperature_K": 4,
    "a_angstrom": 5.812,
    "b_angstrom": 10.067,
    "c_angstrom": 6.626,
    "alpha_deg": 90.0,
    "beta_deg": 107.16,
    "gamma_deg": 90.0,
}


def _parse_ei_from_source(source_txt):
    """Extract incident energy (meV) from source_txt attribute.

    Example: '..._Ei28meV_4K.txt' → 28
    """
    match = re.search(r"Ei(\d+)meV", source_txt)
    if match:
        return int(match.group(1))
    return None


def _build_qpath_map(h5_path):
    """Read NIPS_all_Qpaths.h5 and map each path to its Ei.

    Returns dict: {ei_meV: [list of path_names]}
    """
    ei_to_paths = {28: [], 60: [], 100: []}

    with h5py.File(h5_path, "r") as f:
        for path_name in sorted(f.keys()):
            group = f[path_name]
            source_txt = group.attrs.get("source_txt", "")
            if isinstance(source_txt, bytes):
                source_txt = source_txt.decode()

            ei = _parse_ei_from_source(source_txt)
            if ei in ei_to_paths:
                ei_to_paths[ei].append(path_name)
            else:
                print(f"  WARNING: Could not parse Ei from {path_name}: {source_txt}")

    return ei_to_paths


def generate(output_dir, n_entities=10):
    """Generate SEQUOIA manifests in the generic broker standard.

    Args:
        output_dir: Directory to write Parquet files.
        n_entities: Max number of entities (up to 3).

    Returns:
        (ent_df, art_df): Entity and artifact DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Reading Q-path structure from {Path(QPATHS_H5).name}...")
    ei_to_paths = _build_qpath_map(QPATHS_H5)

    ent_records = []
    art_records = []

    for ei_meV in sorted(ei_to_paths.keys()):
        if len(ent_records) >= n_entities:
            break

        paths = ei_to_paths[ei_meV]
        uid = f"seq_Ei{ei_meV}"

        # Entity record
        record = {
            "uid": uid,
            "key": f"H_{uid[:8]}",
            "incident_energy_meV": ei_meV,
        }
        record.update(CRYSTAL_PARAMS)
        ent_records.append(record)

        # Q-path Intensity artifacts
        for path_name in paths:
            art_records.append({
                "uid": uid,
                "type": f"{path_name}_intensity",
                "file": "NIPS_all_Qpaths.h5",
                "dataset": f"{path_name}/Intensity",
            })

        # Merged MDE artifact
        if ei_meV in MDE_MERGED:
            art_records.append({
                "uid": uid,
                "type": "mde_merged",
                "file": MDE_MERGED[ei_meV],
                "dataset": "MDEventWorkspace/event_data/event_data",
            })

        # Background MDE artifact
        if ei_meV in MDE_BACKGROUND:
            art_records.append({
                "uid": uid,
                "type": "mde_background",
                "file": MDE_BACKGROUND[ei_meV],
                "dataset": "MDEventWorkspace/event_data/event_data",
            })

        print(f"    Ei={ei_meV} meV: {len(paths)} Q-paths + 2 MDE files")

    ent_df = pd.DataFrame(ent_records)
    art_df = pd.DataFrame(art_records)

    # Write Parquet files
    ent_out = output_dir / "sequoia_entities.parquet"
    art_out = output_dir / "sequoia_artifacts.parquet"
    ent_df.to_parquet(ent_out, index=False)
    art_df.to_parquet(art_out, index=False)

    print(f"  SEQUOIA output: {len(ent_df)} entities, {len(art_df)} artifacts")
    print(f"  Written to: {output_dir}")

    return ent_df, art_df
