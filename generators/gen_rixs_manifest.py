"""
Generate RIXS manifests in the generic broker standard.

Converts NetCDF3 files to individual HDF5 files (one per run number),
then generates entity + artifact manifests.

Source files:
    /sdf/.../data-source/LS/S_52.nc        — 1 run (energy scan)
    /sdf/.../data-source/LS/S_139_140.nc   — 6 runs (time delay)

Each run becomes one entity with 6 artifacts:
    S           — RIXS spectrum, shape (motor, energy)
    I_rn        — Raw intensity, shape (motor, energy)
    I0          — Incident intensity, shape (motor,)
    counts      — Photon counts, shape (motor,)
    energy_grid — Energy axis, shape (energy,)
    motor_pos   — Motor positions, shape (motor,)

Interface:
    generate(output_dir, n_entities=10) → (ent_df, art_df)
"""

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RIXS_SOURCE_DIR = "/sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/LS"
CONVERTED_DIR = "/sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/cwang31-data-broker/converted/rixs"

# Map source file to experiment type
SOURCE_FILES = [
    ("S_52.nc", "energy_scan"),
    ("S_139_140.nc", "time_delay"),
]

# Artifacts to extract per run (name → HDF5 dataset name in converted file)
ARTIFACT_TYPES = ["S", "I_rn", "I0", "counts", "energy_grid", "motor_pos"]


def _convert_netcdf_to_hdf5(nc_path, experiment_type, converted_dir):
    """Convert one NetCDF file to individual HDF5 files, one per run.

    Returns list of (run_number, h5_filename, attrs) tuples.
    """
    import xarray as xr

    converted_dir = Path(converted_dir)
    converted_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    run_numbers = ds.coords["rn"].values
    attrs = dict(ds.attrs)

    results = []
    for idx, rn in enumerate(run_numbers):
        rn_int = int(rn)
        h5_name = f"rixs_{rn_int:03d}.h5"
        h5_path = converted_dir / h5_name

        with h5py.File(h5_path, "w") as f:
            # 2D arrays: select this run along rn dimension
            f.create_dataset("S", data=ds["S"].values[idx])           # (motor, energy)
            f.create_dataset("I_rn", data=ds["I_rn"].values[idx])     # (motor, energy)

            # 1D arrays: select this run along rn dimension
            f.create_dataset("I0", data=ds["I0"].values[idx])         # (motor,)
            f.create_dataset("counts", data=ds["counts"].values[idx]) # (motor,)

            # Shared coordinate arrays (same for all runs in this file)
            f.create_dataset("energy_grid", data=ds.coords["energy"].values)  # (energy,)
            f.create_dataset("motor_pos", data=ds.coords["motor"].values)     # (motor,)

            # Store metadata as attributes
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["run_number"] = rn_int
            f.attrs["experiment_type"] = experiment_type

        results.append((rn_int, h5_name, attrs))

    ds.close()
    return results


def generate(output_dir, n_entities=10):
    """Generate RIXS manifests in the generic broker standard.

    Args:
        output_dir: Directory to write Parquet files.
        n_entities: Max number of entities to include.

    Returns:
        (ent_df, art_df): Entity and artifact DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ent_records = []
    art_records = []

    for nc_file, exp_type in SOURCE_FILES:
        nc_path = Path(RIXS_SOURCE_DIR) / nc_file
        if not nc_path.exists():
            print(f"  WARNING: {nc_path} not found, skipping")
            continue

        print(f"  Converting {nc_file} → HDF5...")
        results = _convert_netcdf_to_hdf5(nc_path, exp_type, CONVERTED_DIR)

        for rn, h5_name, attrs in results:
            if len(ent_records) >= n_entities:
                break

            uid = f"rixs_{rn:03d}"

            # Entity record with detector geometry metadata
            record = {
                "uid": uid,
                "key": f"H_{uid[:8]}",
                "run_number": rn,
                "experiment_type": exp_type,
            }
            for attr_name in ["twotheta", "chi", "phi", "theta", "x", "y", "z", "dety"]:
                if attr_name in attrs:
                    record[attr_name] = float(attrs[attr_name])
            ent_records.append(record)

            # Artifact records
            for art_type in ARTIFACT_TYPES:
                art_records.append({
                    "uid": uid,
                    "type": art_type,
                    "file": h5_name,
                    "dataset": art_type,
                })

    ent_df = pd.DataFrame(ent_records)
    art_df = pd.DataFrame(art_records)

    # Write Parquet files
    ent_out = output_dir / "rixs_entities.parquet"
    art_out = output_dir / "rixs_artifacts.parquet"
    ent_df.to_parquet(ent_out, index=False)
    art_df.to_parquet(art_out, index=False)

    print(f"  RIXS output: {len(ent_df)} entities, {len(art_df)} artifacts")
    print(f"  Written to: {output_dir}")

    return ent_df, art_df
