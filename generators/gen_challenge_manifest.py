"""
Generate Challenge (YbBi2IO4) manifests in the generic broker standard.

Converts CSV experimental data files to a single HDF5 file, then
generates entity + artifact manifests.

Source files (all CSV):
    cef_intensities_11K.csv  — CEF spectrum intensities (264 × 101)
    cef_energies_11K.csv     — CEF energy axis (101,)
    cef_Qs_11K.csv           — CEF Q values (264,)
    magnon_intensities.csv   — Magnon spectrum (90 × 39)
    magnon_energies.csv      — Magnon energy axis (39,)
    magnon_Qs.csv            — Magnon Q values (90,)
    magdata_0p4K.csv         — Magnetization at 0.4K (N × 2)
    magdata_1p8K.csv         — Magnetization at 1.8K (N × 2)
    magdata_5p0K.csv         — Magnetization at 5.0K (N × 2)

One entity (the YbBi2IO4 benchmark), 9 artifacts.

Interface:
    generate(output_dir, n_entities=10) → (ent_df, art_df)
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CHALLENGE_DIR = "/sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/challenge/experimental_data"
CONVERTED_DIR = "/sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/cwang31-data-broker/converted/challenge"

# CSV file → HDF5 dataset name mapping
CSV_TO_DATASET = {
    "cef_intensities_11K.csv": "cef_spectrum",
    "cef_energies_11K.csv":    "cef_energies",
    "cef_Qs_11K.csv":          "cef_Qs",
    "magnon_intensities.csv":  "magnon_spectrum",
    "magnon_energies.csv":     "magnon_energies",
    "magnon_Qs.csv":           "magnon_Qs",
    "magdata_0p4K.csv":        "mag_0p4K",
    "magdata_1p8K.csv":        "mag_1p8K",
    "magdata_5p0K.csv":        "mag_5p0K",
}


def _convert_csvs_to_hdf5(source_dir, converted_dir):
    """Convert all CSV files to a single HDF5 file.

    Returns the HDF5 filename (relative).
    """
    source_dir = Path(source_dir)
    converted_dir = Path(converted_dir)
    converted_dir.mkdir(parents=True, exist_ok=True)

    h5_name = "YbBi2IO4.h5"
    h5_path = converted_dir / h5_name

    with h5py.File(h5_path, "w") as f:
        for csv_file, ds_name in CSV_TO_DATASET.items():
            csv_path = source_dir / csv_file
            data = np.loadtxt(csv_path, delimiter=",")
            f.create_dataset(ds_name, data=data)
            print(f"    {csv_file} → {ds_name}: shape={data.shape}")

    return h5_name


def generate(output_dir, n_entities=10):
    """Generate Challenge manifests in the generic broker standard.

    Args:
        output_dir: Directory to write Parquet files.
        n_entities: Max number of entities (always 1 for this dataset).

    Returns:
        (ent_df, art_df): Entity and artifact DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Converting CSVs → HDF5...")
    h5_name = _convert_csvs_to_hdf5(CHALLENGE_DIR, CONVERTED_DIR)

    # Single entity
    uid = "challenge_YbBi2IO4"
    ent_records = [{
        "uid": uid,
        "key": "H_challeng",
        "material_formula": "YbBi2IO4",
        "structure": "2D quantum magnet",
        "n_parameters": 23,
    }]

    # One artifact per dataset
    art_records = []
    for ds_name in CSV_TO_DATASET.values():
        art_records.append({
            "uid": uid,
            "type": ds_name,
            "file": h5_name,
            "dataset": ds_name,
        })

    ent_df = pd.DataFrame(ent_records)
    art_df = pd.DataFrame(art_records)

    # Write Parquet files
    ent_out = output_dir / "challenge_entities.parquet"
    art_out = output_dir / "challenge_artifacts.parquet"
    ent_df.to_parquet(ent_out, index=False)
    art_df.to_parquet(art_out, index=False)

    print(f"  Challenge output: {len(ent_df)} entities, {len(art_df)} artifacts")
    print(f"  Written to: {output_dir}")

    return ent_df, art_df
