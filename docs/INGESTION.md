# Ingestion Workflow

The generic broker ingests datasets **one modality at a time** using two CLI
commands: `broker-generate` (create Parquet manifests) and `broker-ingest`
(bulk-register into the SQLite catalog).

## Prerequisites

**1. Start the Tiled server:**

```bash
cd cwang31-data-broker/
export BROKER=/sdf/data/lcls/ds/prj/prjmaiqmag01/results/cwang31/proj-vdp-generic-broker/tiled_poc
uv run --with $BROKER tiled serve config config.yml --api-key secret
```

**2. Ensure `config.yml` lists the dataset's `base_dir`** under
`readable_storage` (see [Server Configuration](#server-configuration) below).

## Per-Dataset Workflow

Each dataset follows 4 steps:

### 1. Write a manifest generator

Create `generators/gen_<name>_manifest.py` with a `generate(output_dir, n_entities)` function
that produces two Parquet files:

- `<name>_entities.parquet` — one row per entity
- `<name>_artifacts.parquet` — one row per artifact

See [Manifest Schema](#manifest-schema) for column details.

### 2. Write a dataset config

Create `datasets/<name>.yaml`:

```yaml
label: Human Readable Name    # display name in the catalog
generator: gen_<name>_manifest # module name (no .py)
base_dir: /absolute/path/to/data
```

### 3. Generate manifests

```bash
uv run --with $BROKER broker-generate datasets/<name>.yaml -n <N>
```

This imports the generator module and calls `generate("manifests", n_entities=N)`,
producing `manifests/<name>_entities.parquet` and `manifests/<name>_artifacts.parquet`.

The `-n` flag controls how many entities to include (useful for testing with
a subset before full ingestion).

### 4. Register into catalog

```bash
uv run --with $BROKER broker-ingest datasets/<name>.yaml
```

This reads the Parquet manifests and bulk-inserts rows into `catalog.db`.
After ingestion, the dataset is queryable via the Tiled server.

## Manifest Schema

### Entity DataFrame

| Column | Description |
|--------|-------------|
| `uid` | Unique identifier (used as foreign key by artifacts) |
| `key` | Tiled catalog key, e.g. `H_636ce3e4` |
| *metadata columns* | Physics parameters specific to the dataset (e.g. `Ja_meV`, `Dc_meV`, `tenDq`) |

### Artifact DataFrame

| Column | Description |
|--------|-------------|
| `uid` | Foreign key linking to the parent entity |
| `type` | Artifact type, unique per (entity, type) pair, e.g. `mh_powder_30T` |
| `file` | HDF5 file path **relative to `base_dir`** |
| `dataset` | HDF5 internal dataset path, e.g. `/entry/data` |
| `index` | *(optional)* Batch index for datasets with multiple arrays per file |

The broker stores these as entity metadata with keys `path_<type>`,
`dataset_<type>`, and `index_<type>`.

## Ingestion History

The 6 datasets were ingested in three phases:

| Phase | Datasets | Entities | Artifacts | Cumulative Data |
|-------|----------|----------|-----------|-----------------|
| Dec 2025 | VDP (entities only) | 10,000 | 0 | 0 GB |
| Jan 2026 | VDP (full artifact deployment) | 10,000 | 110,000 | 110 GB |
| Feb 2026 | EDRIXS, NiPS3 Multimodal, RIXS, Challenge, SEQUOIA | 27,627 | 165,823 | 151 GB |

Commands used for the Feb 2026 phase (each run separately):

```bash
uv run --with $BROKER broker-generate datasets/edrixs.yaml -n 10000
uv run --with $BROKER broker-ingest   datasets/edrixs.yaml

uv run --with $BROKER broker-generate datasets/nips3_multimodal_synthetic.yaml -n 10000
uv run --with $BROKER broker-ingest   datasets/nips3_multimodal_synthetic.yaml

uv run --with $BROKER broker-generate datasets/rixs.yaml
uv run --with $BROKER broker-ingest   datasets/rixs.yaml

uv run --with $BROKER broker-generate datasets/challenge.yaml
uv run --with $BROKER broker-ingest   datasets/challenge.yaml

uv run --with $BROKER broker-generate datasets/sequoia.yaml
uv run --with $BROKER broker-ingest   datasets/sequoia.yaml
```

## Server Configuration

`config.yml` maps each dataset's HDF5 base directory as `readable_storage`
so Tiled can serve the files:

```yaml
trees:
  - path: /
    tree: catalog
    args:
      uri: "sqlite:///catalog.db"
      writable_storage: "storage"
      readable_storage:
        - /path/to/vdp/data          # VDP
        - /path/to/EDRIXS            # EDRIXS
        - /path/to/NiPS3/data        # NiPS3 Multimodal
        - /path/to/converted/rixs    # RIXS (NetCDF3 -> HDF5)
        - /path/to/converted/challenge  # Challenge (CSV -> HDF5)
        - /path/to/neutrons_SEQUOIA  # SEQUOIA
```

Each entry must match the `base_dir` from the corresponding `datasets/*.yaml`.
When adding a new dataset, add its `base_dir` here before running `broker-ingest`.

## Dataset Summary

| Dataset | Entities | Artifacts/Entity | Source Format | Notes |
|---------|----------|-------------------|---------------|-------|
| VDP | 10,000 | 11 | Individual HDF5 | Sunny.jl simulations |
| EDRIXS | 10,000 | 1 | Monolithic HDF5 | Index-based batching |
| NiPS3 Multimodal | 7,616 | 6 | Individual HDF5 | INS + magnetization |
| RIXS | 7 | 6 | NetCDF3 -> HDF5 | Converted at generation time |
| Challenge | 1 | 9 | CSV -> HDF5 | Converted at generation time |
| SEQUOIA | 3 | ~26 | HDF5 + NXS | Q-path intensities + MDE |
