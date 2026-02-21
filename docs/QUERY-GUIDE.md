# Query Guide

Practical reference for querying the multi-modal Tiled catalog.  Covers
dataset discovery, entity-level physics queries, data retrieval
performance, and known limitations.

All examples assume a running server and this setup:

```python
from tiled.client import from_uri
from tiled.queries import Key, In, KeyPresent

client = from_uri("http://localhost:8006", api_key="secret")
```

---

## 1. Two Query Levels

The catalog has a two-level hierarchy:

```
root/                          ← Level 1: dataset containers
  VDP/                         ← Level 2: entity containers
    H_636ce3e4/
    H_7a1b2c3d/
  EDRIXS/
    H_edx00000/
  ...
```

- **Level 1 (root):** Search dataset metadata — material, facility,
  data_type, organization.  Scans ~6 rows.
- **Level 2 (within a dataset):** Search entity metadata — physics
  parameters like `Ja_meV`, `tenDq`, `experiment_type`.  Scans up to
  10K rows per dataset.

Root search **never** sees entity-level fields.  To query physics
parameters you must first scope to a dataset:

```python
# This returns 0 results — Ja_meV is an entity field, not a dataset field
client.search(Key("Ja_meV") > 0.5)

# This works — scoped to VDP dataset
client["VDP"].search(Key("Ja_meV") > 0.5)
```

---

## 2. Dataset Discovery (Level 1)

Search by metadata fields on dataset containers.  These queries scan
only the ~6 dataset rows and return in milliseconds.

### Equality

```python
# Find all datasets about NiPS3 (4 results, ~12 ms)
client.search(Key("material") == "NiPS3")

# Find experimental datasets (2 results)
client.search(Key("data_type") == "experimental")
```

### Membership

```python
# Find datasets for NiPS3 OR YbBi2IO4 (5 results)
client.search(In("material", ["NiPS3", "YbBi2IO4"]))
```

### Field existence

```python
# Find datasets that have a "facility" field (2 results)
client.search(KeyPresent("facility"))

# Find datasets missing the "producer" field (4 results)
client.search(KeyPresent("producer", exists=False))
```

### Combined (AND)

Chain `.search()` calls for AND logic:

```python
# NiPS3 AND experimental (2 results)
client.search(Key("material") == "NiPS3").search(
    Key("data_type") == "experimental"
)
```

---

## 3. Entity Queries (Level 2)

Physics parameter search within a dataset.  Always scope to a dataset
first.

### Range filters

```python
vdp = client["VDP"]

# Single range (~2,500 hits, ~116 ms)
subset = vdp.search(Key("Ja_meV") > 0.5)

# Multi-field AND — chain .search() calls (~123 ms)
subset = vdp.search(Key("Ja_meV") > 0.5).search(Key("Dc_meV") < -0.5)
```

### String equality

```python
rixs = client["RIXS"]
subset = rixs.search(Key("experiment_type") == "energy_scan")
```

### Count without fetching data

```python
# Fast count — does not transfer entity data (~15 ms)
n = len(vdp.search(Key("Ja_meV") > 0.5))
```

### Pagination

```python
# First 10 keys only (~17 ms)
keys = list(vdp.keys()[:10])

# First 10 entities
entities = list(vdp.values()[:10])
```

### Sorting

```python
# Sort by Ja_meV ascending (~107 ms)
# Note: must use tuple (field, direction), not a bare string
sorted_vdp = vdp.sort(("Ja_meV", 1))    # 1 = ascending
keys = list(sorted_vdp.keys()[:5])
```

### Distinct values

```python
# Unique values of spin_s with counts (~1.8 sec — slow on 10K rows)
d = vdp.distinct("spin_s", counts=True)
```

---

## 4. Data Retrieval Performance

Two retrieval modes are available.  Benchmarks are from `tests/test_queries.py`
on the VDP dataset (entity `H_636ce3e4`).

| Operation | Mode A (h5py) | Mode B (Tiled) |
|-----------|---------------|----------------|
| Single array (`mh_powder_30T`) | ~19 ms | ~951 ms |
| Sliced read (`ins_12meV[100:200]`) | — | ~83 ms |
| Batch read (10 entities) | ~203 ms | ~871 ms |

### When to use each mode

**Mode A (expert, h5py):**
- Lowest latency for bulk reads
- Best for ML training loops that need many arrays
- Requires knowing `base_dir` (from dataset YAML config)
- Reads full arrays only (no server-side slicing)

```python
h = vdp["H_636ce3e4"]
meta = h.metadata
file_rel = meta["path_mh_powder_30T"]
dataset  = meta["dataset_mh_powder_30T"]
index    = meta.get("index_mh_powder_30T")

import h5py
with h5py.File(f"{base_dir}/{file_rel}", "r") as f:
    arr = f[dataset][()] if index is None else f[dataset][int(index)]
```

**Mode B (visualizer, Tiled adapters):**
- No file paths needed — data served via HTTP
- Supports slicing (`arr[100:200]`) for partial reads
- Better for dashboards and interactive exploration
- Higher latency per call but simpler API

```python
arr = vdp["H_636ce3e4"]["mh_powder_30T"].read()       # full array
arr = vdp["H_636ce3e4"]["ins_12meV"][100:200]          # sliced
```

---

## 5. Limitations & Gotchas

### Root search only sees dataset containers

Entity-level metadata fields (`Ja_meV`, `tenDq`, etc.) are invisible at
the root level.  To search across all datasets, iterate:

```python
for dk in client.keys():
    hits = client[dk].search(Key("Ja_meV") > 0.5)
    if len(hits) > 0:
        print(f"{dk}: {len(hits)} matches")
```

### `distinct()` is slow

On 10K-row datasets, `distinct()` takes ~1.8 seconds.  Avoid in
interactive notebooks.  For precomputed distinct values, query the
Parquet manifests directly.

### Sort API requires a tuple

```python
# Correct:
vdp.sort(("Ja_meV", 1))       # 1 = ascending, -1 = descending

# Wrong — raises an error:
vdp.sort("Ja_meV")
```

### No OR queries across fields

Tiled's `.search()` only supports AND (intersection).  There is no
built-in OR combinator.  For OR logic, run separate queries and merge
results client-side:

```python
set_a = set(vdp.search(Key("Ja_meV") > 0.5).keys())
set_b = set(vdp.search(Key("Dc_meV") < -0.5).keys())
union = set_a | set_b
```

### Stale server after re-ingestion

If you re-run `broker-ingest`, the server may cache old metadata.
Restart the Tiled server to pick up changes.

### Combined queries at root level

Chained `.search()` calls are ANDed together.
This works at both root and entity levels but cannot express OR.
