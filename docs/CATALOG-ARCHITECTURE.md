# Catalog Architecture

How the Tiled SQLite catalog stores multi-modal data, and known
scalability considerations.

## Current Design: Single Flat Table

All entities from every dataset live in one `nodes` table:

```
nodes (193,451 rows: 27,627 entities + 165,823 artifacts)
├── id                INTEGER PRIMARY KEY
├── parent            INTEGER       -- 0 for entities, entity_id for artifacts
├── key               TEXT          -- "H_636ce3e4", "mh_powder_30T", ...
├── structure_family  TEXT          -- "container" (entity) or "array" (artifact)
├── metadata          TEXT (JSON)   -- all physics params + locator tuples
└── specs, access_blob, ...
```

Metadata is a **JSON blob per row**.  A VDP entity stores
`{"Ja_meV": 0.5, "Dc_meV": -0.1, ...}` while an EDRIXS entity stores
`{"tenDq": 3.4, "F2_dd": 1.2, ...}`.  There is no `source` or
`modality` column — the only way to distinguish datasets is by the
presence of dataset-specific parameter keys.

Supporting tables:

| Table | Rows | Purpose |
|-------|------|---------|
| `nodes` | 193,451 | Entity and artifact tree |
| `nodes_closure` | ~358K | Ancestor/descendant lookups |
| `data_sources` | 165,823 | Links artifacts to HDF5 files |
| `assets` | 117,632 | Deduplicated file URIs |
| `data_source_asset_association` | 165,823 | Many-to-many join |
| `metadata_fts5` | — | Full-text search (text only) |

## How Queries Work

A Tiled query like:

```python
client.search(Key("tenDq") > 3.0)
```

translates to:

```sql
SELECT ... FROM nodes
WHERE parent = 0
  AND json_extract(metadata, '$.tenDq') > 3.0
```

### Indexes available

| Index | Columns | Helps with |
|-------|---------|------------|
| `ix_nodes_parent` | `parent` | Narrowing to top-level entities |
| `top_level_metadata` | `parent, time_created, id, metadata, access_blob` | Covering index (avoids table lookup) |
| `metadata_fts5` | Full-text on metadata JSON | Text search only, not numeric |

### The scan problem

`json_extract()` on a JSON column **cannot be indexed** in SQLite.
Every query scans all 27,627 top-level entities, extracting the
requested key from each JSON blob — including entities from unrelated
datasets that don't even have the key.

At 27K entities this is still fast (single-digit milliseconds).  At
100K+ entities across many modalities, every query pays the cost of
scanning all modalities.

## No Modality Filter

The dataset `label` from `datasets/*.yaml` is used only for logging
during ingestion.  It is **not stored** as entity metadata.  There is
currently no way to write:

```python
client.search(Key("source") == "EDRIXS")  # does not work today
```

## Possible Improvements

| Approach | Effort | Benefit |
|----------|--------|---------|
| Inject `source` into entity metadata at ingest time | Low | Enables `Key("source") == "EDRIXS"` chaining; still JSON-scanned |
| SQLite generated column + index on `json_extract(metadata, '$.source')` | Medium | True indexed pre-filtering (requires SQLite 3.31+) |
| Add a real `modality` column to the `nodes` table | Medium | Native SQL index; requires Tiled schema change |
| Separate `catalog.db` per modality | Low | Full isolation; loses unified cross-modal queries |

### Recommended short-term fix

Inject `"source": "<label>"` into every entity's metadata during
`broker-ingest`, sourced from the YAML config's `label` field.  This
is a one-line change in `bulk_register.py` (or `catalog.py`) and
immediately gives users a modality filter — even though the underlying
scan cost is unchanged, it makes intent explicit and is a prerequisite
for any future indexing optimization.

### Longer-term considerations

- **PostgreSQL migration**: Tiled supports PostgreSQL catalogs, which
  offer `JSONB` with GIN indexes — these can index into JSON paths
  natively, eliminating the full-scan problem.
- **Partitioned catalogs**: If cross-modal queries are rare, separate
  catalogs per modality avoid the scan entirely while keeping the same
  broker workflow.
- **Materialized views**: Pre-compute per-modality views with
  extracted columns for the most common query patterns.
