# Running the Data Broker

## Prerequisites

Set the `BROKER` variable to the broker package path:

```bash
export BROKER=/sdf/data/lcls/ds/prj/prjmaiqmag01/results/cwang31/proj-vdp-generic-broker/tiled_poc
export UV_CACHE_DIR=/sdf/data/lcls/ds/prj/prjmaiqmag01/results/cwang31/.UV_CACHE
```

All commands below are run from the `cwang31-data-broker/` directory:

```bash
cd /sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/cwang31-data-broker
```

## Start the Tiled Server

```bash
uv run --with $BROKER tiled serve config config.yml --api-key secret
```

This starts the server on `http://localhost:8006` serving all 6 datasets
from `catalog.db`.  Keep this terminal open.

## Launch the Demo Notebook

In a **second terminal** (with the same exports):

```bash
cd /sdf/data/lcls/ds/prj/prjmaiqmag01/results/data-source/cwang31-data-broker

uv run --with $BROKER \
  --with marimo --with matplotlib --with numpy \
  --with h5py --with 'ruamel.yaml' \
  marimo edit demo_multimodal.py
```

This opens the multimodal ingestion showcase in your browser.  The
notebook connects to the server on port 8006 and demonstrates dataset
discovery, Mode A (h5py) and Mode B (Tiled) retrieval across all 6
datasets.

## Other Docs

| Document | Description |
|----------|-------------|
| [INGESTION.md](INGESTION.md) | How to generate manifests and ingest datasets |
| [QUERY-GUIDE.md](QUERY-GUIDE.md) | Query capabilities, performance, and gotchas |
