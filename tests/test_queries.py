# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tiled[server]",
#     "pandas",
#     "h5py",
#     "ruamel.yaml",
# ]
# ///
"""
Systematic query capability & performance tests.

Runs 23 tests against a live Tiled server (port 8006) covering:
  Level 1: Dataset discovery (root-level metadata search)
  Level 2: Entity queries (within-dataset, physics parameters)
  Level 3: Data retrieval (Mode A vs Mode B performance)
  Level 4: Cross-dataset queries (root-level entity search)

Usage:
  # Start server first:
  uv run --with $BROKER tiled serve config config.yml --api-key secret

  # Then run tests:
  uv run --with 'tiled[server]' --with pandas --with h5py \
    --with 'ruamel.yaml' python tests/test_queries.py
"""

import os
import time
from pathlib import Path

import h5py
from ruamel.yaml import YAML
from tiled.client import from_uri
from tiled.queries import Key, In, KeyPresent


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

TILED_URL = os.environ.get("TILED_URL", "http://localhost:8006")
API_KEY = os.environ.get("TILED_API_KEY", "secret")


def load_base_dirs():
    """Load base_dir from each dataset YAML config."""
    yaml = YAML()
    base_dirs = {}
    for cfg_path in sorted(Path("datasets").glob("*.yaml")):
        with open(cfg_path) as f:
            cfg = yaml.load(f)
        base_dirs[cfg["key"]] = cfg["base_dir"]
    return base_dirs


def run_test(name, fn):
    """Run a test function, return (name, passed, count, time_ms, error)."""
    t0 = time.perf_counter()
    try:
        passed, count = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return name, passed, count, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return name, False, 0, elapsed, str(e)


# ---------------------------------------------------------------------------
# Level 1: Dataset Discovery
# ---------------------------------------------------------------------------

def level1_tests(client):
    tests = []

    # 1. Search by material
    def t1():
        r = client.search(Key("material") == "NiPS3")
        n = len(r)
        return n == 4, n
    tests.append(("L1.1  material == NiPS3", t1))

    # 2. Search by data_type
    def t2():
        r = client.search(Key("data_type") == "experimental")
        n = len(r)
        return n == 2, n
    tests.append(("L1.2  data_type == experimental", t2))

    # 3. Chained: material + data_type
    def t3():
        r = client.search(Key("material") == "NiPS3").search(
            Key("data_type") == "experimental"
        )
        n = len(r)
        return n == 2, n
    tests.append(("L1.3  NiPS3 + experimental (AND)", t3))

    # 4. Search by facility
    def t4():
        r = client.search(Key("facility") == "SNS")
        n = len(r)
        return n == 1, n
    tests.append(("L1.4  facility == SNS", t4))

    # 5. Membership: In()
    def t5():
        r = client.search(In("material", ["NiPS3", "YbBi2IO4"]))
        n = len(r)
        return n == 5, n
    tests.append(("L1.5  material IN [NiPS3, YbBi2IO4]", t5))

    # 6. Field existence: KeyPresent
    def t6():
        r = client.search(KeyPresent("facility"))
        n = len(r)
        return n == 2, n
    tests.append(("L1.6  KeyPresent(facility)", t6))

    # 7. Field absence
    def t7():
        r = client.search(KeyPresent("producer", exists=False))
        n = len(r)
        return n == 4, n
    tests.append(("L1.7  producer missing", t7))

    return tests


# ---------------------------------------------------------------------------
# Level 2: Entity Queries
# ---------------------------------------------------------------------------

def level2_tests(client):
    tests = []

    vdp = client["VDP"]
    edrixs = client["EDRIXS"]
    rixs = client["RIXS"]
    challenge = client["Challenge"]

    # 8. Single range filter
    def t8():
        r = vdp.search(Key("Ja_meV") > 0.5)
        n = len(r)
        return n > 0, n
    tests.append(("L2.8  VDP: Ja > 0.5", t8))

    # 9. Multi-field AND
    def t9():
        r = vdp.search(Key("Ja_meV") > 0.5).search(Key("Dc_meV") < -0.5)
        n = len(r)
        return n > 0, n
    tests.append(("L2.9  VDP: Ja > 0.5 AND Dc < -0.5", t9))

    # 10. EDRIXS multi-field
    def t10():
        r = edrixs.search(Key("tenDq") > 3.0).search(Key("Gam_c") < 0.2)
        n = len(r)
        return n > 0, n
    tests.append(("L2.10 EDRIXS: tenDq > 3 AND Gam_c < 0.2", t10))

    # 11. String equality
    def t11():
        r = rixs.search(Key("experiment_type") == "energy_scan")
        n = len(r)
        return n > 0, n
    tests.append(("L2.11 RIXS: experiment_type == energy_scan", t11))

    # 12. Tiny dataset equality
    def t12():
        r = challenge.search(Key("material_formula") == "YbBi2IO4")
        n = len(r)
        return n == 1, n
    tests.append(("L2.12 Challenge: material_formula == YbBi2IO4", t12))

    # 13. Count-only (len)
    def t13():
        n = len(vdp)
        return n == 10000, n
    tests.append(("L2.13 VDP: len() count-only", t13))

    # 14. Pagination
    def t14():
        keys = list(vdp.keys()[:10])
        return len(keys) == 10, len(keys)
    tests.append(("L2.14 VDP: .keys()[:10] pagination", t14))

    # 15. Sorted retrieval — sort(("field", direction)) where 1=ascending
    def t15():
        sorted_vdp = vdp.sort(("Ja_meV", 1))
        keys = list(sorted_vdp.keys()[:5])
        return len(keys) == 5, len(keys)
    tests.append(("L2.15 VDP: sorted by Ja_meV, first 5", t15))

    # 16. Distinct values
    def t16():
        d = vdp.distinct("spin_s", counts=True)
        # d is a dict with metadata_key -> list of {value, count}
        n_unique = len(d.get("metadata", {}).get("spin_s", []))
        return n_unique > 0, n_unique
    tests.append(("L2.16 VDP: distinct(spin_s)", t16))

    return tests


# ---------------------------------------------------------------------------
# Level 3: Data Retrieval (Mode A vs Mode B)
# ---------------------------------------------------------------------------

def level3_tests(client, base_dirs):
    tests = []

    vdp = client["VDP"]
    sample_key = "H_636ce3e4"

    # 17. Mode B: full array read
    def t17():
        arr = vdp[sample_key]["mh_powder_30T"].read()
        return arr.shape[0] > 0, arr.shape[0]
    tests.append(("L3.17 Mode B: full array read", t17))

    # 18. Mode B: partial read (slicing)
    def t18():
        arr = vdp[sample_key]["ins_12meV"][100:200]
        return arr.shape[0] == 100, arr.shape[0]
    tests.append(("L3.18 Mode B: sliced read [100:200]", t18))

    # 19. Mode A: query -> locator -> h5py
    def t19():
        h = vdp[sample_key]
        meta = h.metadata
        file_rel = meta["path_mh_powder_30T"]
        dataset = meta["dataset_mh_powder_30T"]
        full_path = f"{base_dirs['VDP']}/{file_rel}"
        with h5py.File(full_path, "r") as f:
            arr = f[dataset][()]
        return arr.shape[0] > 0, arr.shape[0]
    tests.append(("L3.19 Mode A: locator -> h5py read", t19))

    # 20. Mode B batch: 10 entities
    def t20():
        keys = list(vdp.keys()[:10])
        total = 0
        for k in keys:
            arr = vdp[k]["mh_powder_30T"].read()
            total += arr.shape[0]
        return total > 0, len(keys)
    tests.append(("L3.20 Mode B batch: 10 entities", t20))

    # 21. Mode A batch: 10 entities via h5py
    def t21():
        keys = list(vdp.keys()[:10])
        total = 0
        for k in keys:
            meta = vdp[k].metadata
            file_rel = meta["path_mh_powder_30T"]
            dataset = meta["dataset_mh_powder_30T"]
            full_path = f"{base_dirs['VDP']}/{file_rel}"
            with h5py.File(full_path, "r") as f:
                arr = f[dataset][()]
            total += arr.shape[0]
        return total > 0, len(keys)
    tests.append(("L3.21 Mode A batch: 10 entities h5py", t21))

    return tests


# ---------------------------------------------------------------------------
# Level 4: Cross-Dataset Queries
# ---------------------------------------------------------------------------

def level4_tests(client):
    tests = []

    # 22. Root search only sees dataset containers, NOT entities.
    #     Entity-level fields like Ja_meV are invisible at root.
    #     This is expected — search is scoped to direct children.
    def t22():
        r = client.search(Key("Ja_meV") > 0.5)
        n = len(r)
        return n == 0, n  # Expected: 0 (entity fields not at root)
    tests.append(("L4.22 Root: entity field invisible (expected 0)", t22))

    # 23. Cross-dataset entity search requires iterating datasets.
    #     This tests the recommended pattern.
    def t23():
        total = 0
        for dk in client.keys():
            ds = client[dk]
            r = ds.search(Key("Ja_meV") > 0.5)
            total += len(r)
        return total > 0, total
    tests.append(("L4.23 Cross-dataset: iterate + search", t23))

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Connecting to {TILED_URL}...")
    client = from_uri(TILED_URL, api_key=API_KEY)
    print(f"Connected — {len(client)} datasets: {list(client.keys())}\n")

    base_dirs = load_base_dirs()

    all_tests = []
    all_tests.extend(level1_tests(client))
    all_tests.extend(level2_tests(client))
    all_tests.extend(level3_tests(client, base_dirs))
    all_tests.extend(level4_tests(client))

    results = []
    for name, fn in all_tests:
        result = run_test(name, fn)
        results.append(result)
        status = "PASS" if result[1] else "FAIL"
        print(f"  {status}  {name}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'#':<6} {'Test':<45} {'Result':<6} {'Count':>8} {'Time (ms)':>10}")
    print("-" * 80)

    passed = 0
    failed = 0
    for i, (name, ok, count, ms, err) in enumerate(results, 1):
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{i:<6} {name:<45} {status:<6} {count:>8} {ms:>10.1f}")
        if err:
            print(f"       ERROR: {err}")

    print("-" * 80)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
