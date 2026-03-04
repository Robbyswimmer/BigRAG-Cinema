#!/usr/bin/env python3
"""Auto-generate benchmark YAML configs for all available parquet datasets.

Scans data/parquet/ for dataset directories, determines row count tier,
and generates an appropriate benchmark config in conf/.

Usage:
    python scripts/cluster/generate_bench_configs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
CONF_DIR = PROJECT_ROOT / "conf"


# Tier thresholds based on estimated row counts from file sizes
# (parquet with embeddings is ~1.5 KB/row)
def estimate_rows(parquet_path: Path) -> int:
    """Estimate row count from total parquet file size."""
    if parquet_path.is_dir():
        total_bytes = sum(f.stat().st_size for f in parquet_path.glob("*.parquet"))
    elif parquet_path.is_file():
        total_bytes = parquet_path.stat().st_size
    else:
        return 0
    # ~1.5 KB per row with 384-dim float32 embeddings
    return int(total_bytes / 1500)


def tier_config(est_rows: int) -> dict:
    """Return num_repetitions and num_queries based on dataset size."""
    if est_rows < 500_000:
        return {"num_repetitions": 3, "num_queries": 50, "tier": "small"}
    elif est_rows < 5_000_000:
        return {"num_repetitions": 2, "num_queries": 50, "tier": "medium"}
    elif est_rows < 20_000_000:
        return {"num_repetitions": 1, "num_queries": 30, "tier": "large"}
    else:
        return {"num_repetitions": 1, "num_queries": 20, "tier": "xlarge"}


def generate_config(dataset_name: str, parquet_path: Path) -> str:
    """Generate YAML config string for a dataset."""
    est_rows = estimate_rows(parquet_path)
    tier = tier_config(est_rows)
    ds_lower = dataset_name.lower()

    return f"""# Auto-generated benchmark config -- {dataset_name} (~{est_rows:,} rows, {tier['tier']})
dataset:
  source_path: "data/parquet/{dataset_name}.parquet"
  fractions: [0.10, 0.25, 0.50, 0.75, 1.0]

embedding:
  model_name: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 512
  normalize: true

queries:
  num_queries: {tier['num_queries']}
  seed: 42

strategies:
  - name: "filter_first"
  - name: "vector_first"
  - name: "hybrid_parallel"
  - name: "adaptive"

output:
  results_dir: "results/raw_metrics/{ds_lower}"

misc:
  num_repetitions: {tier['num_repetitions']}
  warm_up_runs: 2
  log_level: "INFO"
"""


def main():
    if not PARQUET_DIR.exists():
        print(f"ERROR: Parquet directory not found: {PARQUET_DIR}")
        sys.exit(1)

    # Find all parquet datasets (directories or files)
    datasets = []
    for p in sorted(PARQUET_DIR.iterdir()):
        name = p.stem.replace(".parquet", "") if p.is_file() else p.name.replace(".parquet", "")
        if p.name.startswith("."):
            continue
        datasets.append((name, p))

    if not datasets:
        print("No parquet datasets found.")
        sys.exit(0)

    created = 0
    skipped = 0
    for name, parquet_path in datasets:
        config_path = CONF_DIR / f"bench_{name.lower()}.yaml"
        if config_path.exists():
            est = estimate_rows(parquet_path)
            print(f"  {name}: config exists ({config_path.name}), skipping")
            skipped += 1
            continue

        config_content = generate_config(name, parquet_path)
        config_path.write_text(config_content)
        est = estimate_rows(parquet_path)
        tier = tier_config(est)
        print(f"  {name}: created {config_path.name} "
              f"(~{est:,} rows, {tier['tier']}, "
              f"{tier['num_queries']}q x {tier['num_repetitions']}r)")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped, {created + skipped} total")


if __name__ == "__main__":
    main()
