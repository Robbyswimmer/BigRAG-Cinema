"""
bigrag.analysis.report_tables -- Generate LaTeX and Markdown tables from metrics.

Transforms benchmark summary dicts into formatted table strings
that can be copy-pasted directly into a LaTeX document or Markdown
README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STRATEGY_DISPLAY = {
    "filter_first": "Filter-First",
    "vector_first": "Vector-First",
    "hybrid_parallel": "Hybrid Parallel",
    "adaptive": "Adaptive",
}

DATASET_DISPLAY = {
    "all_beauty": "All Beauty",
    "amazon_fashion": "Amazon Fashion",
    "appliances": "Appliances",
    "arts_crafts_and_sewing": "Arts \\& Crafts",
    "baby_products": "Baby Products",
}


def _fmt(val: float, decimals: int = 1) -> str:
    """Format a number with comma separators."""
    if val >= 1000:
        return f"{val:,.{decimals}f}"
    return f"{val:.{decimals}f}"


def _fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _strat(name: str) -> str:
    return STRATEGY_DISPLAY.get(name, name)


def _ds(name: str) -> str:
    return DATASET_DISPLAY.get(name, name.replace("_", " ").title())


def _bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def load_all_results(metrics_dir: str | Path) -> dict[str, dict]:
    """Load all benchmark_results.json files from a metrics directory."""
    metrics_path = Path(metrics_dir)
    results = {}
    for p in sorted(metrics_path.iterdir()):
        json_file = p / "benchmark_results.json"
        if p.is_dir() and json_file.exists():
            with open(json_file) as f:
                results[p.name] = json.load(f)
    return results


def _get_metric(summary: dict, strategy: str, fraction: str, metric: str, stat: str) -> float:
    try:
        return float(
            summary["strategies"][strategy]["fractions"][fraction]["summary"]["metrics"][metric][stat]
        )
    except (KeyError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Table 1: Cross-Dataset Strategy Comparison (full-scale performance)
# ---------------------------------------------------------------------------

def table_cross_dataset_comparison(all_results: dict[str, dict]) -> str:
    """Strategy performance at full scale across all datasets."""
    datasets = list(all_results.keys())
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]

    ncols = len(datasets) + 1
    col_spec = "l" + "r" * len(datasets)

    lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    # Header
    header = "Strategy"
    for ds in datasets:
        total_rows = all_results[ds].get("config", {}).get("total_rows", 0)
        header += f" & {_ds(ds)} ({_fmt_rows(total_rows)})"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    # Subheader: Median Latency (ms)
    lines.append("\\multicolumn{" + str(ncols) + "}{l}{\\textit{Median Latency (ms)}} \\\\")
    lines.append("\\hline")

    for strat in strategies:
        vals = []
        for ds in datasets:
            v = _get_metric(all_results[ds], strat, "1.0000", "latency_ms", "median")
            vals.append(v)
        best_idx = vals.index(min(vals)) if vals else -1
        row = _strat(strat)
        for i, v in enumerate(vals):
            cell = _fmt(v, 0)
            if i == best_idx:
                cell = _bold(cell)
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")

    # Subheader: P99 Latency (ms)
    lines.append("\\multicolumn{" + str(ncols) + "}{l}{\\textit{P99 Latency (ms)}} \\\\")
    lines.append("\\hline")

    for strat in strategies:
        vals = []
        for ds in datasets:
            v = _get_metric(all_results[ds], strat, "1.0000", "latency_ms", "p99")
            vals.append(v)
        best_idx = vals.index(min(vals)) if vals else -1
        row = _strat(strat)
        for i, v in enumerate(vals):
            cell = _fmt(v, 0)
            if i == best_idx:
                cell = _bold(cell)
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")

    # Subheader: Mean Throughput (qps)
    lines.append("\\multicolumn{" + str(ncols) + "}{l}{\\textit{Mean Throughput (qps)}} \\\\")
    lines.append("\\hline")

    for strat in strategies:
        vals = []
        for ds in datasets:
            v = _get_metric(all_results[ds], strat, "1.0000", "throughput_qps", "mean")
            vals.append(v)
        best_idx = vals.index(max(vals)) if vals else -1
        row = _strat(strat)
        for i, v in enumerate(vals):
            cell = _fmt(v, 2)
            if i == best_idx:
                cell = _bold(cell)
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Cross-dataset strategy comparison at full data scale. "
        "Bold indicates best strategy per dataset per metric.}",
        "\\label{tab:cross-dataset}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Scalability Analysis (latency growth from 10% to 100%)
# ---------------------------------------------------------------------------

def table_scalability(all_results: dict[str, dict]) -> str:
    """Scalability factor: how much latency grows from 10% to 100% data."""
    datasets = list(all_results.keys())
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]

    col_spec = "l" + "rr" * len(datasets)
    lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    # Header row 1: dataset names spanning 2 cols each
    header1 = "Strategy"
    for ds in datasets:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{_ds(ds)}}}"
    header1 += " \\\\"
    lines.append(header1)

    # Header row 2: sub-columns
    header2 = ""
    for _ in datasets:
        header2 += " & 10\\% (ms) & Factor"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\hline")

    for strat in strategies:
        row = _strat(strat)
        for ds in datasets:
            lat_10 = _get_metric(all_results[ds], strat, "0.1000", "latency_ms", "median")
            lat_100 = _get_metric(all_results[ds], strat, "1.0000", "latency_ms", "median")
            factor = lat_100 / lat_10 if lat_10 > 0 else 0.0
            row += f" & {_fmt(lat_10, 0)} & {factor:.1f}$\\times$"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Scalability analysis: median latency at 10\\% data fraction and "
        "growth factor to full scale (100\\%). Lower factor = better scalability.}",
        "\\label{tab:scalability}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Latency Percentile Breakdown per strategy (single dataset)
# ---------------------------------------------------------------------------

def table_latency_percentiles(summary: dict, dataset_name: str = "") -> str:
    """Full latency percentile breakdown: p50, p95, p99, min, max at each fraction."""
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]
    fractions = ["0.1000", "0.2500", "0.5000", "0.7500", "1.0000"]
    frac_labels = ["10\\%", "25\\%", "50\\%", "75\\%", "100\\%"]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Strategy & Fraction & P50 & P95 & P99 & Min & Max \\\\",
        "\\hline",
    ]

    for strat in strategies:
        for i, (frac, flabel) in enumerate(zip(fractions, frac_labels)):
            p50 = _get_metric(summary, strat, frac, "latency_ms", "median")
            p95 = _get_metric(summary, strat, frac, "latency_ms", "p95")
            p99 = _get_metric(summary, strat, frac, "latency_ms", "p99")
            mn = _get_metric(summary, strat, frac, "latency_ms", "min")
            mx = _get_metric(summary, strat, frac, "latency_ms", "max")
            label = _strat(strat) if i == 0 else ""
            lines.append(
                f"{label} & {flabel} & {_fmt(p50,0)} & {_fmt(p95,0)} & "
                f"{_fmt(p99,0)} & {_fmt(mn,0)} & {_fmt(mx,0)} \\\\"
            )
        lines.append("\\hline")

    ds_label = _ds(dataset_name) if dataset_name else "Selected Dataset"
    lines.extend([
        "\\end{tabular}",
        f"\\caption{{Latency percentile breakdown (ms) -- {ds_label}.}}",
        f"\\label{{tab:percentiles-{dataset_name or 'dataset'}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: Strategy Winner Summary
# ---------------------------------------------------------------------------

def table_strategy_winners(all_results: dict[str, dict]) -> str:
    """Which strategy wins per dataset and per metric."""
    datasets = list(all_results.keys())
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]
    metrics_spec = [
        ("Median Latency", "latency_ms", "median", "min"),
        ("P99 Latency", "latency_ms", "p99", "min"),
        ("Mean Throughput", "throughput_qps", "mean", "max"),
        ("Latency Variance", "latency_ms", None, "min"),  # special: p99/p50 ratio
    ]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "l" * len(datasets) + "}",
        "\\hline",
    ]

    header = "Metric"
    for ds in datasets:
        header += f" & {_ds(ds)}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for label, metric, stat, best_fn in metrics_spec:
        row = label
        for ds in datasets:
            best_val = None
            best_strat = ""
            for strat in strategies:
                if stat is None:
                    # Variance proxy: p99/p50 ratio
                    p50 = _get_metric(all_results[ds], strat, "1.0000", metric, "median")
                    p99 = _get_metric(all_results[ds], strat, "1.0000", metric, "p99")
                    v = p99 / p50 if p50 > 0 else float("inf")
                else:
                    v = _get_metric(all_results[ds], strat, "1.0000", metric, stat)
                if best_val is None:
                    best_val = v
                    best_strat = strat
                elif best_fn == "min" and v < best_val:
                    best_val = v
                    best_strat = strat
                elif best_fn == "max" and v > best_val:
                    best_val = v
                    best_strat = strat
            row += f" & {_strat(best_strat)}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")

    # Win counts
    row = "\\textbf{Total Wins}"
    for ds in datasets:
        wins: dict[str, int] = {}
        for label, metric, stat, best_fn in metrics_spec:
            best_val = None
            best_strat = ""
            for strat in strategies:
                if stat is None:
                    p50 = _get_metric(all_results[ds], strat, "1.0000", metric, "median")
                    p99 = _get_metric(all_results[ds], strat, "1.0000", metric, "p99")
                    v = p99 / p50 if p50 > 0 else float("inf")
                else:
                    v = _get_metric(all_results[ds], strat, "1.0000", metric, stat)
                if best_val is None or (best_fn == "min" and v < best_val) or (best_fn == "max" and v > best_val):
                    best_val = v
                    best_strat = strat
            wins[best_strat] = wins.get(best_strat, 0) + 1
        # Find the strategy with most wins
        top = max(wins, key=wins.get) if wins else ""
        row += f" & {_strat(top)} ({wins.get(top, 0)}/4)"
    row += " \\\\"
    lines.append(row)

    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Best-performing strategy per dataset at full scale. "
        "Latency Variance measured as P99/P50 ratio (lower = more consistent).}",
        "\\label{tab:winners}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: Speedup of Filter-First over other strategies
# ---------------------------------------------------------------------------

def table_speedup(all_results: dict[str, dict]) -> str:
    """Speedup of filter_first compared to other strategies."""
    datasets = list(all_results.keys())
    strategies = ["vector_first", "hybrid_parallel", "adaptive"]

    col_spec = "l" + "r" * len(datasets)
    lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    header = "Comparison"
    for ds in datasets:
        header += f" & {_ds(ds)}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for strat in strategies:
        row = f"vs. {_strat(strat)}"
        for ds in datasets:
            ff_lat = _get_metric(all_results[ds], "filter_first", "1.0000", "latency_ms", "median")
            other_lat = _get_metric(all_results[ds], strat, "1.0000", "latency_ms", "median")
            speedup = other_lat / ff_lat if ff_lat > 0 else 0.0
            row += f" & {speedup:.1f}$\\times$"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Speedup of Filter-First over other strategies (median latency ratio at full scale). "
        "Higher = Filter-First is faster by that factor.}",
        "\\label{tab:speedup}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 6: Dataset Summary
# ---------------------------------------------------------------------------

def table_dataset_summary(all_results: dict[str, dict]) -> str:
    """Summary of datasets used in the benchmark."""
    datasets = list(all_results.keys())

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Dataset & Total Rows & Queries & Repetitions \\\\",
        "\\hline",
    ]

    total_rows_all = 0
    for ds in datasets:
        cfg = all_results[ds].get("config", {})
        total_rows = cfg.get("total_rows", 0)
        num_queries = cfg.get("num_queries", 0)
        num_reps = cfg.get("num_repetitions", 0)
        total_rows_all += total_rows
        lines.append(
            f"{_ds(ds)} & {total_rows:,} & {num_queries} & {num_reps} \\\\"
        )

    lines.append("\\hline")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total_rows_all:,}}} & & \\\\")
    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Benchmark dataset summary from the Amazon Reviews 2023 corpus.}",
        "\\label{tab:datasets}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 7: Per-Dataset Detailed Performance (one table per dataset)
# ---------------------------------------------------------------------------

def table_per_dataset_detail(summary: dict, dataset_name: str = "") -> str:
    """Detailed per-fraction performance for a single dataset."""
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]
    fractions = ["0.1000", "0.2500", "0.5000", "0.7500", "1.0000"]
    frac_labels = ["10\\%", "25\\%", "50\\%", "75\\%", "100\\%"]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Strategy & Fraction & Median Lat. (ms) & P95 Lat. (ms) & Throughput (qps) & Results \\\\",
        "\\hline",
    ]

    for strat in strategies:
        for i, (frac, flabel) in enumerate(zip(fractions, frac_labels)):
            p50 = _get_metric(summary, strat, frac, "latency_ms", "median")
            p95 = _get_metric(summary, strat, frac, "latency_ms", "p95")
            tp = _get_metric(summary, strat, frac, "throughput_qps", "mean")
            rc = _get_metric(summary, strat, frac, "result_count", "mean")
            label = _strat(strat) if i == 0 else ""
            lines.append(
                f"{label} & {flabel} & {_fmt(p50,0)} & {_fmt(p95,0)} & "
                f"{tp:.2f} & {rc:.1f} \\\\"
            )
        lines.append("\\hline")

    ds_label = _ds(dataset_name) if dataset_name else "Dataset"
    lines.extend([
        "\\end{tabular}",
        f"\\caption{{Detailed benchmark results -- {ds_label}.}}",
        f"\\label{{tab:detail-{dataset_name or 'dataset'}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 8: Recall@K per strategy per selectivity bucket
# ---------------------------------------------------------------------------

def table_recall_at_k(recall_results: list[dict]) -> str:
    """Recall@K comparison across strategies and selectivity buckets."""
    if not recall_results:
        return ""

    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]
    buckets = ["low", "medium", "high"]

    lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(recall_results) + 1) + "}",
        "\\hline",
    ]

    # Header
    header = "Strategy"
    for result in recall_results:
        ds_name = result.get("dataset", "")
        header += f" & {_ds(ds_name)}"
    header += " & Mean \\\\"
    lines.append(header)
    lines.append("\\hline")

    for strat in strategies:
        row = _strat(strat)
        strat_vals = []
        for result in recall_results:
            val = result.get("strategies", {}).get(strat, {}).get("mean_recall", 0.0)
            strat_vals.append(val)
            cell = f"{val:.4f}"
            if val >= 1.0:
                cell = _bold(cell)
            row += f" & {cell}"
        mean_val = sum(strat_vals) / len(strat_vals) if strat_vals else 0.0
        mean_cell = f"{mean_val:.4f}"
        if mean_val >= 1.0:
            mean_cell = _bold(mean_cell)
        row += f" & {mean_cell} \\\\"
        lines.append(row)

    lines.append("\\hline")

    # Selectivity breakdown (use first dataset as representative)
    if recall_results:
        lines.append(
            "\\multicolumn{" + str(len(recall_results) + 2)
            + "}{l}{\\textit{By Selectivity Bucket (first dataset)}} \\\\"
        )
        lines.append("\\hline")
        first = recall_results[0]
        for strat in strategies:
            strat_data = first.get("strategies", {}).get(strat, {})
            by_sel = strat_data.get("by_selectivity", {})
            bucket_vals = [str(by_sel.get(b, "---")) for b in buckets]
            row = f"{_strat(strat)} & Low: {bucket_vals[0]} & Med: {bucket_vals[1]} & High: {bucket_vals[2]}"
            # Pad remaining columns
            for _ in range(len(recall_results) - 2):
                row += " &"
            row += " \\\\"
            lines.append(row)
        lines.append("\\hline")

    top_k = recall_results[0].get("top_k", 10) if recall_results else 10
    lines.extend([
        "\\end{tabular}",
        f"\\caption{{Recall@{top_k} per strategy. Filter-First serves as ground truth (recall=1.0). "
        "Bold indicates perfect recall.}",
        f"\\label{{tab:recall-at-{top_k}}}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 9: Shuffle Cost Comparison
# ---------------------------------------------------------------------------

def table_shuffle_cost(all_results: dict[str, dict]) -> str | None:
    """Shuffle cost comparison table. Returns None if all shuffle values are 0."""
    datasets = list(all_results.keys())
    strategies = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]

    # Check if any non-zero shuffle data exists
    has_shuffle_data = False
    for ds in datasets:
        for strat in strategies:
            read_bytes = _get_metric(all_results[ds], strat, "1.0000", "shuffle_read_bytes", "mean")
            write_bytes = _get_metric(all_results[ds], strat, "1.0000", "shuffle_write_bytes", "mean")
            if read_bytes > 0 or write_bytes > 0:
                has_shuffle_data = True
                break
        if has_shuffle_data:
            break

    if not has_shuffle_data:
        return None

    col_spec = "l" + "rr" * len(datasets)
    lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    # Header
    header1 = "Strategy"
    for ds in datasets:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{_ds(ds)}}}"
    header1 += " \\\\"
    lines.append(header1)

    header2 = ""
    for _ in datasets:
        header2 += " & Read (MB) & Write (MB)"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\hline")

    for strat in strategies:
        row = _strat(strat)
        for ds in datasets:
            read_mb = _get_metric(all_results[ds], strat, "1.0000", "shuffle_read_bytes", "mean") / 1e6
            write_mb = _get_metric(all_results[ds], strat, "1.0000", "shuffle_write_bytes", "mean") / 1e6
            row += f" & {read_mb:.1f} & {write_mb:.1f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "\\caption{Mean shuffle cost per query at full scale (MB). "
        "Lower shuffle cost indicates less data movement between partitions.}",
        "\\label{tab:shuffle-cost}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Original functions (preserved for backwards compatibility)
# ---------------------------------------------------------------------------

def _extract_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    strategies = summary.get("strategies", {})
    if not isinstance(strategies, dict):
        return rows

    for strategy, payload in strategies.items():
        fractions = payload.get("fractions", {})
        for fraction_key, fraction_payload in fractions.items():
            metrics = fraction_payload.get("summary", {}).get("metrics", {})
            latency_mean = metrics.get("latency_ms", {}).get("mean", 0.0)
            latency_p95 = metrics.get("latency_ms", {}).get("p95", 0.0)
            throughput_mean = metrics.get("throughput_qps", {}).get("mean", 0.0)
            result_mean = metrics.get("result_count", {}).get("mean", 0.0)
            rows.append(
                {
                    "strategy": str(strategy),
                    "fraction": str(fraction_key),
                    "latency_mean_ms": float(latency_mean),
                    "latency_p95_ms": float(latency_p95),
                    "throughput_mean_qps": float(throughput_mean),
                    "result_count_mean": float(result_mean),
                }
            )
    rows.sort(key=lambda r: (r["strategy"], r["fraction"]))
    return rows


def generate_latex_table(summary: dict, caption: str = "Benchmark Results") -> str:
    rows = _extract_rows(summary)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Strategy & Fraction & Mean Latency (ms) & P95 Latency (ms) & Throughput (qps) & Mean Results \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['fraction']} & "
            f"{row['latency_mean_ms']:.3f} & {row['latency_p95_ms']:.3f} & "
            f"{row['throughput_mean_qps']:.3f} & {row['result_count_mean']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_markdown_table(summary: dict) -> str:
    rows = _extract_rows(summary)
    lines = [
        "| Strategy | Fraction | Mean Latency (ms) | P95 Latency (ms) | Throughput (qps) | Mean Results |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['fraction']} | "
            f"{row['latency_mean_ms']:.3f} | {row['latency_p95_ms']:.3f} | "
            f"{row['throughput_mean_qps']:.3f} | {row['result_count_mean']:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main: generate all tables and write to files
# ---------------------------------------------------------------------------

def generate_all_tables(
    metrics_dir: str | Path,
    output_dir: str | Path,
    recall_results_path: str | Path | None = None,
) -> None:
    """Generate all LaTeX tables and save to output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(metrics_dir)
    if not all_results:
        print("No benchmark results found.")
        return

    print(f"Loaded results for {len(all_results)} datasets: {', '.join(all_results.keys())}")

    # Cross-dataset tables
    tables = {
        "table_dataset_summary.tex": table_dataset_summary(all_results),
        "table_cross_dataset.tex": table_cross_dataset_comparison(all_results),
        "table_scalability.tex": table_scalability(all_results),
        "table_winners.tex": table_strategy_winners(all_results),
        "table_speedup.tex": table_speedup(all_results),
    }

    # Per-dataset tables (pick the largest dataset for the detailed percentile table)
    largest_ds = max(all_results.keys(), key=lambda k: all_results[k].get("config", {}).get("total_rows", 0))
    tables[f"table_percentiles_{largest_ds}.tex"] = table_latency_percentiles(
        all_results[largest_ds], largest_ds
    )
    tables[f"table_detail_{largest_ds}.tex"] = table_per_dataset_detail(
        all_results[largest_ds], largest_ds
    )

    # Recall@K table (if recall results exist)
    if recall_results_path is not None:
        recall_path = Path(recall_results_path)
    else:
        recall_path = Path(metrics_dir).parent / "recall_results.json"
    if recall_path.exists():
        with open(recall_path) as f:
            recall_data = json.load(f)
        recall_tex = table_recall_at_k(recall_data)
        if recall_tex:
            tables["table_recall.tex"] = recall_tex
            print(f"  Including recall table from {recall_path}")
    else:
        print(f"  No recall results found at {recall_path} — skipping recall table")

    # Shuffle cost table (only if non-zero data exists)
    shuffle_tex = table_shuffle_cost(all_results)
    if shuffle_tex:
        tables["table_shuffle_cost.tex"] = shuffle_tex
    else:
        print("  All shuffle values are 0 — skipping shuffle table")

    for filename, content in tables.items():
        path = output_path / filename
        path.write_text(content)
        print(f"  wrote {path}")

    print(f"\nGenerated {len(tables)} tables in {output_path}")
