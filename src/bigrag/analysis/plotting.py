"""
bigrag.analysis.plotting -- Latency CDFs, throughput bars, scaling curves.

Generates matplotlib / seaborn figures suitable for inclusion in the
project report and presentation slides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from bigrag.analysis.report_tables import generate_latex_table, generate_markdown_table
from bigrag.analysis.statistics import confidence_interval


def _extract_strategy_records(metrics: dict) -> dict[str, dict]:
    if "strategies" not in metrics:
        return {}

    out: dict[str, dict] = {}
    for strategy, payload in metrics.get("strategies", {}).items():
        frac_payload = payload.get("fractions", {})
        all_latencies: list[float] = []
        all_throughputs: list[float] = []
        scaling: list[tuple[float, float]] = []
        for fraction_key, fraction_data in frac_payload.items():
            records = fraction_data.get("records", [])
            latencies = [
                float(r["latency_ms"])
                for r in records
                if isinstance(r.get("latency_ms"), (int, float))
            ]
            throughputs = [
                float(r["throughput_qps"])
                for r in records
                if isinstance(r.get("throughput_qps"), (int, float))
            ]
            all_latencies.extend(latencies)
            all_throughputs.extend(throughputs)
            if latencies:
                scaling.append((float(fraction_key), float(np.median(latencies))))

        scaling.sort(key=lambda x: x[0])
        out[strategy] = {
            "latencies": all_latencies,
            "throughputs": all_throughputs,
            "scaling": scaling,
        }
    return out


def plot_latency_cdf(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot cumulative distribution functions of per-query latency.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by strategy name.
    output_path : Path | None
        If provided, save the figure to this path.
    """
    extracted = _extract_strategy_records(metrics)
    plt.figure(figsize=(9, 5))
    for strategy, data in extracted.items():
        vals = np.asarray(data["latencies"], dtype=float)
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        y = np.arange(1, vals.size + 1) / float(vals.size)
        plt.plot(vals, y, label=strategy, linewidth=2)

    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Per-Query Latency CDF by Strategy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def plot_throughput_bars(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot a grouped bar chart of throughput (queries/sec) per strategy.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by strategy name.
    output_path : Path | None
        If provided, save the figure to this path.
    """
    extracted = _extract_strategy_records(metrics)
    strategies: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    for strategy, data in extracted.items():
        vals = np.asarray(data["throughputs"], dtype=float)
        if vals.size == 0:
            continue
        strategies.append(strategy)
        means.append(float(np.mean(vals)))
        low, high = confidence_interval(vals.tolist(), confidence=0.95)
        errors.append(max(0.0, float(high - np.mean(vals))))

    plt.figure(figsize=(9, 5))
    x = np.arange(len(strategies))
    plt.bar(x, means, yerr=errors, capsize=4)
    plt.xticks(x, strategies, rotation=20, ha="right")
    plt.ylabel("Throughput (qps)")
    plt.title("Mean Throughput by Strategy (95% CI)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def plot_scaling_curves(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Plot latency vs. data fraction scaling curves per strategy.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics keyed by (strategy, fraction).
    output_path : Path | None
        If provided, save the figure to this path.
    """
    extracted = _extract_strategy_records(metrics)
    plt.figure(figsize=(9, 5))
    for strategy, data in extracted.items():
        points = data["scaling"]
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, marker="o", linewidth=2, label=strategy)

    plt.xlabel("Dataset Fraction")
    plt.ylabel("Median Latency (ms)")
    plt.title("Latency Scaling by Dataset Fraction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def generate_plots(metrics_dir: str | Path, output_dir: str | Path, fmt: str = "png") -> None:
    """Load benchmark results and generate figures/tables for reporting."""
    metrics_path = Path(metrics_dir).expanduser().resolve() / "benchmark_results.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing benchmark results file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_cdf(metrics, out_dir / f"latency_cdf.{fmt}")
    plot_throughput_bars(metrics, out_dir / f"throughput_bars.{fmt}")
    plot_scaling_curves(metrics, out_dir / f"scaling_curves.{fmt}")

    tables_dir = out_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "benchmark_summary.md").write_text(
        generate_markdown_table(metrics),
        encoding="utf-8",
    )
    (tables_dir / "benchmark_summary.tex").write_text(
        generate_latex_table(metrics),
        encoding="utf-8",
    )
