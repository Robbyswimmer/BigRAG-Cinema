"""
bigrag.analysis.plotting -- Publication-quality benchmark visualizations.

Generates matplotlib / seaborn figures suitable for inclusion in the
project report and presentation slides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from bigrag.analysis.report_tables import generate_latex_table, generate_markdown_table
from bigrag.analysis.statistics import confidence_interval

# ── Seaborn / matplotlib style ──────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
STRATEGY_COLORS = {
    "filter_first": "#2196F3",
    "vector_first": "#FF9800",
    "hybrid_parallel": "#4CAF50",
    "adaptive": "#9C27B0",
}
STRATEGY_ORDER = ["filter_first", "vector_first", "hybrid_parallel", "adaptive"]
DPI = 180


def _strategy_color(name: str) -> str:
    return STRATEGY_COLORS.get(name, "#666666")


def _sorted_strategies(keys):
    return sorted(keys, key=lambda k: STRATEGY_ORDER.index(k) if k in STRATEGY_ORDER else 99)


# ── Data extraction helpers ─────────────────────────────────────────────

def _extract_strategy_records(metrics: dict) -> dict[str, dict]:
    if "strategies" not in metrics:
        return {}

    out: dict[str, dict] = {}
    for strategy, payload in metrics.get("strategies", {}).items():
        frac_payload = payload.get("fractions", {})
        all_latencies: list[float] = []
        all_throughputs: list[float] = []
        scaling: list[tuple[float, float]] = []
        per_fraction: dict[float, list[float]] = {}
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
            frac_val = float(fraction_key)
            per_fraction[frac_val] = latencies
            all_latencies.extend(latencies)
            all_throughputs.extend(throughputs)
            if latencies:
                scaling.append((frac_val, float(np.median(latencies))))

        scaling.sort(key=lambda x: x[0])
        out[strategy] = {
            "latencies": all_latencies,
            "throughputs": all_throughputs,
            "scaling": scaling,
            "per_fraction": per_fraction,
        }
    return out


def _get_total_rows(metrics: dict) -> int:
    return int(metrics.get("config", {}).get("total_rows", 0))


# ── Figure 1: Latency CDF ──────────────────────────────────────────────

def plot_latency_cdf(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in _sorted_strategies(extracted):
        data = extracted[strategy]
        vals = np.asarray(data["latencies"], dtype=float)
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        y = np.arange(1, vals.size + 1) / float(vals.size)
        ax.plot(vals, y, label=strategy.replace("_", " ").title(),
                linewidth=2.5, color=_strategy_color(strategy))

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Per-Query Latency CDF by Strategy", fontweight="bold")
    ax.axhline(y=0.50, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.95, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(ax.get_xlim()[0], 0.51, " p50", fontsize=9, color="grey")
    ax.text(ax.get_xlim()[0], 0.96, " p95", fontsize=9, color="grey")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Throughput bar chart ──────────────────────────────────────

def plot_throughput_bars(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    strategies: list[str] = []
    means: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []

    for strategy in _sorted_strategies(extracted):
        data = extracted[strategy]
        vals = np.asarray(data["throughputs"], dtype=float)
        if vals.size == 0:
            continue
        strategies.append(strategy)
        m = float(np.mean(vals))
        means.append(m)
        low, high = confidence_interval(vals.tolist(), confidence=0.95)
        ci_lo.append(m - low)
        ci_hi.append(high - m)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))
    colors = [_strategy_color(s) for s in strategies]
    bars = ax.bar(x, means, yerr=[ci_lo, ci_hi], capsize=5,
                  color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_hi) * 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=15, ha="right")
    ax.set_ylabel("Throughput (queries/sec)")
    ax.set_title("Mean Throughput by Strategy (95% CI)", fontweight="bold")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: Scaling curves ───────────────────────────────────────────

def plot_scaling_curves(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    total_rows = _get_total_rows(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in _sorted_strategies(extracted):
        data = extracted[strategy]
        points = data["scaling"]
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=8,
                label=strategy.replace("_", " ").title(),
                color=_strategy_color(strategy))

    ax.set_xlabel("Dataset Fraction")
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title("Latency Scaling by Dataset Size", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if total_rows:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
        ax2.set_xticks(fracs)
        ax2.set_xticklabels([f"{int(total_rows * f):,}" for f in fracs], fontsize=9)
        ax2.set_xlabel("Number of Rows", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: Latency box plots per fraction ───────────────────────────

def plot_latency_boxplots(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    rows = []
    for strategy in _sorted_strategies(extracted):
        data = extracted[strategy]
        for frac_val, latencies in sorted(data["per_fraction"].items()):
            for lat in latencies:
                rows.append({
                    "Strategy": strategy.replace("_", " ").title(),
                    "Fraction": f"{frac_val:.0%}",
                    "frac_sort": frac_val,
                    "Latency (ms)": lat,
                })

    if not rows:
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values("frac_sort")

    fractions = df["Fraction"].unique()
    n_frac = len(fractions)
    fig, axes = plt.subplots(1, n_frac, figsize=(4 * n_frac, 6), sharey=True)
    if n_frac == 1:
        axes = [axes]

    palette = {s.replace("_", " ").title(): _strategy_color(s) for s in STRATEGY_ORDER}

    for ax, frac_label in zip(axes, fractions):
        subset = df[df["Fraction"] == frac_label]
        sns.boxplot(data=subset, x="Strategy", y="Latency (ms)",
                    hue="Strategy", palette=palette, ax=ax,
                    width=0.6, fliersize=3, legend=False)
        ax.set_title(f"{frac_label}", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.suptitle("Latency Distribution by Strategy & Dataset Fraction",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 5: Latency heatmap ──────────────────────────────────────────

def plot_latency_heatmap(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    strat_names = _sorted_strategies(extracted)
    all_fracs = sorted({
        f for s in extracted.values() for f in s["per_fraction"]
    })

    matrix = np.full((len(strat_names), len(all_fracs)), np.nan)
    for i, strategy in enumerate(strat_names):
        for j, frac in enumerate(all_fracs):
            lats = extracted[strategy]["per_fraction"].get(frac, [])
            if lats:
                matrix[i, j] = float(np.median(lats))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=[f"{f:.0%}" for f in all_fracs],
        yticklabels=[s.replace("_", " ").title() for s in strat_names],
        ax=ax, linewidths=1, linecolor="white",
        cbar_kws={"label": "Median Latency (ms)"},
    )
    ax.set_xlabel("Dataset Fraction")
    ax.set_ylabel("")
    ax.set_title("Median Latency Heatmap (ms)", fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 6: Percentile breakdown (grouped bars) ──────────────────────

def plot_percentile_bars(
    metrics: dict,
    output_path: Optional[Path] = None,
) -> None:
    extracted = _extract_strategy_records(metrics)
    strat_names = _sorted_strategies(extracted)
    percentiles = [50, 95, 99]
    pct_labels = ["p50", "p95", "p99"]

    data_dict: dict[str, list[float]] = {lbl: [] for lbl in pct_labels}
    labels: list[str] = []
    for strategy in strat_names:
        vals = np.asarray(extracted[strategy]["latencies"], dtype=float)
        if vals.size == 0:
            continue
        labels.append(strategy.replace("_", " ").title())
        for p, lbl in zip(percentiles, pct_labels):
            data_dict[lbl].append(float(np.percentile(vals, p)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.22
    pct_colors = ["#4CAF50", "#FF9800", "#F44336"]

    for k, (lbl, color) in enumerate(zip(pct_labels, pct_colors)):
        offset = (k - 1) * width
        bars = ax.bar(x + offset, data_dict[lbl], width, label=lbl,
                      color=color, edgecolor="white", linewidth=1)
        for bar, val in zip(bars, data_dict[lbl]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Percentiles by Strategy (All Fractions)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 7: Cross-dataset scaling comparison ─────────────────────────

def plot_cross_dataset_scaling(
    dataset_results: list[tuple[str, dict]],
    output_path: Optional[Path] = None,
) -> None:
    """Compare median latency at 100% fraction across multiple datasets.

    Parameters
    ----------
    dataset_results : list of (label, metrics_dict) tuples
        Each entry is a dataset name and its benchmark_results.json contents.
    """
    import pandas as pd

    rows = []
    for ds_label, metrics in dataset_results:
        total_rows = _get_total_rows(metrics)
        extracted = _extract_strategy_records(metrics)
        for strategy in _sorted_strategies(extracted):
            # Use the largest fraction available
            per_frac = extracted[strategy]["per_fraction"]
            if not per_frac:
                continue
            max_frac = max(per_frac.keys())
            lats = per_frac[max_frac]
            if lats:
                rows.append({
                    "Dataset": ds_label,
                    "Rows": total_rows,
                    "Strategy": strategy.replace("_", " ").title(),
                    "strategy_raw": strategy,
                    "Median Latency (ms)": float(np.median(lats)),
                    "p95 Latency (ms)": float(np.percentile(lats, 95)),
                })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("Rows")

    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy_raw in STRATEGY_ORDER:
        label = strategy_raw.replace("_", " ").title()
        subset = df[df["strategy_raw"] == strategy_raw]
        if subset.empty:
            continue
        ax.plot(subset["Rows"], subset["Median Latency (ms)"],
                marker="s", linewidth=2.5, markersize=9,
                label=label, color=_strategy_color(strategy_raw))

    ax.set_xlabel("Dataset Size (rows)")
    ax.set_ylabel("Median Latency (ms) at 100%")
    ax.set_title("Latency Scaling Across Datasets", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_throughput(
    dataset_results: list[tuple[str, dict]],
    output_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart of throughput across datasets."""
    import pandas as pd

    rows = []
    for ds_label, metrics in dataset_results:
        extracted = _extract_strategy_records(metrics)
        for strategy in _sorted_strategies(extracted):
            vals = np.asarray(extracted[strategy]["throughputs"], dtype=float)
            if vals.size == 0:
                continue
            rows.append({
                "Dataset": ds_label,
                "Strategy": strategy.replace("_", " ").title(),
                "strategy_raw": strategy,
                "Mean QPS": float(np.mean(vals)),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    datasets = df["Dataset"].unique()
    strat_labels = [s.replace("_", " ").title() for s in STRATEGY_ORDER
                    if s.replace("_", " ").title() in df["Strategy"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_ds = len(datasets)
    width = 0.8 / n_ds
    x = np.arange(len(strat_labels))
    ds_colors = ["#42A5F5", "#66BB6A", "#EF5350", "#AB47BC"]

    for i, ds in enumerate(datasets):
        subset = df[df["Dataset"] == ds]
        vals = []
        for sl in strat_labels:
            match = subset[subset["Strategy"] == sl]
            vals.append(float(match["Mean QPS"].iloc[0]) if len(match) else 0)
        offset = (i - n_ds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds,
               color=ds_colors[i % len(ds_colors)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Throughput (queries/sec)")
    ax.set_title("Throughput Comparison Across Datasets", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Master entry point ──────────────────────────────────────────────────

def generate_plots(
    metrics_dir: str | Path,
    output_dir: str | Path,
    fmt: str = "png",
    label: str | None = None,
) -> None:
    """Load benchmark results and generate all figures + tables."""
    metrics_path = Path(metrics_dir).expanduser().resolve() / "benchmark_results.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing benchmark results file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f" ({label})" if label else ""
    print(f"Generating figures{suffix} ...")
    plot_latency_cdf(metrics, out_dir / f"latency_cdf.{fmt}")
    print("  [1/6] latency_cdf")
    plot_throughput_bars(metrics, out_dir / f"throughput_bars.{fmt}")
    print("  [2/6] throughput_bars")
    plot_scaling_curves(metrics, out_dir / f"scaling_curves.{fmt}")
    print("  [3/6] scaling_curves")
    plot_latency_boxplots(metrics, out_dir / f"latency_boxplots.{fmt}")
    print("  [4/6] latency_boxplots")
    plot_latency_heatmap(metrics, out_dir / f"latency_heatmap.{fmt}")
    print("  [5/6] latency_heatmap")
    plot_percentile_bars(metrics, out_dir / f"percentile_bars.{fmt}")
    print("  [6/6] percentile_bars")

    tables_dir = out_dir.parent / "tables"
    if label:
        tables_dir = out_dir.parent / "tables" / label.lower()
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "benchmark_summary.md").write_text(
        generate_markdown_table(metrics),
        encoding="utf-8",
    )
    (tables_dir / "benchmark_summary.tex").write_text(
        generate_latex_table(metrics),
        encoding="utf-8",
    )
    print(f"  Tables saved to {tables_dir}")
    print(f"Done — 6 figures saved to {out_dir}")


def generate_cross_dataset_plots(
    results_base_dir: str | Path,
    output_dir: str | Path,
    fmt: str = "png",
) -> None:
    """Load results from multiple dataset subdirs and generate comparison plots."""
    base = Path(results_base_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_results: list[tuple[str, dict]] = []
    for subdir in sorted(base.iterdir()):
        results_file = subdir / "benchmark_results.json"
        if subdir.is_dir() and results_file.exists():
            metrics = json.loads(results_file.read_text(encoding="utf-8"))
            label = subdir.name.replace("_", " ").title()
            dataset_results.append((label, metrics))
            print(f"  Loaded {label}: {_get_total_rows(metrics):,} rows")

    if len(dataset_results) < 2:
        print("  Need at least 2 datasets for cross-dataset plots, skipping.")
        return

    print(f"Generating cross-dataset comparison plots ...")
    plot_cross_dataset_scaling(dataset_results, out_dir / f"cross_dataset_scaling.{fmt}")
    print("  [1/2] cross_dataset_scaling")
    plot_cross_dataset_throughput(dataset_results, out_dir / f"cross_dataset_throughput.{fmt}")
    print("  [2/2] cross_dataset_throughput")
    print(f"Done — 2 cross-dataset figures saved to {out_dir}")
