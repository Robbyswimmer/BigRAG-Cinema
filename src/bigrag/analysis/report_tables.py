"""
bigrag.analysis.report_tables -- Generate LaTeX and Markdown tables from metrics.

Transforms benchmark summary dicts into formatted table strings
that can be copy-pasted directly into a LaTeX document or Markdown
README.
"""

from __future__ import annotations


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
    """Render *summary* as a LaTeX ``tabular`` environment string.

    Parameters
    ----------
    summary : dict
        Aggregated benchmark metrics (strategy -> metric -> value).
    caption : str
        Table caption text.

    Returns
    -------
    str
        A complete LaTeX table string (``\\begin{table}...\\end{table}``).
    """
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
    """Render *summary* as a GitHub-Flavoured Markdown table.

    Parameters
    ----------
    summary : dict
        Aggregated benchmark metrics (strategy -> metric -> value).

    Returns
    -------
    str
        A Markdown-formatted table string.
    """
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
