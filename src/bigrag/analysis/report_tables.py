"""
bigrag.analysis.report_tables -- Generate LaTeX and Markdown tables from metrics.

Transforms benchmark summary dicts into formatted table strings
that can be copy-pasted directly into a LaTeX document or Markdown
README.
"""

from __future__ import annotations


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
    raise NotImplementedError("generate_latex_table is not yet implemented")


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
    raise NotImplementedError("generate_markdown_table is not yet implemented")
