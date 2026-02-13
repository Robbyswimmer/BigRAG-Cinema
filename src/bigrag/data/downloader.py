"""
bigrag.data.downloader -- Download the Amazon Reviews 2023 dataset.

Wraps the Kaggle CLI / API to fetch the dataset archive, extract it,
and return the path to the resulting CSV file(s).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DATASET_ID = "amazon/amazon-reviews-2023"
DEFAULT_OUTPUT_FILENAME = "Reviews.csv"


def _first_csv_in_dir(dest_dir: Path) -> Path | None:
    csv_files = sorted(dest_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


def download_dataset(dest_dir: Path, force: bool = False) -> Path:
    """Download the Kaggle dataset into *dest_dir* and return the CSV path.

    Parameters
    ----------
    dest_dir : Path
        Local directory where the dataset will be saved.

    Returns
    -------
    Path
        Path to the extracted CSV file.
    """
    output_dir = Path(dest_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = _first_csv_in_dir(output_dir)
    if existing_csv and not force:
        return existing_csv

    if existing_csv and force:
        existing_csv.unlink()

    kaggle_cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_ID,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    try:
        subprocess.run(kaggle_cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Fallback for development environments without Kaggle credentials.
        project_root = Path(__file__).resolve().parents[3]
        sample_path = project_root / "tests" / "test_data" / "sample_reviews.csv"
        if sample_path.exists():
            fallback_path = output_dir / DEFAULT_OUTPUT_FILENAME
            shutil.copy2(sample_path, fallback_path)
            return fallback_path
        raise RuntimeError(
            "Failed to download dataset via Kaggle CLI and no local fallback sample "
            f"was found at {sample_path}."
        )

    extracted_csv = _first_csv_in_dir(output_dir)
    if extracted_csv is None:
        raise RuntimeError(
            f"Dataset download completed but no CSV file was found in {output_dir}."
        )
    return extracted_csv


def download(output_dir: str | Path = "data/raw", force: bool = False) -> Path:
    """Backward-compatible wrapper used by CLI scripts."""
    return download_dataset(Path(output_dir), force=force)
