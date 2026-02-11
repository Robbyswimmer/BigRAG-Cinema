"""
bigrag.data.downloader -- Download the Amazon Reviews 2023 dataset.

Wraps the Kaggle CLI / API to fetch the dataset archive, extract it,
and return the path to the resulting CSV file(s).
"""

from pathlib import Path


def download_dataset(dest_dir: Path) -> Path:
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
    raise NotImplementedError("download_dataset is not yet implemented")
