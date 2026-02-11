"""
bigrag.data.embedder -- Sentence-transformer batch embedding.

Uses a HuggingFace sentence-transformers model to encode a list of
text strings into dense vectors, batching for GPU/CPU throughput.
"""

from __future__ import annotations

import numpy as np


def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode *texts* into a 2-D numpy array of embeddings.

    Parameters
    ----------
    texts : list[str]
        Input sentences / review texts.
    model_name : str
        HuggingFace sentence-transformers model identifier.
    batch_size : int
        Number of texts encoded per forward pass.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), embedding_dim)``.
    """
    raise NotImplementedError("generate_embeddings is not yet implemented")
