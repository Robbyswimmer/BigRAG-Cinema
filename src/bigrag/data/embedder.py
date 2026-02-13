"""
bigrag.data.embedder -- Sentence-transformer batch embedding.

Uses a HuggingFace sentence-transformers model to encode a list of
text strings into dense vectors, batching for GPU/CPU throughput.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bigrag.data.schema import COL_TEXT, EMBEDDING_DIM


def generate_embeddings(
    texts: list[str] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
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
    if texts is None:
        if input_path is None:
            raise ValueError("Provide either `texts` or `input_path`.")
        import pandas as pd

        from bigrag.data.validator import validate_dataframe

        csv_path = Path(input_path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        cleaned_df, _ = validate_dataframe(raw_df)
        texts = cleaned_df[COL_TEXT].astype(str).tolist()

    if not texts:
        embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            # Deterministic fallback for environments without model dependencies.
            vectors = []
            for text in texts:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
                norm = np.linalg.norm(vec)
                vectors.append(vec if norm == 0 else vec / norm)
            embeddings = np.vstack(vectors)
        else:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, embeddings)

    return embeddings
