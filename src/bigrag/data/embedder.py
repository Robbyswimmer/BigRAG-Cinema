"""
bigrag.data.embedder -- Sentence-transformer batch embedding.

Uses a HuggingFace sentence-transformers model to encode a list of
text strings into dense vectors, batching for GPU/CPU throughput.
Automatically uses Apple MPS acceleration when available.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bigrag.data.schema import COL_TEXT, EMBEDDING_DIM

# Module-level cache so the model is loaded once and reused.
_MODEL_CACHE: dict = {}


def _pick_device() -> str:
    """Return the best available torch device string."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str):
    """Return a cached SentenceTransformer, loading it once on first call."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        device = _pick_device()
        print(f"Loading model '{model_name}' on device '{device}' ...")
        _MODEL_CACHE[model_name] = (SentenceTransformer(model_name, device=device), device)
    return _MODEL_CACHE[model_name]


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
        from bigrag.data.schema import load_raw_dataframe
        from bigrag.data.validator import validate_dataframe

        print(f"Loading data from {input_path} ...")
        raw_df = load_raw_dataframe(input_path)
        print(f"  Loaded {len(raw_df):,} raw rows. Validating ...")
        cleaned_df, report = validate_dataframe(raw_df)
        print(
            f"  Validation complete: {report['final_rows']:,} rows "
            f"({report['invalid_rows']:,} invalid, "
            f"{report['duplicates_removed']:,} duplicates removed)"
        )
        texts = cleaned_df[COL_TEXT].astype(str).tolist()

    if not texts:
        embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        try:
            model, device = _get_model(model_name)
        except ImportError:
            print("sentence-transformers not installed — using deterministic hash fallback")
            vectors = []
            for text in texts:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
                norm = np.linalg.norm(vec)
                vectors.append(vec if norm == 0 else vec / norm)
            embeddings = np.vstack(vectors)
        else:
            total = len(texts)
            show_progress = total > 100
            if show_progress:
                print(
                    f"Encoding {total:,} texts | batch_size={batch_size} | "
                    f"device={device} | dim={EMBEDDING_DIM}"
                )

            start = time.perf_counter()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            elapsed = time.perf_counter() - start

            if show_progress:
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"  Done: {total:,} texts in {elapsed:.1f}s "
                    f"({rate:,.0f} texts/sec)"
                )

    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, embeddings)
        size_mb = out.stat().st_size / 1e6
        print(f"  Saved embeddings to {out} ({size_mb:.1f} MB)")

    return embeddings
