"""
Sentence-embedding helpers for semantic book matching (all-MiniLM-L6-v2).
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_model_lock = threading.Lock()
_encode_lock = threading.Lock()
_model_load_failed = False
_CACHE_MAX = 1500
_embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _cache_key(text: str) -> str:
    return text.strip().lower()[:800]


def get_model():
    global _model, _model_load_failed
    if _model_load_failed:
        raise RuntimeError("Sentence embedding model failed to load earlier in this process.")
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(_MODEL_NAME)
            except Exception as exc:
                _model_load_failed = True
                logger.exception("Failed to load sentence-transformers model: %s", exc)
                raise RuntimeError("Could not load embedding model") from exc
        return _model


def _cache_get(key: str) -> Optional[np.ndarray]:
    vec = _embedding_cache.get(key)
    if vec is None:
        return None
    _embedding_cache.move_to_end(key)
    return vec


def _cache_set(key: str, vec: np.ndarray) -> None:
    if key in _embedding_cache:
        _embedding_cache.move_to_end(key)
    _embedding_cache[key] = vec.copy()
    while len(_embedding_cache) > _CACHE_MAX:
        _embedding_cache.popitem(last=False)


def encode_texts_normalized(texts: List[str]) -> np.ndarray:
    """
    Return L2-normalized embeddings (rows); cosine similarity = dot product.
    """
    with _encode_lock:
        model = get_model()
        dim = model.get_sentence_embedding_dimension()
        n = len(texts)
        if n == 0:
            return np.zeros((0, dim), dtype=np.float32)

        out = np.zeros((n, dim), dtype=np.float32)
        pending_idx: List[int] = []
        pending_texts: List[str] = []

        for i, raw in enumerate(texts):
            key = _cache_key(raw)
            hit = _cache_get(key)
            if hit is not None:
                out[i] = hit
            else:
                pending_idx.append(i)
                pending_texts.append(raw)

        if pending_texts:
            encoded = model.encode(
                pending_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for j, idx in enumerate(pending_idx):
                vec = encoded[j].astype(np.float32, copy=False)
                _cache_set(_cache_key(texts[idx]), vec)
                out[idx] = vec

        return out


def cosine_scores_to_query(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    q = query_embedding.reshape(-1)
    return doc_embeddings @ q
