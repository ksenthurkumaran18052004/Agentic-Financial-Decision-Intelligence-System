"""
Build and persist a FAISS index from the knowledge base.
Uses sentence-transformers (all-MiniLM-L6-v2) — fast, runs locally.
"""

import numpy as np
import faiss
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAISS_INDEX_PATH, METADATA_PATH
from src.rag.knowledge_base import get_all_documents, get_texts

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def build_index(save: bool = True) -> tuple[faiss.Index, list[dict]]:
    docs  = get_all_documents()
    texts = get_texts()

    print(f"Embedding {len(texts)} knowledge base documents...")
    vectors = embed(texts).astype("float32")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product = cosine on normalised vecs
    index.add(vectors)

    if save:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(docs, f)
        print(f"FAISS index saved  → {FAISS_INDEX_PATH}")
        print(f"Metadata saved     → {METADATA_PATH}")

    return index, docs


def load_index() -> tuple[faiss.Index, list[dict]]:
    if not FAISS_INDEX_PATH.exists():
        return build_index()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


if __name__ == "__main__":
    build_index(save=True)
