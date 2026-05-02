"""
RAG retriever — given a transaction context, returns relevant fraud rules.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.rag.embeddings import load_index, embed

_index  = None
_docs   = None


def _ensure_loaded():
    global _index, _docs
    if _index is None:
        _index, _docs = load_index()


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """Return top_k relevant fraud knowledge entries for the given query."""
    _ensure_loaded()
    vec = embed([query]).astype("float32")
    scores, indices = _index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            doc = dict(_docs[idx])
            doc["relevance_score"] = round(float(score), 4)
            results.append(doc)
    return results


def retrieve_for_transaction(tx: dict, top_k: int = 4) -> list[dict]:
    """Build a query from transaction features and retrieve relevant rules."""
    parts = [
        f"Transaction: amount ${tx.get('amount', 0):.2f}",
        f"merchant type {tx.get('merchant_type', 'unknown')}",
        f"location {tx.get('location', 'unknown')}",
        f"hour {tx.get('hour', 12)}",
    ]
    if tx.get("is_late_night"):
        parts.append("late night")
    if tx.get("amount_vs_avg", 1.0) > 3.0:
        parts.append("unusually high amount")
    if not tx.get("is_home_city", 1):
        parts.append("foreign location")
    if not tx.get("is_known_merchant", 1):
        parts.append("unknown merchant")

    query = ". ".join(parts)
    return retrieve(query, top_k=top_k)
