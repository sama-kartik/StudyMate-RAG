"""Cosine-similarity search over stored chunks.

The web app does this in the browser (public/static/app.js); this Python version is the
same algorithm, used by the evaluation script and scripts/. Vectors are unit
length, so cosine similarity = dot product.
"""
from typing import Dict, List

# Gemini embeddings: clearly related passages score ~0.65+, unrelated text ~0.5.
# Below this the passage is treated as irrelevant (keep in sync with app.js).
MIN_SCORE = 0.5


def top_k(query_vec: List[float], chunks: List[Dict], k: int = 5,
          min_score: float = MIN_SCORE) -> List[Dict]:
    scored = []
    for c in chunks:
        s = sum(a * b for a, b in zip(query_vec, c["embedding"]))
        if s >= min_score:
            scored.append({**c, "score": round(s, 4)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]
