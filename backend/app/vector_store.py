"""Tiny in-memory vector store with cosine similarity search.

Kept dependency-free on purpose so the retrieval math is visible.
Swap for pgvector / FAISS / Pinecone when data grows.
"""
import numpy as np

from .chunker import Chunk


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = _normalize(vectors)
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q = _normalize(query_vec.reshape(1, -1))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def documents(self) -> list[str]:
        return sorted({c.doc_name for c in self.chunks})
