"""Embedding providers.

- OpenAIEmbedder: real semantic embeddings (text-embedding-3-small).
- HashEmbedder: offline fallback (hashed bag-of-words) so the app runs with no API key.
  Not semantic, but good enough to play with keyword-style retrieval.
"""
import hashlib
import os
import re

import numpy as np


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                out[row, idx] += 1.0
        return out


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


def get_embedder():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return HashEmbedder()
