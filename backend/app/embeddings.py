"""Embedding providers.

- OpenAIEmbedder: real semantic embeddings (text-embedding-3-small).
- HashEmbedder: offline fallback (hashed bag-of-words) so the app runs with no API key.
  Not semantic, but good enough to play with keyword-style retrieval.
"""
import hashlib
import os
import re

import numpy as np


STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been
do does did have has had i me my we our you your it its this that these those what which who whom
when where why how should can could would will shall may might must use used get gets not no""".split())


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in STOPWORDS:
                    continue
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
