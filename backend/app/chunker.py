"""Split documents into overlapping word chunks.

Overlap keeps a sentence that straddles a boundary retrievable from either side.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str


def chunk_text(text: str, doc_name: str, size: int = 120, overlap: int = 30) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, step, i = [], size - overlap, 0
    while i < len(words):
        piece = " ".join(words[i : i + size])
        chunks.append(Chunk(doc_name=doc_name, chunk_id=len(chunks), text=piece))
        if i + size >= len(words):
            break
        i += step
    return chunks
