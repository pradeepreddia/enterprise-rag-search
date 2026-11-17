"""RAG pipeline: ingest -> chunk -> embed -> retrieve -> generate with citations."""
import os

from .chunker import chunk_text
from .embeddings import get_embedder
from .vector_store import VectorStore

PROMPT = """You are an internal knowledge assistant.
Answer ONLY from the numbered sources below. Cite sources inline like [1], [2].
If the sources don't contain the answer, say "I couldn't find that in the documents."

Sources:
{sources}

Question: {question}
Answer:"""


class RAGPipeline:
    def __init__(self, embedder=None, llm=None):
        self.embedder = embedder or get_embedder()
        self.store = VectorStore()
        self.llm = llm  # callable(prompt) -> str; None = auto

    def ingest(self, text: str, doc_name: str) -> int:
        chunks = chunk_text(text, doc_name)
        if not chunks:
            return 0
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(chunks)

    def ask(self, question: str, k: int = 4) -> dict:
        hits = self.store.search(self.embedder.embed([question])[0], k=k)
        if not hits:
            return {"answer": "I couldn't find that in the documents.", "citations": []}

        sources = "\n\n".join(f"[{i}] ({c.doc_name}) {c.text}" for i, (c, _) in enumerate(hits, 1))
        answer = self._generate(PROMPT.format(sources=sources, question=question), hits)
        citations = [
            {"id": i, "doc": c.doc_name, "chunk": c.chunk_id, "score": round(s, 3), "snippet": c.text[:200]}
            for i, (c, s) in enumerate(hits, 1)
        ]
        return {"answer": answer, "citations": citations}

    def _generate(self, prompt: str, hits) -> str:
        if self.llm:
            return self.llm(prompt)
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI

            resp = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content
        # Offline mode: return the best-matching passage instead of a generated answer.
        top = hits[0][0]
        return f"(offline mode: no OPENAI_API_KEY) Most relevant passage [1] from {top.doc_name}: {top.text[:400]}"
