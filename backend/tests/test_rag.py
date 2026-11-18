import numpy as np
import pytest

from app.chunker import chunk_text
from app.embeddings import HashEmbedder
from app.rag import RAGPipeline
from app.vector_store import VectorStore


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, "doc", size=100, overlap=20)
    assert len(chunks) == 3
    assert chunks[1].text.split()[0] == "w80"


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", "doc", size=10, overlap=10)


def test_vector_search_ranks_closest_first():
    store = VectorStore()
    chunks = chunk_text("alpha", "a") + chunk_text("beta", "b")
    store.add(chunks, np.array([[1.0, 0.0], [0.0, 1.0]]))
    hits = store.search(np.array([0.9, 0.1]), k=2)
    assert hits[0][0].doc_name == "a"


def test_rag_returns_citations_offline():
    rag = RAGPipeline(embedder=HashEmbedder(), llm=lambda p: "PTO is 20 days [1]")
    rag.ingest("Employees get 20 days of paid time off per year.", "hr.md")
    rag.ingest("The VPN must be used on public wifi.", "it.md")
    out = rag.ask("How many days of paid time off?")
    assert out["citations"][0]["doc"] == "hr.md"
    assert "[1]" in out["answer"]


def test_empty_store():
    rag = RAGPipeline(embedder=HashEmbedder())
    assert rag.ask("anything")["citations"] == []


@pytest.mark.parametrize("question,doc", [
    ("How many PTO days do employees get?", "hr_handbook.md"),
    ("When do deployments go out?", "onboarding.md"),
    ("What should I use on public wifi?", "it_security.md"),
])
def test_sample_questions_hit_right_doc(question, doc):
    from pathlib import Path
    rag = RAGPipeline(embedder=HashEmbedder(), llm=lambda p: "ok")
    for f in sorted((Path(__file__).resolve().parents[2] / "sample_docs").glob("*.md")):
        rag.ingest(f.read_text(), f.name)
    assert rag.ask(question)["citations"][0]["doc"] == doc
