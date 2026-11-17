"""Flask API for enterprise-rag-search."""
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from pypdf import PdfReader

from .rag import RAGPipeline

app = Flask(__name__)
CORS(app)
rag = RAGPipeline()

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_docs"


def _read_upload(f) -> str:
    if f.filename.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": rag.store.documents()}


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="no files uploaded"), 400
    added = {f.filename: rag.ingest(_read_upload(f), f.filename) for f in files}
    return {"chunks_added": added, "documents": rag.store.documents()}


@app.post("/api/ask")
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="question is required"), 400
    return rag.ask(question)


@app.post("/api/load-samples")
def load_samples():
    for p in sorted(SAMPLE_DIR.glob("*.md")):
        if p.name not in rag.store.documents():
            rag.ingest(p.read_text(), p.name)
    return {"documents": rag.store.documents()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
