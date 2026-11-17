import { useEffect, useState } from "react";

export default function App() {
  const [docs, setDocs] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setDocs(d.documents));
  }, []);

  const upload = async (e) => {
    const form = new FormData();
    [...e.target.files].forEach((f) => form.append("files", f));
    const d = await (await fetch("/api/upload", { method: "POST", body: form })).json();
    setDocs(d.documents);
  };

  const loadSamples = async () => {
    const d = await (await fetch("/api/load-samples", { method: "POST" })).json();
    setDocs(d.documents);
  };

  const ask = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    setResult(await r.json());
    setLoading(false);
  };

  return (
    <main>
      <h1>Enterprise RAG Search</h1>
      <p className="sub">Upload docs, ask questions, get answers with sources.</p>

      <section className="card">
        <input type="file" multiple accept=".txt,.md,.pdf" onChange={upload} />
        <button className="ghost" onClick={loadSamples}>Load sample docs</button>
        <div className="docs">{docs.length ? docs.map((d) => <span key={d}>{d}</span>) : "No documents yet"}</div>
      </section>

      <form className="card ask" onSubmit={ask}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. How many PTO days do we get?" />
        <button disabled={loading}>{loading ? "Thinking…" : "Ask"}</button>
      </form>

      {result && (
        <section className="card">
          <h3>Answer</h3>
          <p>{result.answer}</p>
          <h3>Sources</h3>
          {result.citations.map((c) => (
            <div key={c.id} className="cite">
              <b>[{c.id}] {c.doc}</b> <small>score {c.score}</small>
              <p>{c.snippet}…</p>
            </div>
          ))}
        </section>
      )}
    </main>
  );
}
