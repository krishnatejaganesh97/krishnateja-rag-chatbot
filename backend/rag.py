# Teja RAG Chatbot — Product 1
# A working RAG doc chatbot starter: FastAPI + LangChain + ChromaDB (with a
# local demo fallback so it runs with NO API key).
#
# What Teja learns from this project (see README.md for the full walkthrough):
#   1. FastAPI — how a chat API is structured
#   2. Chunking — splitting docs into retrievable pieces (LangChain splitter)
#   3. Embeddings — turning text into vectors (OpenAI / Azure / mock demo)
#   4. Vector search — ChromaDB retrieval, with an in-memory fallback
#   5. RAG prompting — grounding an LLM answer in retrieved chunks
#   6. Citations — every answer must point at the source chunk it came from
#   7. Evals — a script that checks answers, not vibes
#
# Run:  python -m uvicorn backend.main:app --reload --port 8000
# Then open http://localhost:8000 in a browser (frontend is served by FastAPI).

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Config — all secrets come from environment variables. Never hard-code keys.
# See .env.example. Copy it to .env and fill it in only on your own machine.
# ---------------------------------------------------------------------------
# LLM_PROVIDER: "openai" | "azure" | "mock" (default: mock if no key is set)
# OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini)
# AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT
# CHROMA_PATH (default ./data/chroma_db), USE_CHROMA ("true"/"false")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_docs"


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    id: str
    text: str
    source: str  # file name, e.g. "technobot.md" — this is what citations show
    score: float = 0.0


# ---------------------------------------------------------------------------
# 1. CHUNKING — LangChain's RecursiveCharacterTextSplitter when available.
#    Why chunk? An LLM can't fit a whole document in a prompt, and retrieval
#    works best on small, self-contained passages (~500 chars with overlap).
# ---------------------------------------------------------------------------
def chunk_text(text: str, source: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(text)
    except Exception:
        # Fallback: same idea, plain Python — split on paragraphs, then sentences.
        pieces = _simple_split(text, chunk_size, chunk_overlap)
    return [
        Chunk(id=f"{source}#{i}", text=p.strip(), source=source)
        for i, p in enumerate(pieces) if p.strip()
    ]


def _simple_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    for para in paras:
        while len(para) > chunk_size:
            cut = para.rfind(". ", 0, chunk_size)
            cut = cut + 1 if cut > 100 else chunk_size
            out.append(para[:cut])
            para = para[max(0, cut - overlap):]
        out.append(para)
    return out


def load_sample_docs() -> List[Chunk]:
    """Load every .md/.txt file in data/sample_docs/ and chunk it."""
    chunks: List[Chunk] = []
    for path in sorted(SAMPLE_DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), source=path.name))
    return chunks


# ---------------------------------------------------------------------------
# 2. EMBEDDINGS — vectors for each chunk.
#    Real mode: OpenAI / Azure OpenAI embeddings (needs a key).
#    Demo mode: a deterministic hash-based mock embedding. It is NOT a real
#    semantic embedding — it only lets Teja test the whole pipeline locally
#    without spending money or sharing a key. Retrieval in demo mode therefore
#    also uses keyword overlap (see retrieve()) so answers stay grounded.
# ---------------------------------------------------------------------------
def _mock_embed(text: str, dim: int = 384) -> List[float]:
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Embedder:
    def __init__(self) -> None:
        self.provider = "mock"
        self._openai = None
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.provider = "azure"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            resp = client.embeddings.create(
                input=texts, model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
            return [d.embedding for d in resp.data]
        if self.provider == "azure":
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            resp = client.embeddings.create(
                input=texts,
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            )
            return [d.embedding for d in resp.data]
        return [_mock_embed(t) for t in texts]


# ---------------------------------------------------------------------------
# 3. VECTOR SEARCH — ChromaDB when available, in-memory fallback otherwise.
#    The interface is the same either way: add(chunks) then search(query, k).
#    ChromaDB persists to ./data/chroma_db so the index survives restarts.
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []
        self._chroma = None
        self._collection = None
        if os.getenv("USE_CHROMA", "true").lower() == "true":
            try:
                import chromadb

                self._chroma = chromadb.PersistentClient(
                    path=str(BASE_DIR / os.getenv("CHROMA_PATH", "data/chroma_db"))
                )
                self._collection = self._chroma.get_or_create_collection("teja_docs")
            except Exception as e:  # Chroma not installed / unavailable -> fallback
                print(f"[rag] ChromaDB unavailable ({e}); using in-memory store.")

    @property
    def backend(self) -> str:
        return "chromadb" if self._collection is not None else "in-memory"

    def add(self, chunks: List[Chunk]) -> None:
        if self._collection is not None:
            existing = set(self._collection.get().get("ids", []))
            new = [c for c in chunks if c.id not in existing]
            if new:
                self._collection.add(
                    ids=[c.id for c in new],
                    documents=[c.text for c in new],
                    metadatas=[{"source": c.source} for c in new],
                    embeddings=self.embedder.embed([c.text for c in new]),
                )
            # Keep an in-memory copy too, so citations work the same way.
            self.chunks = chunks
        else:
            self.chunks = chunks
            self.vectors = self.embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        if self._collection is not None:
            count = self._collection.count() or k
            # Demo/mock mode: hash embeddings are a weak signal, so pull a
            # wider set from Chroma and re-rank by real keyword overlap.
            # Real (OpenAI/Azure) embeddings are a strong signal — trust top-k.
            n = min(10, count) if self.embedder.provider == "mock" else min(k, count)
            res = self._collection.query(
                query_embeddings=self.embedder.embed([query]),
                n_results=n,
            )
            out = []
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                out.append(Chunk(id=meta.get("source", ""), text=doc, source=meta.get("source", ""), score=1 - (dist or 0)))
            return _rerank_by_keywords(query, out)[:k] if self.embedder.provider == "mock" else out
        qvec = self.embedder.embed([query])[0]
        scored = sorted(
            enumerate(self.chunks),
            key=lambda pair: _cosine(qvec, self.vectors[pair[0]]),
            reverse=True,
        )
        ranked = [c for _, c in scored]
        # In mock mode, keyword overlap is a better signal than hash vectors.
        if self.embedder.provider == "mock":
            return _rerank_by_keywords(query, ranked[: k * 2])[:k]
        return ranked[:k]


def _rerank_by_keywords(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """Boost chunks that share real words with the question (demo mode)."""
    words = set(re.findall(r"[a-z0-9]+", query.lower())) - {
        "what", "where", "when", "who", "how", "does", "did", "the", "a", "an",
        "is", "are", "was", "were", "in", "on", "at", "of", "for", "to", "and",
    }
    def kw_score(c: Chunk) -> int:
        text = c.text.lower()
        return sum(1 for w in words if w in text)
    return sorted(chunks, key=kw_score, reverse=True)


# ---------------------------------------------------------------------------
# 4. RAG — retrieve, then answer ONLY from the retrieved chunks.
#    Real mode: call OpenAI / Azure Chat with a grounding prompt.
#    Demo mode: extractive answer — return the most relevant sentences from
#    the retrieved chunks, with citations. No hallucinating, no invented facts.
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    text: str
    sources: List[dict] = field(default_factory=list)  # [{source, text, score}]
    mode: str = "mock"  # "openai" | "azure" | "mock"
    model: str = "mock-extractive"


class RAGChatbot:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore(self.embedder)
        self.store.add(load_sample_docs())
        self.mode = self.embedder.provider

    def ask(self, question: str, k: int = 3) -> Answer:
        chunks = self.store.search(question, k=k)
        if not chunks:
            return Answer("I don't know — no relevant documents were found.", [], self.mode)
        sources = [
            {"source": c.source, "text": c.text, "score": round(c.score, 3)}
            for c in chunks
        ]
        if self.mode in ("openai", "azure"):
            return Answer(self._llm_answer(question, chunks), sources, self.mode,
                          os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        # Deduplicate sources by file for a cleaner UI (keep first per file).
        seen, uniq = set(), []
        for s in sources:
            if s["source"] not in seen:
                seen.add(s["source"])
                uniq.append(s)
        return Answer(self._extractive_answer(question, chunks), uniq, "mock")

    def _llm_answer(self, question: str, chunks: List[Chunk]) -> str:
        context = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say 'I don't know based on the documents provided.' "
            "Cite the source file for each fact, e.g. [technobot.md].\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        if self.mode == "azure":
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        else:
            from openai import OpenAI

            client = OpenAI()
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extractive_answer(question: str, chunks: List[Chunk]) -> str:
        # Pick the 2-3 sentences from the retrieved chunks most relevant to
        # the question, and cite their source files. This is how a RAG system
        # should behave when it cannot call an LLM: ground, don't invent.
        # Score on CONTENT words only (stopwords like "what/does/for" would
        # otherwise let unrelated sentences win, e.g. a certifications line
        # beating the PhD line for "Where does Teja study for his PhD?").
        stop = {
            "what", "where", "when", "who", "how", "does", "did", "do", "the",
            "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
            "for", "to", "and", "his", "her", "he", "she", "it", "does",
        }
        content_words = {
            w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop
        }
        sentences: List[tuple[int, str, str]] = []
        for c in chunks:
            for sent in re.split(r"(?<=[.!])\s+", c.text):
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                score = sum(1 for w in content_words if w in sent.lower())
                sentences.append((score, sent, c.source))
        sentences.sort(key=lambda s: s[0], reverse=True)
        top = [s for s in sentences if s[0] > 0][:3]
        if not top:
            # No sentence shares a content word with the question — return
            # the top retrieved chunk verbatim rather than inventing an answer.
            c = chunks[0]
            return f"Based on [{c.source}]: {c.text}"
        lines = [f"- {sent} [{src}]" for _, sent, src in top]
        return "Based on the documents:\n" + "\n".join(lines)


# Global singleton, built once at import — FastAPI reuses it for every request.
_bot: Optional[RAGChatbot] = None


def get_bot() -> RAGChatbot:
    global _bot
    if _bot is None:
        _bot = RAGChatbot()
    return _bot
