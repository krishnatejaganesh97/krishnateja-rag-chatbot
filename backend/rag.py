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

# Chunking defaults (see chunk_text()). 500/50 was chosen by experiment
# (scripts/eval.py, 8 questions): 500/50 and 800/100 both score 8/8, but
# 500/50 keeps 1-3 paragraphs per chunk — big enough for context, small
# enough that a chunk stays on-topic and citations stay precise. 800/100
# merges whole docs into 4 chunks, so a citation points at a much bigger
# passage. 300/30 drops to 7/8 (the SQL Server/MongoDB sentence loses its
# surrounding context). Re-run scripts/eval.py if you change these.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Query expansion for mock-mode keyword retrieval. The mock embedding is
# not semantic, so a question word that never appears in the docs (e.g.
# "databases" — the docs say "SQL Server and MongoDB") would score zero.
# These are small, doc-grounded synonym maps, not answer leakage: they map
# a generic word to the concrete terms the docs actually use.
QUERY_EXPANSIONS = {
    "database": ["sql", "mongodb", "server"],
    "degree": ["master", "bachelor", "science", "engineering"],
}

# Light stemming so "build" matches "built", "worked" matches "work", etc.
# Applied identically to questions and chunks, so both sides stay comparable.
_IRREGULAR_STEMS = {"built": "build", "building": "build"}


def _stem(word: str) -> str:
    if word in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[word]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        # Strip "es" only when the singular ends in s/x/z/ch/sh
        # ("classes" -> "class", "boxes" -> "box"); otherwise strip just
        # "s" ("databases" -> "database", NOT "databas" — the old rule
        # broke query expansion for "databases", which is why that eval
        # question failed).
        if word.endswith(("sses", "xes", "zes", "ches", "shes")):
            return word[:-2]
        return word[:-1]
    return word


def _tokens(text: str) -> List[str]:
    """Lowercase alphanumeric words, stemmed. Used for questions and chunks."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _query_tokens(query: str) -> List[str]:
    """Content tokens for a question: stemmed, stopwords removed, expanded.

    Expansion adds the concrete doc terms for a generic question word
    (see QUERY_EXPANSIONS) — e.g. "databases" also matches "sql" and
    "mongodb" — so retrieval works on meaning, not just exact words.
    """
    stop = {
        "what", "where", "when", "who", "how", "does", "did", "do", "the",
        "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
        "for", "to", "and", "his", "her", "he", "she", "it", "which",
        "with", "from", "by", "as", "that", "this",
    }
    out: List[str] = []
    for w in _tokens(query):
        if w in stop:
            continue
        if w not in out:
            out.append(w)
        for extra in QUERY_EXPANSIONS.get(w, []):
            if extra not in out:
                out.append(extra)
    return out


def _idf_weights(chunks: List[Chunk]) -> dict:
    """Inverse document frequency over the indexed chunks.

    A word in every chunk ("teja") tells retrieval nothing; a word in one
    chunk ("cumberlands", "satyam", "mongodb") is highly discriminative.
    idf(w) = log((N+1) / (df+1)) + 1, so common words score ~1 and rare
    words score ~2.5 on this 8-chunk index.
    """
    n = len(chunks) or 1
    df: dict = {}
    for c in chunks:
        for w in set(_tokens(c.text)):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((n + 1) / (d + 1)) + 1.0 for w, d in df.items()}


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
def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
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
        self.idf: dict = {}  # set in add(); see _idf_weights()
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
        # IDF weights are recomputed whenever the index changes (sample docs
        # at startup, plus any uploaded docs) — retrieval ranking depends on
        # them, so they must cover exactly the indexed chunks.
        self.chunks = chunks
        self.idf = _idf_weights(chunks)
        if self._collection is not None:
            stored = dict(zip(
                self._collection.get().get("ids", []),
                self._collection.get().get("documents", []),
            ))
            wanted = {c.id: c for c in chunks}
            # Chroma persists across restarts, so a changed chunk_size or an
            # edited doc must UPDATE the stored chunk, not be silently
            # skipped because its id already exists (the old bug: stale
            # chunks survived a re-chunk). Only new/changed chunks are
            # re-embedded, so real-embedding mode doesn't pay on every start.
            new = [c for c in chunks if c.id not in stored]
            changed = [c for c in chunks if c.id in stored and stored[c.id] != c.text]
            stale = [i for i in stored if i not in wanted]
            if stale:
                self._collection.delete(ids=stale)
            if new or changed:
                upsert = new + changed
                self._collection.upsert(
                    ids=[c.id for c in upsert],
                    documents=[c.text for c in upsert],
                    metadatas=[{"source": c.source} for c in upsert],
                    embeddings=self.embedder.embed([c.text for c in upsert]),
                )
        else:
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
            return _rerank_by_keywords(query, out, self.idf)[:k] if self.embedder.provider == "mock" else out
        qvec = self.embedder.embed([query])[0]
        scored = sorted(
            enumerate(self.chunks),
            key=lambda pair: _cosine(qvec, self.vectors[pair[0]]),
            reverse=True,
        )
        ranked = [c for _, c in scored]
        # In mock mode, keyword overlap is a better signal than hash vectors.
        if self.embedder.provider == "mock":
            return _rerank_by_keywords(query, ranked[: k * 2], self.idf)[:k]
        return ranked[:k]


def _rerank_by_keywords(query: str, chunks: List[Chunk], idf: dict | None = None) -> List[Chunk]:
    """Rank chunks by IDF-weighted keyword overlap with the question.

    The old version counted every shared word equally and matched by
    substring ("ai" matched "available", "teja" counted as much as
    "cumberlands"), so a chunk scored just for saying "Teja" could outrank
    the chunk with the actual answer. Now:
      * whole-word, stemmed matching ("build" matches "built"),
      * each word weighted by IDF — "teja" (in every chunk) is worth ~1,
        "cumberlands"/"satyam"/"mongodb" (in one chunk) ~2.5,
      * generic question words expand to the concrete terms the docs use
        (see _query_tokens / QUERY_EXPANSIONS),
      * ties keep the vector-search order (stable sort).
    """
    idf = idf or {}
    default_idf = math.log((len(chunks) + 1) / 1) + 1.0 if chunks else 1.0
    tokens = _query_tokens(query)

    def kw_score(c: Chunk) -> float:
        present = set(_tokens(c.text))
        return sum(idf.get(w, default_idf) for w in tokens if w in present)

    return [c for _, c in sorted(
        enumerate(chunks), key=lambda p: (-kw_score(p[1]), p[0])
    )]


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
        # Mock mode: extractive answer. Only the files the answer actually
        # cites are returned as sources — previously every retrieved chunk's
        # file was listed, so a PhD answer "cited" elite-tech.md even though
        # no sentence from it was used.
        text, cited = self._extractive_answer(question, chunks)
        cited_sources = [s for s in sources if s["source"] in cited]
        # Deduplicate sources by file for a cleaner UI (keep first per file).
        seen, uniq = set(), []
        for s in cited_sources:
            if s["source"] not in seen:
                seen.add(s["source"])
                uniq.append(s)
        return Answer(text, uniq, "mock")

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

    def _extractive_answer(self, question: str, chunks: List[Chunk]) -> tuple[str, set]:
        # Pick the sentences from the retrieved chunks most relevant to the
        # question, and cite their source files. This is how a RAG system
        # should behave when it cannot call an LLM: ground, don't invent.
        #
        # Scoring is IDF-weighted over stemmed, whole-word matches (same
        # scheme as _rerank_by_keywords), so a sentence scores for the
        # discriminative words ("cumberlands", "mongodb") and not just for
        # saying "Teja" — the old equal-weight version let a sentence like
        # "Teja reported to Satyam B..." into the answer to "Where does Teja
        # study for his PhD?" because it contained "Teja".
        #
        # Only sentences scoring at least half as well as the best sentence
        # are kept: the best sentence sets the bar for what "relevant"
        # means for this question, so filler sentences from a retrieved
        # chunk don't ride along. Returns (answer text, cited source files).
        idf = self.store.idf or {}
        default_idf = math.log((len(self.store.chunks) + 1) / 1) + 1.0 if self.store.chunks else 1.0
        tokens = _query_tokens(question)
        sentences: List[tuple[float, str, str]] = []
        for c in chunks:
            for sent in re.split(r"(?<=[.!])\s+", c.text):
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                present = set(_tokens(sent))
                score = sum(idf.get(w, default_idf) for w in tokens if w in present)
                sentences.append((score, sent, c.source))
        sentences.sort(key=lambda s: s[0], reverse=True)
        if not sentences or sentences[0][0] <= 0:
            # No sentence shares a content word with the question — return
            # the top retrieved chunk verbatim rather than inventing an answer.
            c = chunks[0]
            return f"Based on [{c.source}]: {c.text}", {c.source}
        best = sentences[0][0]
        top = [s for s in sentences if s[0] >= 0.5 * best][:3]
        lines = [f"- {sent} [{src}]" for _, sent, src in top]
        return ("Based on the documents:\n" + "\n".join(lines),
                {src for _, _, src in top})


# Global singleton, built once at import — FastAPI reuses it for every request.
_bot: Optional[RAGChatbot] = None


def get_bot() -> RAGChatbot:
    global _bot
    if _bot is None:
        _bot = RAGChatbot()
    return _bot
