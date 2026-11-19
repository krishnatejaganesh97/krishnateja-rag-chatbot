# Teja RAG Chatbot — Product 1

A working **RAG (Retrieval-Augmented Generation) doc chatbot**, built to learn
by building. Ask it about Teja's PhD, Technobot, and Elite Tech work — every
answer cites the source document it came from.

This is Product 1 of 3 in Teja's 60-day plan to get an AI job in 1–2 months:
Product 2 is a LangGraph agentic workflow, Product 3 is a PhD research
paper agent — reads a pile of papers, summarizes them, finds
contradictions, and builds a literature review outline with citations.
All three projects are kept open source.

> **Runs with NO API key.** In demo/mock mode the pipeline works end-to-end
> (chunk → embed → retrieve → answer with citations) using a local mock
> embedding + extractive answering. Add an OpenAI or Azure OpenAI key and the
> same code calls a real LLM instead — no code change needed.

## Demo

Screenshot of the working app (mock mode, no API key) — asked "What AI work
did Teja do at Technobot?" and the answer cites `technobot.md` and
`about-teja.md`:

![RAG chatbot answering a question about Technobot, with citations](docs/screenshot-chat.png)

## Folder structure

```
krishnateja-rag-chatbot/
├── backend/
│   ├── main.py        # FastAPI app: /health, /chat, /upload, /documents, /
│   ├── rag.py         # The RAG pipeline: chunking, embeddings, vector search, answering
│   └── eval.py        # Eval script: 5 questions, checks keywords + citations
├── frontend/
│   └── index.html     # Single-page chat UI (served by FastAPI at /)
├── data/
│   └── sample_docs/   # Sample documents — Teja's own public facts only
│       ├── about-teja.md
│       ├── technobot.md
│       └── elite-tech.md
├── requirements.txt
├── .env.example       # Copy to .env — never commit real keys
└── README.md
```

## Run it (5 minutes)

```bash
cd krishnateja-rag-chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Demo mode (no key needed):
python -m uvicorn backend.main:app --reload --port 8000

# Then open http://localhost:8000
```

Try these questions in the UI (or with curl):
- "Where does Teja study for his PhD?"
- "What AI work did Teja do at Technobot?"
- "What did Teja build at Elite Tech Info?"

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What AI work did Teja do at Technobot?"}'
```

## Run the eval

```bash
python backend/eval.py
# Expected in demo mode: 5/5 passed. Writes eval_results.json.
```

## Use a real LLM (optional)

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=...  (or the AZURE_OPENAI_* values)
# Restart the server. /health will now show mode: "openai" (or "azure").
```

Never put a key in the code, in git, or in a screenshot. `.env` is git-ignored.

## What each part teaches — read the code in this order

### 1. Python + FastAPI — `backend/main.py`
A REST API with Pydantic models (`ChatRequest` / `ChatResponse`), so bad
input is rejected with a clear error instead of crashing. Endpoints are
small and single-purpose: `/health`, `/chat`, `/upload`, `/documents`.
**Learn:** how a chat product's backend is structured, and how the frontend
talks to it over HTTP/JSON.

### 2. Chunking — `backend/rag.py: chunk_text()`
Documents are split into ~500-character chunks with 50 characters of
overlap, using LangChain's `RecursiveCharacterTextSplitter` (with a plain-
Python fallback). Overlap matters: a fact split across a boundary is lost.
**Learn:** why RAG never sends a whole document to the LLM, and how chunk
size / overlap change retrieval quality. Try changing `chunk_size` to 200
and to 2000, re-run the eval, and see what happens.

### 3. Embeddings — `backend/rag.py: Embedder`
Each chunk becomes a vector. Real mode uses OpenAI / Azure OpenAI
embeddings. Demo mode uses a deterministic hash-based mock embedding —
explicitly labelled as *not* a real semantic embedding — so the pipeline
can be tested for free. In demo mode retrieval also re-ranks by keyword
overlap, so answers stay grounded.
**Learn:** what an embedding is for (similarity search), and why the
embedding model and the LLM are two separate calls with two separate costs.

### 4. Vector search — `backend/rag.py: VectorStore`
ChromaDB stores chunks + vectors persistently in `data/chroma_db/`, so the
index survives restarts. If ChromaDB is unavailable, an in-memory store
with cosine similarity is used instead — same interface, same citations.
**Learn:** how retrieval works (query → vector → nearest chunks), and how
to swap a vector store without rewriting the rest of the app.

### 5. RAG prompting — `backend/rag.py: RAGChatbot.ask()`
Retrieve top-3 chunks, then prompt the LLM: *"Answer using ONLY the
context below. If the context does not contain the answer, say 'I don't
know based on the documents provided.' Cite the source file for each fact."*
**Learn:** grounding is a prompt discipline, not a model property. The
same LLM hallucinates without context and stays honest with it.

### 6. Citations — `backend/rag.py: Answer.sources` + `frontend/index.html`
Every answer returns its source chunks, and the UI shows them as
expandable source cards. An answer without a citation is treated as a
failure in the eval.
**Learn:** citations are a product feature (trust), not a decoration.
Recruiters and users both ask "where did that come from?"

### 7. Evals — `backend/eval.py`
5 questions, each checked for (a) an expected keyword and (b) a citation
in the answer text. Run it after every change. A fluent but uncited answer
fails — that is the hallucination failure mode RAG exists to prevent.
**Learn:** evals are how AI engineers work: test the system, don't just
demo it. Next step: add your own questions and expected keywords.

## Stretch exercises (do these next)

1. **PDF upload:** add `pypdf` and extract text from uploaded PDFs in
   `/upload` (currently .txt/.md only).
2. **Better retrieval:** try `top_k=5`, try a smaller chunk size, and
   compare eval scores. Write down what changed and why.
3. **Real LLM eval:** set an OpenAI key, run `backend/eval.py`, and
   compare the answers with demo mode. Which answers got better? Which
   got worse?
4. **Streaming:** make `/chat` stream tokens (FastAPI `StreamingResponse`)
   so the UI shows the answer as it is generated.
5. **Azure:** set the `AZURE_OPENAI_*` variables in `.env` and run the
   same eval against Azure OpenAI + Azure AI Search concepts from the
   60-day plan.

## What this project proves (for recruiters)

- Python + FastAPI backend, clean REST API
- LangChain chunking, OpenAI/Azure embeddings, ChromaDB vector search
- RAG prompting with citations — grounded answers, not hallucinations
- Evals — a test script that proves the system works
- A chat UI with source display and document upload

## Facts used in the sample docs

Only Teja's own public facts: PhD in AI at University of the Cumberlands
(classes started May 2026, in progress); Technobot LLC, May 2024–Apr 2026,
AI forecasting / vendor selection with OpenAI GPT + LangChain, React +
.NET Core inventory app, Azure AD, real-time APIs, CI/CD; Elite Tech Info
Inc, Oct 2023–May 2024, ServiceNow portals / IT asset workflows, REST
APIs, SCCM/JAMF. No metrics, clients, or projects were invented.
