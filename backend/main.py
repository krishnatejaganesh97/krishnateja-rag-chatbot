"""FastAPI backend for the Teja RAG chatbot.

Endpoints (what Teja learns: REST API design with FastAPI + Pydantic):
  GET  /health          — is the service up, which mode, how many chunks indexed?
  POST /chat            — {question} -> {answer, sources, mode}
  POST /upload          — upload a .txt/.md file, chunk + index it, then ask about it
  GET  /documents       — list the indexed sample documents
  GET  /                — serves the frontend chat UI (frontend/index.html)

Run from the project root:
  python -m uvicorn backend.main:app --reload --port 8000
  then open http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env (if present) BEFORE rag.py reads env vars.
load_dotenv()

from backend.rag import chunk_text, get_bot  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Teja RAG Chatbot",
    description="RAG doc chatbot: FastAPI + LangChain + ChromaDB. Demo mode runs without an API key.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="The question to answer from the documents.")
    top_k: int = Field(3, ge=1, le=5, description="How many chunks to retrieve.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str  # "openai" | "azure" | "mock"
    model: str


@app.get("/health")
def health():
    bot = get_bot()
    return {
        "status": "ok",
        "mode": bot.mode,  # mock = no key, runs locally; openai/azure = live LLM
        "vector_store": bot.store.backend,
        "chunks_indexed": len(bot.store.chunks),
        "docs": sorted({c.source for c in bot.store.chunks}),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bot = get_bot()
    result = bot.ask(req.question, k=req.top_k)
    return ChatResponse(
        answer=result.text, sources=result.sources, mode=result.mode, model=result.model
    )


@app.get("/documents")
def list_documents():
    bot = get_bot()
    return {"documents": sorted({c.source for c in bot.store.chunks})}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a .txt or .md doc, index it, and ask about it in /chat.

    What this teaches: file upload in FastAPI, then the same chunk -> embed
    -> index pipeline as the sample docs. PDF upload is left as a stretch
    exercise (hint: pypdf, see README).
    """
    if not file.filename or not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(400, "Please upload a .txt or .md file.")
    content = (await file.read()).decode("utf-8", errors="ignore")
    dest = UPLOAD_DIR / Path(file.filename).name
    dest.write_text(content, encoding="utf-8")
    bot = get_bot()
    new_chunks = chunk_text(content, source=dest.name)
    bot.store.add(bot.store.chunks + new_chunks)
    return {"filename": dest.name, "chunks_added": len(new_chunks),
            "total_chunks": len(bot.store.chunks)}


# --- Frontend: serve the single-page chat UI at / ---
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
