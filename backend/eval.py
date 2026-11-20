"""Eval script for the Teja RAG chatbot.

What this teaches: RAG evals — don't trust a demo, test it.
The eval set lives in eval/eval_set.json: 8 questions, each with a known
correct answer and an expected source document, taken verbatim from the
sample docs in data/sample_docs/. Each question is checked on two axes:

  1. Accuracy — does the answer contain EVERY expected keyword?
     (A partial answer is a fail: "SQL Server" without "MongoDB" is wrong.)
  2. Citation correctness — is the EXPECTED source both returned in
     `sources` AND cited in the answer text (as [file.md])? Citing the
     wrong document fails, even if the keywords are right — that is the
     hallucination / misattribution failure mode RAG exists to prevent.
     The old eval only checked that *some* citation existed, so a PhD
     answer could "pass" while citing elite-tech.md.

Run from the project root (either entry point works):
  python scripts/eval.py     # preferred
  python backend/eval.py
Exit code 0 = all questions passed, 1 = at least one failed.
Writes eval_results.json at the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag import CHUNK_OVERLAP, CHUNK_SIZE, get_bot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = ROOT / "eval" / "eval_set.json"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def main() -> int:
    bot = get_bot()
    questions = load_eval_set()
    print(f"Mode: {bot.mode} | Vector store: {bot.store.backend} | "
          f"Chunks indexed: {len(bot.store.chunks)} | "
          f"Chunking: {CHUNK_SIZE}/{CHUNK_OVERLAP}\n")
    results = []
    passed = 0
    citations_correct = 0
    for i, q in enumerate(questions, 1):
        question = q["question"]
        keywords = q["keywords"]
        expected_source = q["expected_source"]
        answer = bot.ask(question, k=3)
        found = [k for k in keywords if k.lower() in answer.text.lower()]
        keyword_accuracy = len(found) / len(keywords) if keywords else 1.0
        sources_returned = [s["source"] for s in answer.sources]
        cites_in_text = f"[{expected_source}]" in answer.text
        source_returned = expected_source in sources_returned
        citation_ok = cites_in_text and source_returned
        citations_correct += citation_ok
        ok = keyword_accuracy == 1.0 and citation_ok
        passed += ok
        results.append({
            "id": q.get("id", f"q{i}"),
            "question": question,
            "expected_answer": q["expected_answer"],
            "expected_source": expected_source,
            "expected_keywords": keywords,
            "keywords_found": found,
            "keyword_accuracy": round(keyword_accuracy, 2),
            "sources_returned": sources_returned,
            "cites_expected_source_in_text": cites_in_text,
            "expected_source_returned": source_returned,
            "citation_correct": citation_ok,
            "pass": ok,
            "answer": answer.text,
        })
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] Q{i} ({q.get('id', '')}): {question}")
        print(f"       expected answer: {q['expected_answer']} [{expected_source}]")
        print(f"       keywords {len(found)}/{len(keywords)}: "
              f"{'all found' if keyword_accuracy == 1.0 else 'MISSING ' + str([k for k in keywords if k not in found])}")
        print(f"       citation: expected [{expected_source}] in text: "
              f"{'yes' if cites_in_text else 'NO'} | returned as source: "
              f"{'yes' if source_returned else 'NO'} | sources: {sources_returned}")
        print(f"       answer: {answer.text[:300]}\n")

    accuracy = passed / len(questions) if questions else 0.0
    citation_rate = citations_correct / len(questions) if questions else 0.0
    print(f"Result: {passed}/{len(questions)} passed "
          f"| accuracy {accuracy:.0%} | citation correctness {citation_rate:.0%}")
    out = ROOT / "eval_results.json"
    out.write_text(json.dumps({
        "mode": bot.mode,
        "vector_store": bot.store.backend,
        "chunks_indexed": len(bot.store.chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "passed": passed,
        "total": len(questions),
        "accuracy": round(accuracy, 2),
        "citation_correctness": round(citation_rate, 2),
        "results": results,
    }, indent=2))
    print(f"Wrote {out}")
    return 0 if passed == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
