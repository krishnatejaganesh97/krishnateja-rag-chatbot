"""Eval script for the Teja RAG chatbot.

What this teaches: RAG evals — don't trust a demo, test it.
For each of 5 sample questions we check two things:
  1. Does the answer contain an expected keyword (is it on-topic)?
  2. Does it cite at least one source file (is it grounded)?

An answer that is fluent but uncited FAILS — that is the exact failure
mode RAG is supposed to prevent (hallucination).

Run from the project root:
  python backend/eval.py
Exit code 0 = all 5 passed, 1 = at least one failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag import get_bot  # noqa: E402

# (question, expected keyword in the answer, lowercased)
TESTS = [
    ("Where does Teja study for his PhD?", "cumberlands"),
    ("What AI work did Teja do at Technobot?", "langchain"),
    ("What did Teja build at Elite Tech Info?", "servicenow"),
    ("What programming languages and frameworks does Teja use?", "python"),
    ("How many years of experience does Teja have?", "4"),
]


def main() -> int:
    bot = get_bot()
    print(f"Mode: {bot.mode} | Vector store: {bot.store.backend} | "
          f"Chunks indexed: {len(bot.store.chunks)}\n")
    results = []
    passed = 0
    for i, (question, keyword) in enumerate(TESTS, 1):
        answer = bot.ask(question, k=3)
        has_keyword = keyword.lower() in answer.text.lower()
        has_citation = len(answer.sources) > 0 and any(
            s["source"] in answer.text or True for s in answer.sources
        )
        # Citation check that matters: sources must be returned AND the
        # answer must name at least one of them (mock mode embeds [file] in text).
        cites_in_text = any(s["source"] in answer.text for s in answer.sources)
        ok = has_keyword and cites_in_text
        passed += ok
        results.append({
            "question": question, "expected_keyword": keyword,
            "has_keyword": has_keyword, "sources_returned": [s["source"] for s in answer.sources],
            "cites_source_in_text": cites_in_text, "pass": ok,
            "answer": answer.text,
        })
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] Q{i}: {question}")
        print(f"       keyword '{keyword}': {'yes' if has_keyword else 'NO'} | "
              f"cited in text: {'yes' if cites_in_text else 'NO'}")
        print(f"       sources: {[s['source'] for s in answer.sources]}")
        print(f"       answer: {answer.text[:300]}\n")

    print(f"Result: {passed}/{len(TESTS)} passed")
    out = Path(__file__).resolve().parent.parent / "eval_results.json"
    out.write_text(json.dumps({"mode": bot.mode, "passed": passed,
                               "total": len(TESTS), "results": results}, indent=2))
    print(f"Wrote {out}")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
