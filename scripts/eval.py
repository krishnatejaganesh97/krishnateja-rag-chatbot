"""Run the RAG chatbot eval (preferred entry point).

    python scripts/eval.py

This is a thin wrapper around backend/eval.py, which loads the eval set
from eval/eval_set.json (8 questions, each with a known correct answer and
an expected source document) and checks accuracy (all expected keywords
in the answer) plus citation correctness (the expected source is returned
and cited in the answer text). Exit code 0 = all passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
