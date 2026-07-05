from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_oc.build_faiss_index import *  # noqa: F403
from rag_oc.build_faiss_index import main


if __name__ == "__main__":
    main()
