# memory.py
import os, json, time, logging
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from config import VECTOR_DB_PATH, SQL_DB_PATH, DATA_DIR
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

logger = logging.getLogger("superbrain.memory")
os.makedirs(DATA_DIR, exist_ok=True)

_EMB = None
def embedder():
    global _EMB
    if _EMB is None:
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB

class PersistentMemory:
    def __init__(self, dim=384):
        self.dim = dim
        self.texts = []
        self.metas = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None

    def add(self, texts: List[str], metas: List[Dict]=None):
        metas = metas or [{} for _ in texts]
        embs = embedder().encode(texts, convert_to_numpy=True)
        if self.index is not None:
            self.index.add(embs)
        self.texts.extend(texts)
        self.metas.extend(metas)
        logger.debug("Memory added %d entries", len(texts))

    def search(self, query: str, k: int=5) -> List[Tuple[str, Dict, float]]:
        if not self.texts:
            return []
        qv = embedder().encode([query], convert_to_numpy=True)
        if self.index is not None and self.index.ntotal > 0:
            D,I = self.index.search(qv, k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                results.append((self.texts[idx], self.metas[idx], float(dist)))
            return results
        # fallback linear
        embs = embedder().encode(self.texts, convert_to_numpy=True)
        dists = np.linalg.norm(embs - qv, axis=1)
        ids = np.argsort(dists)[:k]
        return [(self.texts[int(i)], self.metas[int(i)], float(dists[int(i)])) for i in ids]

    def persist_json(self, path=os.path.join(DATA_DIR, "memory.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas}, f, ensure_ascii=False, indent=2)
