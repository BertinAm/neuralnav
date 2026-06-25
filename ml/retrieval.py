"""Semantic retrieval over the knowledge base.

Two backends behind the same KBRetriever interface:
  - Full: sentence-transformers embeddings + FAISS (what the report's
    actual hit-rate@k numbers in notebooks/02 are based on).
  - Lightweight: TF-IDF + cosine similarity, scikit-learn only — used
    automatically under ml/resource_mode.LIGHTWEIGHT_MODE (e.g. Render's
    512MB free tier) where loading sentence-transformers/torch alongside
    the rest of the pipeline gets OOM-killed. Lower retrieval quality, but
    keeps the live demo functional without a paid hosting tier.

Prefers the real kb.json pulled from the Hugging Face Hub repo that
notebooks/01_intent_classification.ipynb pushes to (downloaded into
models/kb.json — see ml/hf_hub.py) over the small dev-fallback sample in
data/kb.json, so the backend automatically uses real data once it's
available without code changes.
"""
import json
from pathlib import Path

import numpy as np

from ml.hf_hub import ensure_file
from ml.resource_mode import LIGHTWEIGHT_MODE

ROOT = Path(__file__).resolve().parent.parent
HUB_KB_PATH = ROOT / "models" / "kb.json"
DEV_KB_PATH = ROOT / "data" / "kb.json"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _resolve_kb_path() -> Path:
    return ensure_file(HUB_KB_PATH, "kb.json") or DEV_KB_PATH


class KBRetriever:
    def __init__(self, kb_path: Path | None = None):
        kb_path = kb_path or _resolve_kb_path()
        self.kb = json.loads(kb_path.read_text())
        texts = [f"{item['question']} {item['answer']}" for item in self.kb]

        if LIGHTWEIGHT_MODE:
            self._init_tfidf(texts)
        else:
            self._init_embeddings(texts)

    def _init_embeddings(self, texts: list[str]):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.backend = "embeddings"
        self.model = SentenceTransformer(EMBED_MODEL)
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype=np.float32))

    def _init_tfidf(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.backend = "tfidf"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.backend == "tfidf":
            return self._search_tfidf(query, top_k)
        return self._search_embeddings(query, top_k)

    def _search_embeddings(self, query: str, top_k: int) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.asarray(q_emb, dtype=np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.kb[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def _search_tfidf(self, query: str, top_k: int) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                continue
            item = dict(self.kb[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results


if __name__ == "__main__":
    retriever = KBRetriever()
    for r in retriever.search("my wifi disconnects randomly"):
        print(r["score"], r["question"], "->", r["answer"])
