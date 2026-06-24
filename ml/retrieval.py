"""Semantic retrieval over the knowledge base using sentence-transformers
embeddings + FAISS for nearest-neighbor search.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "kb.json"
EMBED_MODEL = "all-MiniLM-L6-v2"


class KBRetriever:
    def __init__(self, kb_path: Path = KB_PATH):
        self.kb = json.loads(kb_path.read_text())
        self.model = SentenceTransformer(EMBED_MODEL)
        texts = [f"{item['question']} {item['answer']}" for item in self.kb]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype=np.float32))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
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


if __name__ == "__main__":
    retriever = KBRetriever()
    for r in retriever.search("my wifi disconnects randomly"):
        print(r["score"], r["question"], "->", r["answer"])
