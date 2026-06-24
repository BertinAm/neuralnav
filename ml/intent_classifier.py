"""Inference-time wrapper around the trained intent classifier.

Defaults to the BERT model if it was trained (models/bert_intent/ exists),
falling back to the TF-IDF baseline otherwise — lets the backend run even
if the heavier DL training step hasn't been done yet.
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "baseline_intent.joblib"
BERT_DIR = ROOT / "models" / "bert_intent"


class IntentClassifier:
    def __init__(self):
        self.backend = None
        if BERT_DIR.exists():
            self._load_bert()
        elif BASELINE_PATH.exists():
            self._load_baseline()
        else:
            raise FileNotFoundError(
                "No trained model found. Run ml/train_baseline.py (and optionally ml/train_bert.py) first."
            )

    def _load_baseline(self):
        import joblib
        self.pipeline = joblib.load(BASELINE_PATH)
        self.backend = "baseline"

    def _load_bert(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
        self.model.eval()
        self.labels = json.loads((BERT_DIR / "label_classes.json").read_text())
        self.torch = torch
        self.backend = "bert"

    def predict(self, text: str) -> dict:
        if self.backend == "baseline":
            intent = self.pipeline.predict([text])[0]
            proba = max(self.pipeline.predict_proba([text])[0])
            return {"intent": intent, "confidence": float(proba), "backend": "baseline"}

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        idx = int(self.torch.argmax(probs))
        return {
            "intent": self.labels[idx],
            "confidence": float(probs[idx]),
            "backend": "bert",
        }


if __name__ == "__main__":
    clf = IntentClassifier()
    print(clf.predict("my device won't turn on"))
