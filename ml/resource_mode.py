"""Whether to run the cheap, low-memory pipeline (TF-IDF classifier + TF-IDF
retrieval + regex-only NER) instead of the full DL stack (DistilBERT +
sentence-transformers + spaCy).

Render's free web service tier caps memory at 512MB; torch + a loaded
DistilBERT model + sentence-transformers' MiniLM + spaCy's en_core_web_sm
together comfortably exceed that and get OOM-killed. Render sets RENDER=true
on its containers automatically, so this auto-detects there with no config
needed, while local Docker / Kaggle keep using the full DL stack that the
report's actual results are based on. Override explicitly with
LIGHTWEIGHT_MODE=true/false if needed (e.g. testing this path locally, or
running on a paid Render plan with enough RAM for the full stack).
"""
import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


LIGHTWEIGHT_MODE = _flag("LIGHTWEIGHT_MODE") or (
    "LIGHTWEIGHT_MODE" not in os.environ and _flag("RENDER")
)
