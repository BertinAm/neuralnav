"""Entity extraction: order IDs and error codes via regex (deterministic,
high precision for structured tokens), product/date entities via spaCy's
statistical NER (handles free-text variation regex can't).

spaCy's model is skipped under ml/resource_mode.LIGHTWEIGHT_MODE (e.g.
Render's 512MB free tier) — regex-only extraction still works, just without
the free-text entity types spaCy would catch.
"""
import re
from typing import TypedDict

from ml.resource_mode import LIGHTWEIGHT_MODE

_nlp = None

ORDER_ID_RE = re.compile(r"\b(?:order\s*#?\s*)?([A-Z]{0,3}\d{5,10})\b", re.IGNORECASE)
ERROR_CODE_RE = re.compile(r"\b(E\d{3}|0x[0-9A-Fa-f]{2,4})\b")


class Entities(TypedDict):
    order_ids: list[str]
    error_codes: list[str]
    spacy_entities: list[dict]


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _nlp


def extract_entities(text: str) -> Entities:
    order_ids = [m.group(1) for m in ORDER_ID_RE.finditer(text)]
    error_codes = ERROR_CODE_RE.findall(text)

    spacy_entities = []
    if not LIGHTWEIGHT_MODE:
        doc = _get_nlp()(text)
        spacy_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {
        "order_ids": order_ids,
        "error_codes": error_codes,
        "spacy_entities": spacy_entities,
    }


if __name__ == "__main__":
    print(extract_entities("I'm getting error code E101 on order 4471829"))
