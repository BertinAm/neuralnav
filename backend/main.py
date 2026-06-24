"""FastAPI backend for the customer service chatbot.

Pipeline per request:
  1. Intent classification (BERT if trained, else TF-IDF baseline) using the
     current message plus recent user turns as context for disambiguation
  2. Entity extraction (order IDs, error codes, spaCy NER)
  3. Low confidence or explicit request -> escalate to human
  4. Otherwise -> semantic retrieval over the KB for the best-matching answer
  5. Every turn is logged to SQLite for the admin dashboard / analytics

Run: uvicorn backend.main:app --reload --port 8000
"""
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import db
from ml.intent_classifier import IntentClassifier
from ml.ner import extract_entities
from ml.retrieval import KBRetriever
from ml.slot_filling import SLOT_PROMPTING_INTENTS, looks_like_a_question, parse_address_update
from ml.smalltalk import RESPONSES, detect as detect_smalltalk

CONFIDENCE_THRESHOLD = 0.45
# Retrieval match scores cluster ~0.6-0.85 for genuinely correct matches and
# lower (with overlap) for incorrect ones, per notebooks/02's score
# distribution chart — 0.5 is a heuristic cutoff, not a tuned threshold.
# Escalating on either weak signal (classifier OR retrieval) is more
# conservative than checking classifier confidence alone, which previously
# let weak KB matches answer confidently with no uncertainty signal at all.
RETRIEVAL_SCORE_THRESHOLD = 0.5
ESCALATE_INTENTS = {"escalate_human"}
HISTORY_CONTEXT_TURNS = 2

# In-memory per-session "what are we waiting for" state — e.g. after the
# bot asks for a shipping address, the next turn should be treated as the
# answer instead of being reclassified from scratch. Not persisted (lost on
# restart); fine for this scope, see ml/slot_filling.py for why this exists
# instead of a full dialogue manager.
_pending_slots: dict[str, str] = {}

app = FastAPI(title="NeuralNav Customer Service Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier: IntentClassifier | None = None
_retriever: KBRetriever | None = None


@app.on_event("startup")
def load_models():
    global _classifier, _retriever
    db.init_db()
    _classifier = IntentClassifier()
    _retriever = KBRetriever()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    message_id: int
    session_id: str
    reply: str
    intent: str
    classified_intent: str
    confidence: float
    escalated: bool
    entities: dict
    sources: list[dict]


class FeedbackRequest(BaseModel):
    message_id: int
    rating: str  # "up" or "down"


@app.get("/health")
def health():
    return {"status": "ok"}


def _resolve_pending_slot(session_id: str, message: str) -> dict | None:
    """If this session is waiting on an answer to a slot the bot already
    asked for, and this message looks like that answer, handle it directly
    instead of reclassifying from scratch. Returns None if there's no
    pending slot (or it doesn't apply), so the caller falls through to
    normal classification."""
    pending_slot = _pending_slots.get(session_id)
    if not pending_slot or looks_like_a_question(message):
        return None

    del _pending_slots[session_id]
    if pending_slot != "shipping_address":
        return None

    parsed = parse_address_update(message)
    if parsed:
        current, new = parsed
        reply = (
            f'Got it — I\'ve recorded the change from "{current}" to "{new}". '
            "Our shipping team will update this on your account."
        )
    else:
        reply = (
            f'Got it — I\'ve noted your new shipping details: "{message.strip()}". '
            "Our shipping team will update this on your account."
        )
    return {"intent": "slot_filled_shipping_address", "confidence": 1.0, "escalated": False, "reply": reply, "sources": []}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    entities = extract_entities(req.message)
    slot_result = _resolve_pending_slot(session_id, req.message)
    smalltalk_category = None if slot_result else detect_smalltalk(req.message)

    if slot_result:
        intent, confidence, escalated, reply, sources = (
            slot_result["intent"], slot_result["confidence"], slot_result["escalated"],
            slot_result["reply"], slot_result["sources"],
        )
        classified_intent = intent
    elif smalltalk_category:
        intent = f"smalltalk_{smalltalk_category}"
        classified_intent = intent
        confidence = 1.0
        escalated = False
        reply = RESPONSES[smalltalk_category]
        sources = []
    else:
        recent_user_turns = [t.content for t in req.history if t.role == "user"][-HISTORY_CONTEXT_TURNS:]
        context_text = " ".join(recent_user_turns + [req.message])

        intent_result = _classifier.predict(context_text)
        classified_intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        intent = classified_intent  # may be overridden below by the retrieved entry's intent
        escalated = False
        reply = ""
        sources = []

        if classified_intent in ESCALATE_INTENTS:
            escalated = True
            reply = "I'll connect you with a human agent who can help further. One moment please."
        else:
            sources = _retriever.search(req.message, top_k=3)
            top_score = sources[0]["score"] if sources else 0.0

            if not sources:
                reply = "I'm not sure yet — could you rephrase, or would you like a human agent?"
            elif confidence < CONFIDENCE_THRESHOLD or top_score < RETRIEVAL_SCORE_THRESHOLD:
                # Either signal being weak is reason enough to hand off — a
                # confident-but-irrelevant retrieval match is just as risky
                # as a confident-but-wrong classification.
                escalated = True
                reply = "I'll connect you with a human agent who can help further. One moment please."
                sources = []
            else:
                # The retrieved KB entry's own intent is what's actually
                # answering the question, which can legitimately differ
                # from the classifier's guess — show that, not the
                # classifier's label, so the displayed intent matches the
                # content of the reply.
                intent = sources[0]["intent"]
                reply = sources[0]["answer"]
                if entities["order_ids"]:
                    reply += f" (Reference order: {entities['order_ids'][0]})"

        if intent in SLOT_PROMPTING_INTENTS and not escalated:
            _pending_slots[session_id] = SLOT_PROMPTING_INTENTS[intent]

    # Logged against the classifier's own intent/confidence (not the
    # retrieval-corrected display intent) so dashboard analytics reflect
    # actual model performance, independent of how the UI presents it.
    db.log_message(session_id, "user", req.message, intent=classified_intent, confidence=confidence, escalated=escalated)
    message_id = db.log_message(
        session_id, "assistant", reply, intent=classified_intent, confidence=confidence, escalated=escalated
    )

    return ChatResponse(
        message_id=message_id,
        session_id=session_id,
        reply=reply,
        intent=intent,
        classified_intent=classified_intent,
        confidence=confidence,
        escalated=escalated,
        entities=entities,
        sources=sources,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    db.log_feedback(req.message_id, req.rating)
    return {"status": "ok"}


@app.get("/history/{session_id}")
def history(session_id: str, limit: int = 20):
    return db.get_history(session_id, limit=limit)


@app.get("/stats")
def stats():
    return db.get_stats()
