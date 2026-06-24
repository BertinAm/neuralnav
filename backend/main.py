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
from ml.smalltalk import RESPONSES, detect as detect_smalltalk

CONFIDENCE_THRESHOLD = 0.45
ESCALATE_INTENTS = {"escalate_human"}
HISTORY_CONTEXT_TURNS = 2

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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    entities = extract_entities(req.message)
    smalltalk_category = detect_smalltalk(req.message)

    if smalltalk_category:
        intent = f"smalltalk_{smalltalk_category}"
        confidence = 1.0
        escalated = False
        reply = RESPONSES[smalltalk_category]
        sources = []
    else:
        recent_user_turns = [t.content for t in req.history if t.role == "user"][-HISTORY_CONTEXT_TURNS:]
        context_text = " ".join(recent_user_turns + [req.message])

        intent_result = _classifier.predict(context_text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        escalated = False
        reply = ""

        if intent in ESCALATE_INTENTS or confidence < CONFIDENCE_THRESHOLD:
            escalated = True
            reply = "I'll connect you with a human agent who can help further. One moment please."
            sources = []
        else:
            sources = _retriever.search(req.message, top_k=3)
            if not sources:
                reply = "I'm not sure yet — could you rephrase, or would you like a human agent?"
            else:
                reply = sources[0]["answer"]
                if entities["order_ids"]:
                    reply += f" (Reference order: {entities['order_ids'][0]})"

    db.log_message(session_id, "user", req.message, intent=intent, confidence=confidence, escalated=escalated)
    message_id = db.log_message(session_id, "assistant", reply, intent=intent, confidence=confidence, escalated=escalated)

    return ChatResponse(
        message_id=message_id,
        session_id=session_id,
        reply=reply,
        intent=intent,
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
