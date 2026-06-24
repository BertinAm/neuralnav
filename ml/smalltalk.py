"""Lightweight rule-based detector for greetings/goodbyes/identity questions.

The Bitext dataset the intent classifier is trained on has no chitchat
intent at all (its 27 intents are all task-specific: cancel_order,
place_order, etc.) — so a plain "Hello" gets force-fit into some random
low-confidence task intent and escalates to a human agent, which is wrong.
This catches small talk before it ever reaches the classifier.
"""
import re

_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.IGNORECASE)
_GOODBYE_RE = re.compile(r"\b(bye|goodbye|see you|that'?s all|thanks?(,? (a lot|so much))?|thank you)\b", re.IGNORECASE)
_IDENTITY_RE = re.compile(
    r"\b(who are you|who is this|what are you|what(?:'s| is) your (name|role)|"
    r"are you (a )?(bot|human|ai|chatbot))\b",
    re.IGNORECASE,
)
_CAPABILITY_RE = re.compile(
    r"\b(what can you do|how can you help|what do you do|help me with what)\b",
    re.IGNORECASE,
)

RESPONSES = {
    "greeting": "Hi! I'm the support assistant. What can I help you with today?",
    "goodbye": "You're welcome! Reach out anytime you need help.",
    "identity": "I'm NeuralNav, an automated support assistant — I can help with orders, "
                 "accounts, refunds, and similar requests, and connect you to a human agent if needed.",
    "capability": "I can help with things like order status, cancellations, refunds, account "
                   "issues, and shipping questions. What do you need help with?",
}


def detect(text: str) -> str | None:
    """Return a smalltalk category name if text matches, else None."""
    if _IDENTITY_RE.search(text):
        return "identity"
    if _CAPABILITY_RE.search(text):
        return "capability"
    if _GREETING_RE.search(text):
        return "greeting"
    if _GOODBYE_RE.search(text):
        return "goodbye"
    return None
