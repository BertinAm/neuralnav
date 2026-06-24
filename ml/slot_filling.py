"""Minimal slot-filling so the bot doesn't repeat the same canned question
forever when a user actually answers it.

This is intentionally narrow (one slot: a shipping-address update) rather
than a general dialogue manager — a real fix is RASA's dialogue policy
(see implementation.md §6). This patches the most visible failure mode:
the KB-retrieval-only pipeline has no concept of "I already asked this,"
so it kept re-asking for the same address after the user had just given it.
"""
import re

# Intents whose KB answer is a question expecting a follow-up — once the
# bot sends that answer, the next user turn should be treated as the
# answer to it, not reclassified from scratch.
SLOT_PROMPTING_INTENTS = {"change_shipping_address": "shipping_address"}

_ADDRESS_PAIR_RE = re.compile(
    r"current address is\s*(?P<current>.+?)\s*(?:,|\.|;)?\s*(?:and\s+)?(?:my\s+)?"
    r"new address is\s*(?P<new>.+)",
    re.IGNORECASE,
)


def _clean(raw: str) -> str:
    return raw.strip(" \"'.,;")


def parse_address_update(text: str) -> tuple[str, str] | None:
    """Return (current_address, new_address) if text matches the
    'current address is X ... new address is Y' pattern, else None.

    Uses a non-restrictive `.+?` for the captured spans (anchored on the
    boundary phrase, not a character class) so addresses containing quotes
    or punctuation aren't truncated, then strips quotes/punctuation after.
    """
    match = _ADDRESS_PAIR_RE.search(text)
    if not match:
        return None
    return _clean(match.group("current")), _clean(match.group("new"))


def looks_like_a_question(text: str) -> bool:
    """If the user's next message looks like a new question rather than
    an answer, don't force it into the pending slot — let it reclassify
    normally instead of getting stuck."""
    return "?" in text
