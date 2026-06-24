"""Streamlit chat UI for the NeuralNav customer service bot.

Run: streamlit run frontend/app.py
Expects the FastAPI backend reachable at BACKEND_URL (env var, default localhost).
"""
import os
import time

import requests
import streamlit as st

from avatars import ASSISTANT_AVATAR, USER_AVATAR

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
CONFIDENCE_THRESHOLD = 0.45  # mirrors backend/main.py — shown in the UI so escalations make sense

# Mirrors the 27 intents the classifier was actually trained on (Bitext
# Customer Support dataset) — questions outside these topics will get a
# low-confidence guess and likely escalate, which is expected, not a bug.
EXAMPLE_QUESTIONS = {
    "Orders": [
        "I want to cancel my order",
        "Can you change the item in my order?",
        "Where is my order right now?",
        "I'd like to place a new order",
    ],
    "Refunds & cancellations": [
        "What's your refund policy?",
        "When will I get my refund?",
        "How much is the cancellation fee?",
    ],
    "Account": [
        "How do I create a new account?",
        "I forgot my password",
        "Please delete my account",
        "I'm having trouble registering",
    ],
    "Shipping & delivery": [
        "Can you update my shipping address?",
        "How long does delivery take?",
        "What delivery options do you have?",
    ],
    "Billing": [
        "Can you send me my invoice?",
        "What payment methods do you accept?",
        "I was charged twice for my order",
    ],
    "Support": [
        "I want to talk to a human agent",
        "I have a complaint about my order",
        "How do I contact customer service?",
    ],
}

st.set_page_config(page_title="NeuralNav Support", page_icon="◆", layout="centered")

CSS = """
<style>
:root {
    --bg: #F5F4EF;
    --bg-card: #FAF9F5;
    --border: #E5E1D8;
    --text: #2D2A26;
    --text-muted: #78746C;
    --accent: #D97757;
    --accent-soft: #FBEAE2;
}

.block-container { padding-top: 2rem; max-width: 760px; }

.nn-header {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 1rem 1.3rem; border-radius: 14px; margin-bottom: 1.3rem;
    background: var(--bg-card); border: 1px solid var(--border);
}
.nn-header .mark {
    width: 34px; height: 34px; border-radius: 50%; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.95rem; flex-shrink: 0;
}
.nn-header .title { font-size: 1.05rem; font-weight: 600; margin: 0; color: var(--text); }
.nn-header .subtitle { font-size: 0.8rem; margin: 0; color: var(--text-muted); }
.nn-header .badge {
    margin-left: auto; font-size: 0.72rem; color: #1A7F4B;
    display: flex; align-items: center; gap: 0.35rem;
}
.nn-header .dot { width: 7px; height: 7px; border-radius: 50%; background: #1A7F4B; }

.nn-meta-row { display: flex; gap: 0.4rem; margin: 0.5rem 0 0.3rem 0; flex-wrap: wrap; align-items: center; }
.nn-chip {
    font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 999px;
    font-weight: 500; border: 1px solid var(--border); color: var(--text-muted);
    background: var(--bg-card);
}
.nn-chip-conf-high { color: #1A7F4B; border-color: #BEE3CE; background: #EDF8F1; }
.nn-chip-conf-med { color: #925C13; border-color: #F2DCA8; background: #FBF3DF; }
.nn-chip-conf-low { color: #B3261E; border-color: #F3C6C2; background: #FCEEEC; }

.nn-note { font-size: 0.72rem; color: var(--text-muted); margin: 0.1rem 0 0.4rem 0; }

.nn-escalate {
    background: var(--accent-soft); border: 1px solid #F0C4AC; border-radius: 12px;
    padding: 0.75rem 1rem; font-size: 0.92rem; color: #8A3F1F;
}

.nn-source {
    font-size: 0.8rem; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.55rem 0.8rem; margin-top: 0.35rem; color: var(--text);
}
.nn-source b { color: var(--text); }
.nn-source i { color: var(--text-muted); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="nn-header">
        <div class="mark">N</div>
        <div>
            <p class="title">NeuralNav Support</p>
            <p class="subtitle">ML/DL-powered customer service assistant</p>
        </div>
        <div class="badge"><span class="dot"></span> Online</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?", "meta": None, "message_id": None}
    ]

with st.sidebar:
    st.subheader("Session")
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you today?", "meta": None, "message_id": None}
        ]
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id or 'not started'}`")
    st.divider()

    with st.expander("💬 What can I ask?", expanded=False):
        st.caption(
            "The assistant is trained on e-commerce/account support topics. "
            "Try things like:"
        )
        for category, examples in EXAMPLE_QUESTIONS.items():
            st.markdown(f"**{category}**")
            for ex in examples:
                st.caption(f"· {ex}")

    st.divider()
    st.caption("**How this works**")
    st.caption(
        "Each message is classified into an intent (TF-IDF baseline or "
        "fine-tuned DistilBERT), checked for entities (order IDs, error "
        "codes), then either answered via semantic retrieval over the "
        f"knowledge base or escalated if confidence drops below "
        f"{int(CONFIDENCE_THRESHOLD * 100)}%."
    )
    st.divider()
    st.caption("Backend: " + BACKEND_URL)
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2)
        st.caption("🟢 Backend reachable" if health.ok else "🔴 Backend error")
    except requests.RequestException:
        st.caption("🔴 Backend unreachable")


def confidence_chip(confidence: float) -> str:
    pct = round(confidence * 100)
    if confidence >= 0.7:
        cls = "nn-chip-conf-high"
    elif confidence >= CONFIDENCE_THRESHOLD:
        cls = "nn-chip-conf-med"
    else:
        cls = "nn-chip-conf-low"
    return f'<span class="nn-chip {cls}">confidence {pct}%</span>'


def render_meta(meta: dict):
    if not meta:
        return
    intent_chip = f'<span class="nn-chip">intent: {meta["intent"]}</span>'
    st.markdown(
        f'<div class="nn-meta-row">{intent_chip}{confidence_chip(meta["confidence"])}</div>',
        unsafe_allow_html=True,
    )
    if meta.get("escalated"):
        st.markdown(
            f'<p class="nn-note">Confidence was below {int(CONFIDENCE_THRESHOLD * 100)}%, '
            f'so this was routed to a human agent instead of an automated answer.</p>',
            unsafe_allow_html=True,
        )
    if meta.get("sources"):
        with st.expander(f"{len(meta['sources'])} source(s) used"):
            for s in meta["sources"]:
                st.markdown(
                    f'<div class="nn-source"><b>{s["question"]}</b><br>{s["answer"]}'
                    f'<br><i>match score: {s["score"]:.2f}</i></div>',
                    unsafe_allow_html=True,
                )


def render_feedback(message_id, idx):
    if message_id is None:
        return
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"up_{idx}"):
            try:
                requests.post(f"{BACKEND_URL}/feedback", json={"message_id": message_id, "rating": "up"}, timeout=5)
                st.toast("Thanks for the feedback!")
            except requests.RequestException:
                pass
    with col2:
        if st.button("👎", key=f"down_{idx}"):
            try:
                requests.post(f"{BACKEND_URL}/feedback", json={"message_id": message_id, "rating": "down"}, timeout=5)
                st.toast("Thanks — we'll use this to improve.")
            except requests.RequestException:
                pass


for idx, msg in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        meta = msg.get("meta")
        if meta and meta.get("escalated"):
            st.markdown(f'<div class="nn-escalate">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(msg["content"])
        render_meta(meta)
        if msg["role"] == "assistant":
            render_feedback(msg.get("message_id"), idx)

if prompt := st.chat_input("Describe your issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None, "message_id": None})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown("_typing..._")
        try:
            history_payload = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
            ]
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "history": history_payload,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            time.sleep(0.2)
            placeholder.empty()

            meta = {
                "intent": data["intent"],
                "confidence": data["confidence"],
                "escalated": data["escalated"],
                "sources": data["sources"],
            }
            if data["escalated"]:
                st.markdown(f'<div class="nn-escalate">{data["reply"]}</div>', unsafe_allow_html=True)
            else:
                st.write(data["reply"])
            render_meta(meta)
            render_feedback(data["message_id"], len(st.session_state.messages))

            st.session_state.messages.append(
                {"role": "assistant", "content": data["reply"], "meta": meta, "message_id": data["message_id"]}
            )
        except requests.RequestException as e:
            placeholder.empty()
            err = f"Backend unreachable: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "meta": None, "message_id": None})
