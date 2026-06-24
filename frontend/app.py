"""Streamlit chat UI for the NeuralNav customer service bot.

Run: streamlit run frontend/app.py
Expects the FastAPI backend reachable at BACKEND_URL (env var, default localhost).
"""
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="NeuralNav Support", page_icon="🤖", layout="centered")

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 760px; }

.nn-header {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #c026d3 100%);
    color: white;
}
.nn-header .title { font-size: 1.3rem; font-weight: 700; margin: 0; }
.nn-header .subtitle { font-size: 0.85rem; opacity: 0.9; margin: 0; }
.nn-header .badge {
    margin-left: auto; font-size: 0.72rem; background: rgba(255,255,255,0.18);
    padding: 0.25rem 0.6rem; border-radius: 999px;
}

.nn-meta-row { display: flex; gap: 0.4rem; margin: 0.35rem 0 0.5rem 0; flex-wrap: wrap; }
.nn-chip {
    font-size: 0.72rem; padding: 0.18rem 0.6rem; border-radius: 999px;
    font-weight: 600; border: 1px solid transparent;
}
.nn-chip-intent { background: #eef2ff; color: #4338ca; }
.nn-chip-conf-high { background: #ecfdf5; color: #047857; }
.nn-chip-conf-med { background: #fffbeb; color: #b45309; }
.nn-chip-conf-low { background: #fef2f2; color: #b91c1c; }

.nn-escalate {
    background: #fff7ed; border: 1px solid #fdba74; border-radius: 12px;
    padding: 0.7rem 1rem; font-size: 0.9rem; color: #9a3412;
}

.nn-source {
    font-size: 0.8rem; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.5rem 0.75rem; margin-top: 0.3rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="nn-header">
        <div style="font-size:1.8rem;">🤖</div>
        <div>
            <p class="title">NeuralNav Support</p>
            <p class="subtitle">ML/DL-powered customer service assistant</p>
        </div>
        <div class="badge">● online</div>
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
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you today?", "meta": None, "message_id": None}
        ]
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id or 'not started'}`")
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
    elif confidence >= 0.45:
        cls = "nn-chip-conf-med"
    else:
        cls = "nn-chip-conf-low"
    return f'<span class="nn-chip {cls}">confidence {pct}%</span>'


def render_meta(meta: dict):
    if not meta:
        return
    intent_chip = f'<span class="nn-chip nn-chip-intent">intent: {meta["intent"]}</span>'
    st.markdown(
        f'<div class="nn-meta-row">{intent_chip}{confidence_chip(meta["confidence"])}</div>',
        unsafe_allow_html=True,
    )
    if meta.get("sources"):
        with st.expander(f"📚 {len(meta['sources'])} source(s) used"):
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
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        meta = msg.get("meta")
        if meta and meta.get("escalated"):
            st.markdown(f'<div class="nn-escalate">🔁 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(msg["content"])
        render_meta(meta)
        if msg["role"] == "assistant":
            render_feedback(msg.get("message_id"), idx)

if prompt := st.chat_input("Describe your issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None, "message_id": None})
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
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
                st.markdown(f'<div class="nn-escalate">🔁 {data["reply"]}</div>', unsafe_allow_html=True)
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
