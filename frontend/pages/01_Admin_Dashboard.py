"""Admin dashboard — analytics over logged conversations, pulled from the
backend's /stats endpoint. Streamlit auto-discovers this as a second page
since it lives in pages/ next to the main app.py.
"""
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="NeuralNav Admin", page_icon="◆", layout="wide")

CSS = """
<style>
:root {
    --bg-card: #FAF9F5; --border: #E5E1D8; --text: #2D2A26; --text-muted: #78746C;
}
.nn-page-header {
    padding: 1rem 1.3rem; border-radius: 14px; margin-bottom: 1rem;
    background: var(--bg-card); border: 1px solid var(--border);
}
.nn-page-header h1 { font-size: 1.3rem; margin: 0; color: var(--text); }
.nn-page-header p { font-size: 0.85rem; margin: 0.2rem 0 0 0; color: var(--text-muted); }
.nn-note {
    font-size: 0.8rem; color: var(--text-muted); margin-top: -0.4rem; margin-bottom: 1rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="nn-page-header">
        <h1>Admin Dashboard</h1>
        <p>Live analytics over logged conversations — every chat message and feedback
        click is recorded in Postgres and aggregated here.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Refresh"):
    st.rerun()

try:
    resp = requests.get(f"{BACKEND_URL}/stats", timeout=5)
    resp.raise_for_status()
    stats = resp.json()
except requests.RequestException as e:
    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats["total_conversations"])
col2.metric("Escalated", stats.get("escalated_count", 0))
col3.metric("Resolved automatically", stats.get("resolved_count", 0))
feedback = stats.get("feedback", {})
col4.metric("Escalation rate", f'{stats["escalation_rate"] * 100:.1f}%')

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Intent distribution")
    st.markdown(
        '<p class="nn-note">How many user messages were classified into each intent. '
        'A heavily skewed distribution suggests the KB/model coverage should focus on '
        'the most common intents first.</p>',
        unsafe_allow_html=True,
    )
    intent_dist = stats.get("intent_distribution", {})
    if intent_dist:
        df = pd.DataFrame(list(intent_dist.items()), columns=["intent", "count"]).set_index("intent")
        st.bar_chart(df)
    else:
        st.info("No conversations logged yet.")

with right:
    st.subheader("Confidence distribution")
    st.markdown(
        '<p class="nn-note">Classifier confidence across all user messages. A cluster '
        'near the escalation threshold (45%) means small model improvements would '
        'meaningfully change how often the bot hands off to a human.</p>',
        unsafe_allow_html=True,
    )
    confidences = stats.get("confidences", [])
    if confidences:
        df = pd.DataFrame({"confidence": confidences})
        st.bar_chart(df["confidence"].value_counts(bins=10).sort_index())
    else:
        st.info("No conversations logged yet.")

st.divider()

left2, right2 = st.columns(2)

with left2:
    st.subheader("Escalated vs. resolved")
    st.markdown(
        '<p class="nn-note">Of all user messages, how many were answered automatically '
        'versus handed off to a human agent.</p>',
        unsafe_allow_html=True,
    )
    esc_df = pd.DataFrame(
        {"count": [stats.get("resolved_count", 0), stats.get("escalated_count", 0)]},
        index=["Resolved", "Escalated"],
    )
    st.bar_chart(esc_df)

with right2:
    st.subheader("User feedback")
    st.markdown(
        '<p class="nn-note">👍/👎 clicks from the chat UI — a direct signal of whether '
        'automated answers were actually useful, independent of model confidence.</p>',
        unsafe_allow_html=True,
    )
    if feedback:
        fb_df = pd.DataFrame(
            {"count": [feedback.get("up", 0), feedback.get("down", 0)]},
            index=["👍 Up", "👎 Down"],
        )
        st.bar_chart(fb_df)
    else:
        st.info("No feedback submitted yet.")

st.divider()

st.subheader("Lowest-confidence intents")
st.markdown(
    '<p class="nn-note">Intents where the classifier is, on average, least sure of '
    'itself — good candidates for more training examples or KB entries.</p>',
    unsafe_allow_html=True,
)
low_conf = stats.get("low_confidence_intents", [])
if low_conf:
    low_conf_df = pd.DataFrame(low_conf)
    low_conf_df["avg_confidence"] = (low_conf_df["avg_confidence"] * 100).round(1).astype(str) + "%"
    low_conf_df.columns = ["Intent", "Avg. confidence", "Messages"]
    st.dataframe(low_conf_df, use_container_width=True, hide_index=True)
else:
    st.info("No conversations logged yet.")

st.divider()

st.subheader("Conversation volume over time")
st.markdown(
    '<p class="nn-note">User messages bucketed by hour — spikes here are worth '
    'cross-referencing with the intent distribution above to spot what drove traffic.</p>',
    unsafe_allow_html=True,
)
volume = stats.get("volume_over_time", [])
if volume:
    vol_df = pd.DataFrame(volume)
    vol_df["hour"] = pd.to_datetime(vol_df["hour"])
    vol_df = vol_df.set_index("hour")
    st.line_chart(vol_df["count"])
else:
    st.info("No conversations logged yet.")
