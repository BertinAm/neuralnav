"""Admin dashboard — intent distribution, confidence histogram, escalation
rate, and feedback counts, pulled from the backend's /stats endpoint.

Streamlit auto-discovers this as a second page since it lives in pages/
next to the main app.py.
"""
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="NeuralNav Admin", page_icon="📊", layout="wide")
st.title("📊 NeuralNav Admin Dashboard")
st.caption("Live analytics over logged conversations — refresh to update.")

if st.button("🔄 Refresh"):
    st.rerun()

try:
    resp = requests.get(f"{BACKEND_URL}/stats", timeout=5)
    resp.raise_for_status()
    stats = resp.json()
except requests.RequestException as e:
    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total conversations", stats["total_conversations"])
col2.metric("Escalation rate", f'{stats["escalation_rate"] * 100:.1f}%')
feedback = stats.get("feedback", {})
col3.metric("Feedback (👍 / 👎)", f'{feedback.get("up", 0)} / {feedback.get("down", 0)}')

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Intent distribution")
    intent_dist = stats.get("intent_distribution", {})
    if intent_dist:
        df = pd.DataFrame(list(intent_dist.items()), columns=["intent", "count"]).set_index("intent")
        st.bar_chart(df)
    else:
        st.info("No conversations logged yet.")

with right:
    st.subheader("Confidence distribution")
    confidences = stats.get("confidences", [])
    if confidences:
        df = pd.DataFrame({"confidence": confidences})
        st.bar_chart(df["confidence"].value_counts(bins=10).sort_index())
    else:
        st.info("No conversations logged yet.")
