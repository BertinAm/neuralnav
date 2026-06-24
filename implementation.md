# Implementation Plan — NeuralNav Customer Service Chatbot

Tracks everything that needs to be built, what's done, and what's left.
Update checkboxes as work progresses; pair with `memory.md` for the dated
change log.

## 1. Data
- [x] `data/intents.csv`, `data/kb.json` — small **dev-fallback samples only**
      (see `data/README.md`) — not for training, just local smoke-testing
- [x] Real data wired in: **Bitext Customer Support dataset** (~27 intents,
      ~26.8k examples) pulled directly from the HF Hub inside
      `notebooks/01_intent_classification.ipynb`
- [x] `kb.json` auto-derived from the dataset's own responses in the same
      notebook (no manual KB writing needed)
- [ ] Run the notebook on Kaggle and download the real `kb.json` +
      `intents_real.csv` into `data/`, replacing the dev samples for the
      actual training/deployment run
- [ ] Niche is now whatever Bitext covers (general e-commerce/account/
      order/shipping support, ~10 categories) — confirm this is acceptable
      vs. the originally-discussed electronics-troubleshooting niche

## 2. ML/DL — Kaggle notebooks (T4 GPU)
- [x] `notebooks/01_intent_classification.ipynb`
  - [x] Baseline: TF-IDF + Logistic Regression
  - [x] DL: fine-tuned DistilBERT (`distilbert-base-uncased`)
  - [x] Side-by-side classification reports (per-class F1) for the report
  - [x] Run on Kaggle — fixed BERT undertraining (see memory.md), now gets
        98.5% macro-F1 vs baseline's 87.7% on the low-data set, a real gap
  - [x] HF Hub upload cell at the end — pushes models/reports/kb/figures to
        a Hugging Face repo (`HF_TOKEN` via Kaggle secret or env var)
  - [ ] Download artifacts (from Kaggle Output tab or the HF repo) into
        local `models/`
- [x] `notebooks/02_retrieval_and_ner.ipynb`
  - [x] sentence-transformers embeddings (`all-MiniLM-L6-v2`) + FAISS index over KB
  - [x] spaCy NER + regex entity extraction sanity check (order IDs, error codes)
  - [x] Retrieval evaluation: hit-rate@1/@3/@5 using `intents_real.csv` (notebook
        01's held-out labeled set) as queries against the KB
  - [x] Visualizations: overall + per-intent hit-rate bars, correct-vs-incorrect
        score distribution, PCA scatter of KB embeddings, NER entity-type
        coverage chart — all saved to `figures/` for the report
  - [ ] Run on Kaggle, download `kb_embeddings.npy`, `kb_index.faiss`, `figures/`
        (optional — `ml/retrieval.py` currently rebuilds the index at backend
        startup instead, so these are for the report, not required for the app)

## 3. Backend (FastAPI) — local code, heavy libs run in Docker only
- [x] `backend/main.py` — `/chat` endpoint: intent classify -> entity extract ->
      escalate-or-retrieve -> reply
- [x] `ml/intent_classifier.py` — loads BERT if present, else baseline,
      pulling either from the HF Hub repo via `ml/hf_hub.py` if not local
- [x] `ml/hf_hub.py` — shared helper: download a file/dir from the HF repo
      that notebooks 01/02 push to, only if not already present locally
- [x] `ml/ner.py` — regex (order IDs, error codes) + spaCy statistical NER
- [x] `ml/retrieval.py` — FAISS semantic search; prefers the real `kb.json`
      pulled from the HF Hub repo over the `data/kb.json` dev sample
- [x] `backend/db.py` — PostgreSQL persistence for messages + feedback
      (via `psycopg2`, configured by `DATABASE_URL`; local dev uses the
      `postgres` service in `docker-compose.yml`)
- [x] Conversation history: client sends recent turns in `history`, backend
      concatenates last 2 user turns as context for intent classification
      (helps disambiguate short follow-ups like "yes" or "the second one")
- [x] `POST /feedback` — thumbs up/down per message, logged to SQLite
- [x] `GET /history/{session_id}` — replay a session's messages
- [x] `GET /stats` — aggregates for the admin dashboard (intent distribution,
      escalation rate, confidence list, feedback counts)
- [ ] Swap the lightweight escalation/greeting logic for RASA dialogue
      management if the course wants a trainable dialogue policy (see §6)
- [x] `ml/slot_filling.py` — narrow slot-filling for the one most visible
      failure mode found in testing: the bot used to repeat the same
      "please give me your address" question forever even after the user
      answered it. Now tracks a per-session pending-slot (in-memory, not
      persisted) and treats the next non-question turn as the answer.
- [ ] General multi-turn dialogue state beyond that one slot (current
      "memory" is otherwise context-for-classification only) — see RASA
      option in §6 for the real fix if more slots are needed

## 4. Frontend (Streamlit)
- [x] `frontend/app.py` — redesigned: gradient header, chat bubbles with
      avatars, intent/confidence chips (color-coded), source citation cards,
      distinct escalation styling, typing indicator, sidebar with session
      ID + new-conversation reset + backend health check, per-message
      👍/👎 feedback buttons wired to `/feedback`
- [x] `frontend/pages/01_Admin_Dashboard.py` — second Streamlit page (auto
      multipage): total conversations, escalation rate, feedback counts,
      intent distribution bar chart, confidence histogram
- [ ] Multilingual support (not implemented — would need a multilingual
      DistilBERT variant + language detection step)

## 5. Containerization & local run
- [x] `Dockerfile.backend` — bundles `ml/`, `backend/`, `data/`, `models/`
- [x] `Dockerfile.frontend` — lightweight, uses `requirements-dev.txt`
- [x] `docker-compose.yml` — wires backend + frontend together
- [ ] Verify `docker compose up --build` end-to-end once trained models are
      downloaded into `models/`

## 6. Dialogue management (RASA) — optional/stretch
- [ ] Separate venv, `rasa init`, wire NLU to `data/intents.csv`
- [ ] Connect RASA's TED policy as the dialogue manager instead of the
      if/else logic in `backend/main.py`
- [ ] Document in the report as the "trainable DL dialogue policy" component

## 7. Deployment
- [ ] Push trained model artifacts + repo to GitHub (decide: commit models/
      directly, or use HF Hub + `from_pretrained` to keep the repo small)
- [ ] Deploy backend (Render or Railway, Docker-based) — see README hosting table
- [ ] Deploy frontend (Streamlit Community Cloud, pointed at hosted backend URL)
- [ ] Confirm public URLs work end-to-end before submitting

## 8. Evaluation / report artifacts
- [ ] Baseline vs BERT comparison table (from the two `*_report.json` files)
- [ ] NER precision/recall on a small labeled sample
- [ ] Retrieval hit-rate@k on a held-out query set
- [ ] 5-10 recorded sample conversations (qualitative end-to-end section)
- [ ] Architecture diagram for the report (can generate on request)

## Open decisions (need your input before finalizing)
- Final niche/domain for the KB content (electronics support assumed — confirm or change)
- Whether RASA dialogue management is in scope or out of scope for grading
- Whether to commit trained model weights to git or host them on HF Hub
