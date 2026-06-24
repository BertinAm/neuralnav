# NeuralNav — Customer Service Chatbot

ML/DL capstone project: electronics-support chatbot combining classical ML
(intent baseline), deep learning (fine-tuned DistilBERT), NLP (spaCy NER),
and a retrieval pipeline (sentence-transformers + FAISS) behind a FastAPI
backend with a Streamlit chat frontend.

## Project layout

```
data/            intents.csv (training data), kb.json (knowledge base)
ml/              train_baseline.py, train_bert.py, ner.py, retrieval.py, intent_classifier.py
backend/         FastAPI app (main.py) — the /chat endpoint
frontend/        Streamlit chat UI (app.py)
models/          generated after training (gitignored except reports)
rasa/            optional dialogue-management subproject (see below)
```

## Workflow (no heavy local installs)

Training happens on **Kaggle** (T4 GPU), not locally — see `notebooks/`.
Local disk only ever holds lightweight packages (`requirements-dev.txt`):
FastAPI, Streamlit, requests, pydantic. Heavy ML libs (torch, transformers,
sentence-transformers, faiss, spacy) only run inside Kaggle or inside the
backend's Docker image.

1. Run `notebooks/01_intent_classification.ipynb` on Kaggle (GPU T4, internet
   on) — pulls the real **Bitext Customer Support dataset** straight from the
   HF Hub (no manual upload needed), trains baseline + BERT, and
   auto-derives a real `kb.json` from the dataset's own responses.
   → produces `baseline_intent.joblib`, `bert_intent/`, `*_report.json`, `kb.json`, `intents_real.csv`.
2. Download `kb.json` from step 1's output, upload it as a small Kaggle
   Dataset (e.g. `neuralnav-kb`), then run
   `notebooks/02_retrieval_and_ner.ipynb` → produces `kb_embeddings.npy`,
   `kb_index.faiss` (sanity-checked in the notebook).
3. Download all outputs from the Kaggle "Output" tab into your local
   `models/` folder, and `kb.json` into `data/` (replacing the dev sample —
   see `data/README.md`).
4. Build and run everything via Docker (heavy libs only exist inside the
   container, never on your local disk):

```powershell
docker compose up --build
# postgres -> localhost:5432 (neuralnav/neuralnav)
# backend  -> http://localhost:8000
# frontend -> http://localhost:8501
```

For local *editing* (not running) of backend/frontend code with IDE support:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Backend & model details

- **Intent classification**: `ml/intent_classifier.py` auto-selects the
  fine-tuned DistilBERT model if `models/bert_intent/` exists, otherwise
  falls back to the TF-IDF+LogisticRegression baseline. Both are trained on
  Kaggle in `notebooks/01_intent_classification.ipynb` — keep both reports
  (`models/baseline_report.json`, `models/bert_report.json`) — the
  classical-vs-DL comparison is the centerpiece of the ML/DL writeup.
- **Entity recognition**: `ml/ner.py` — regex for structured tokens (order
  IDs, error codes) + spaCy's statistical NER for free-text entities.
- **Retrieval**: `ml/retrieval.py` embeds `data/kb.json` with
  `all-MiniLM-L6-v2` and does cosine search via FAISS `IndexFlatIP`.
- **Decision logic** (`backend/main.py`): confidence below 0.45, or intent
  `escalate_human`, routes to a human-handoff reply instead of retrieval.
- **API**: single `POST /chat` endpoint, `{message, session_id}` in,
  `{reply, intent, confidence, escalated, entities, sources}` out. `GET
  /health` for liveness checks.

## Dialogue management (RASA) — optional, separate environment

RASA's dependency pins conflict with the `transformers`/`torch` versions
used above, so run it in its own venv if you want trainable dialogue
policy (TED policy) as a graded DL component instead of the lightweight
state machine in `backend/main.py`:

```powershell
python -m venv .venv-rasa
.venv-rasa\Scripts\activate
pip install rasa==3.6.20
rasa init --no-prompt
```

Wire its NLU intents to `data/intents.csv` and connect via the REST channel
so the FastAPI backend can call out to the RASA server for multi-turn state
instead of handling each message independently.

## Database

Conversation logs and feedback are stored in **PostgreSQL** (`backend/db.py`,
via `psycopg2`), configured through the `DATABASE_URL` env var (see
`.env.example`). `docker-compose.yml` runs a local `postgres:16-alpine`
service for dev; swap `DATABASE_URL` to a managed Postgres instance (see
hosting table below) for deployment.

## Hosting options

| Layer | Option | Notes |
|---|---|---|
| **Frontend (Streamlit)** | Streamlit Community Cloud | Free, point it at your GitHub repo, deploys `frontend/app.py` directly — no Docker needed. Set `BACKEND_URL` as a secret pointing at your hosted backend. |
| | HuggingFace Spaces (Streamlit SDK) | Free, also git-push deploy, good if you want everything HF-hosted next to your model. |
| **Backend (FastAPI)** | Render | Free tier web service, builds straight from `Dockerfile.backend`, auto HTTPS, sleeps on idle (free tier) — fine for a course demo. |
| | Railway | Similar to Render, usage-based free credits, slightly faster cold starts. |
| | Fly.io | Free allowance, good if you want the container to stay warm (avoids cold-start lag during grading). |
| | Google Cloud Run / AWS App Runner | True free tier, scales to zero, more setup overhead — pick if your course wants "real cloud" on the resume. |
| **Models** | Bundle into the backend image (current setup) | Simplest: `models/` copied into `Dockerfile.backend`. Fine for these model sizes (DistilBERT ~260MB, MiniLM ~80MB). |
| | HuggingFace Hub + `from_pretrained("your-username/model")` | Decouples model from backend image, faster CI builds, push trained weights there instead of committing them to git. |
| **Vector DB** | FAISS in-process (current setup) | No separate service needed — fine at this KB scale. |
| | Managed (Pinecone/Qdrant Cloud free tier) | Only worth it if your KB grows past what fits comfortably in memory. |
| **Database (conversations/feedback)** | Render Postgres (free tier) | Easiest if backend is already on Render — same dashboard, internal network URL. |
| | Neon / Supabase (free tier) | Serverless Postgres, generous free tier, works from any host — good if backend lives elsewhere. |
| | Railway Postgres | One-click add-on if backend is on Railway. |
| | docker-compose `postgres` service (current local setup) | Local dev only — see `docker-compose.yml`. |

**Recommended combo for this project**: Render (backend, Docker) +
Streamlit Community Cloud (frontend) — both free, both deploy from the same
GitHub repo, and the split mirrors a real microservice architecture for
your report's deployment section.

### Deploying: Render (backend + Postgres)

1. Push this repo to GitHub (already done).
2. In the [Render dashboard](https://dashboard.render.com): **New +** ->
   **Blueprint**, connect this GitHub repo. Render reads `render.yaml` at
   the repo root and creates both the `neuralnav-backend` web service and
   the `neuralnav-db` Postgres database automatically.
3. After the first deploy, open the `neuralnav-backend` service ->
   **Environment** and add `HF_TOKEN` manually (kept out of `render.yaml`
   and out of git on purpose — paste the same token you use in your local
   `.env`/Kaggle secret).
4. Wait for the deploy to finish, then copy the service's public URL
   (`https://neuralnav-backend-xxxx.onrender.com`) — you'll need it for the
   frontend. Confirm it's alive: `curl https://<that-url>/health`.

Free tier note: the service sleeps after 15 min idle and takes ~30-60s to
wake on the next request — expected, not a bug, if a professor's first
request is slow.

### Deploying: Streamlit Community Cloud (frontend)

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, **New app**, pick this repo/branch.
2. Set **Main file path** to `frontend/app.py`. Streamlit Cloud
   auto-detects `frontend/requirements.txt` (lightweight — streamlit,
   requests, pandas) since it sits next to the entry-point script, so it
   won't try to install the heavy backend dependencies.
3. Under **Advanced settings -> Secrets**, add:
   ```
   BACKEND_URL = "https://neuralnav-backend-xxxx.onrender.com"
   ```
   (the Render URL from the step above). `frontend/app.py` reads this via
   `os.environ.get("BACKEND_URL", ...)` — Streamlit Cloud injects secrets
   as env vars automatically.
4. Deploy. Visit the generated `*.streamlit.app` URL and confirm the
   sidebar shows "🟢 Backend reachable" before testing the chat.

## Evaluation artifacts for the report

- `models/baseline_report.json` vs `models/bert_report.json` — accuracy/F1 per intent, classical vs DL.
- NER: spot-check `ml/ner.py` outputs against a labeled sample.
- Retrieval: log `score` from `KBRetriever.search()` as a proxy for hit-rate@k.
- End-to-end: record 5-10 sample conversations through the Streamlit UI for a qualitative section.
