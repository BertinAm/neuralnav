# NeuralNav — ML/DL Customer Service Chatbot

**Live demo:** [neuralnav.streamlit.app](https://neuralnav.streamlit.app/)
**Backend API:** [neuralnav-backend-mxjt.onrender.com](https://neuralnav-backend-mxjt.onrender.com/health)
**Source:** [github.com/BertinAm/neuralnav](https://github.com/BertinAm/neuralnav)

A customer-service chatbot built to demonstrate, measure, and compare classical
ML against deep learning on the same task — not just wire an LLM API to a
chat window. Intent classification (TF-IDF baseline vs. fine-tuned
DistilBERT), entity extraction, semantic retrieval over a knowledge base,
and a full-stack deployment with real persistence and analytics.

---

## The problem

Most "AI chatbot" course/portfolio projects either hardcode responses or
wrap a hosted LLM API — neither actually demonstrates applied ML/DL
technique. NeuralNav is built around a concrete, falsifiable question
instead: **does fine-tuning a small transformer actually outperform a
classical baseline on this task, and by how much?** Every other piece of
the system (retrieval, entity extraction, escalation logic, deployment) is
built to support answering that question end-to-end, in production, not
just in a notebook.

## Architecture

```
┌─────────────┐      ┌──────────────────────────────────────────┐      ┌────────────┐
│  Streamlit   │ HTTP │                 FastAPI                  │      │ PostgreSQL │
│   frontend   │─────▶│  intent classify → entity extract →      │─────▶│ (Render)   │
│ (Streamlit   │      │  retrieve/escalate → log                 │      │            │
│   Cloud)     │      │           (Render, Docker)                │      └────────────┘
└─────────────┘      └──────────────────────────────────────────┘
                                      │
                                      ▼
                          Hugging Face Hub repo
                     (trained models, KB, eval artifacts —
                      pushed from Kaggle, pulled at runtime)
```

Training happens on **Kaggle** (free T4 GPU), pushes artifacts to a
**Hugging Face Hub** repo, and the backend pulls them at startup —
no model weights committed to git, no local GPU required to develop or
deploy.

## ML/DL: the actual comparison

Trained on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(27 intents, ~27k examples), deliberately restricted to **25 examples/class**
for training — at the full dataset size both models hit ~98-99% accuracy on
Bitext's templated phrasing, which flattens the comparison into a non-result.
Under data scarcity, the gap is real and measurable:

| Model | Macro F1 (25 examples/class) |
|---|---|
| TF-IDF + Logistic Regression | 87.7% |
| Fine-tuned DistilBERT | **98.5%** |

DistilBERT's pretrained language understanding lets it generalize from far
fewer labeled examples than a from-scratch classical model — a concrete,
quantified data-efficiency argument for DL, not just a vibes-based one.
Confusion matrices, per-class F1 charts, and the training loss curve are
all generated in [notebooks/01_intent_classification.ipynb](notebooks/01_intent_classification.ipynb).

Retrieval (sentence-transformers + FAISS, evaluated the same way) hits
**87% hit-rate@1 / 98% hit-rate@5** against a held-out query set —
[notebooks/02_retrieval_and_ner.ipynb](notebooks/02_retrieval_and_ner.ipynb).

## Engineering problems solved (not just features built)

A few things that separate this from a tutorial clone:

- **Memory-constrained production deployment.** Render's free tier caps
  memory at 512MB; the full DistilBERT + sentence-transformers + spaCy
  stack OOM-kills on it. Built an automatic lightweight fallback
  ([ml/resource_mode.py](ml/resource_mode.py)) that detects Render and
  switches to a TF-IDF classifier + TF-IDF retrieval pipeline — measured
  at **130MB resident memory**, comfortably under budget — while local
  Docker and the Kaggle notebooks keep using the full DL stack the
  reported numbers above are based on. Required separately calibrating
  confidence/retrieval thresholds per backend, since TF-IDF's probability
  outputs are far less peaked than BERT's softmax (correct predictions
  cluster ~10-30% vs. BERT's ~70%+) — the same absolute cutoff silently
  broke escalation logic under the lightweight path.
- **Escalation that checks retrieval quality, not just classifier
  confidence.** A confidently-wrong intent paired with a weak knowledge-base
  match used to still answer fluently with no uncertainty signal at all.
  Added a second signal (retrieval match score) so either weak signal
  triggers a human handoff.
- **Caught and fixed a live conversational bug**: traced a real multi-turn
  failure (the bot re-asking for a shipping address the user had just
  provided) to the system having no slot-filling state — every turn was
  classified independently with no memory of what it had already asked.
  Fixed with a narrow, scoped slot-tracking mechanism rather than bolting
  on a full dialogue manager.
- **A correctness bug invisible in offline eval**: the Bitext dataset has
  zero greeting/chitchat intents (it's 27 purely task-specific intents),
  so a plain "Hello" got force-classified into a random low-confidence
  intent and escalated — something no amount of staring at a
  classification report would have caught, only live testing did.

## Stack

**ML/DL**: scikit-learn, PyTorch, Hugging Face Transformers & Hub, sentence-transformers, FAISS, spaCy
**Backend**: FastAPI, PostgreSQL, psycopg2, Docker
**Frontend**: Streamlit
**Infra**: Kaggle (training), Hugging Face Hub (model registry), Render (backend + Postgres), Streamlit Community Cloud (frontend)

## What I'd add next

- A real dialogue-management layer (RASA's TED policy) instead of the
  narrow slot-filling patch, for genuine multi-turn task completion
- Multilingual support (intent classification + retrieval in a second language)
- A/B comparing the retrieval-based answers against an LLM-generation
  layer conditioned on the same retrieved context
