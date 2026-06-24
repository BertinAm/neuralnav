"""PostgreSQL persistence for conversation logs and feedback.

Backs the /history and /stats endpoints and the Streamlit admin dashboard.
Connection is configured via the DATABASE_URL env var (see .env.example),
defaulting to the docker-compose 'postgres' service for local/dev use.
"""
import os
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neuralnav:neuralnav@localhost:5432/neuralnav",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    intent TEXT,
    confidence REAL,
    escalated BOOLEAN DEFAULT FALSE,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    rating TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)


def log_message(session_id: str, role: str, text: str, intent: str | None = None,
                 confidence: float | None = None, escalated: bool = False) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, text, intent, confidence, escalated, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (session_id, role, text, intent, confidence, escalated, time.time()),
            )
            return cur.fetchone()[0]


def log_feedback(message_id: int, rating: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (message_id, rating, created_at) VALUES (%s, %s, %s)",
                (message_id, rating, time.time()),
            )


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


def get_stats() -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) c FROM messages WHERE role='user'")
            total = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) c FROM messages WHERE role='assistant' AND escalated=TRUE")
            escalated = cur.fetchone()["c"]

            cur.execute(
                "SELECT intent, COUNT(*) c FROM messages WHERE role='user' AND intent IS NOT NULL "
                "GROUP BY intent ORDER BY c DESC"
            )
            intent_counts = cur.fetchall()

            cur.execute("SELECT confidence FROM messages WHERE role='user' AND confidence IS NOT NULL")
            confidences = cur.fetchall()

            cur.execute("SELECT rating, COUNT(*) c FROM feedback GROUP BY rating")
            feedback_counts = cur.fetchall()

    return {
        "total_conversations": total,
        "escalation_rate": (escalated / total) if total else 0.0,
        "intent_distribution": {r["intent"]: r["c"] for r in intent_counts},
        "confidences": [r["confidence"] for r in confidences],
        "feedback": {r["rating"]: r["c"] for r in feedback_counts},
    }
