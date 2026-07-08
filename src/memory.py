# src/memory.py — long-term memory: user profiles, conversation history, preferences
import json
import sqlite3
import time
from contextlib import contextmanager

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    transcript TEXT,
    face_emotion_label TEXT,
    face_emotion_confidence REAL,
    voice_emotion_label TEXT,
    voice_emotion_confidence REAL,
    response_text TEXT,
    action_type TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    updated_at REAL,
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def ensure_user(user_id: str, display_name: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, display_name, created_at) VALUES (?, ?, ?)",
            (user_id, display_name or user_id, time.time()),
        )


def save_interaction(
    user_id: str,
    transcript: str,
    face_emotion_label: str,
    face_emotion_confidence: float,
    voice_emotion_label: str,
    voice_emotion_confidence: float,
    response_text: str,
    action_type: str,
) -> None:
    ensure_user(user_id)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversations
               (user_id, timestamp, transcript, face_emotion_label, face_emotion_confidence,
                voice_emotion_label, voice_emotion_confidence, response_text, action_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, time.time(), transcript, face_emotion_label, face_emotion_confidence,
             voice_emotion_label, voice_emotion_confidence, response_text, action_type),
        )


def get_recent_history(user_id: str, n: int = config.CONVERSATION_HISTORY_TURNS) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT timestamp, transcript, face_emotion_label, voice_emotion_label, response_text, action_type
               FROM conversations WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, n),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def update_preference(user_id: str, key: str, value) -> None:
    ensure_user(user_id)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (user_id, key, json.dumps(value), time.time()),
        )


def get_preferences(user_id: str) -> dict:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def get_user_profile(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "preferences": get_preferences(user_id),
        "recent_history": get_recent_history(user_id),
    }
