# backend/memory.py
# Upgrade from JSON (Phase 1) to SQLite — same concepts, production-grade storage

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "data/memory/vivek.db"

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name like a dict
    return conn

def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,
            name          TEXT,
            first_seen    TEXT,
            last_seen     TEXT,
            total_messages INTEGER DEFAULT 0,
            personality_notes TEXT DEFAULT '{}',
            topics_discussed  TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT,
            role       TEXT,
            content    TEXT,
            timestamp  TEXT
        );
    """)
    conn.commit()
    conn.close()

def get_or_create_user(user_id: str, name: str = None) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()

    now = datetime.now().isoformat()
    if not row:
        conn.execute(
            """INSERT INTO users
               (user_id, name, first_seen, last_seen, total_messages,
                personality_notes, topics_discussed)
               VALUES (?, ?, ?, ?, 0, '{}', '[]')""",
            (user_id, name, now, now)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    else:
        conn.execute(
            "UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id)
        )
        conn.commit()

    user = dict(row)
    user["personality_notes"] = json.loads(user["personality_notes"])
    user["topics_discussed"]  = json.loads(user["topics_discussed"])
    conn.close()
    return user

def save_user(user: dict):
    conn = get_connection()
    conn.execute(
        """UPDATE users SET
           name = ?, last_seen = ?, total_messages = ?,
           personality_notes = ?, topics_discussed = ?
           WHERE user_id = ?""",
        (
            user["name"],
            datetime.now().isoformat(),
            user["total_messages"],
            json.dumps(user["personality_notes"], ensure_ascii=False),
            json.dumps(user["topics_discussed"],  ensure_ascii=False),
            user["user_id"],
        )
    )
    conn.commit()
    conn.close()

def add_message(user_id: str, role: str, content: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.now().isoformat())
    )
    conn.execute(
        "UPDATE users SET total_messages = total_messages + 1 WHERE user_id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()

def get_context_window(user_id: str, last_n: int = 8) -> list:
    """Return last N messages formatted exactly as Groq API expects."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE user_id = ?
           ORDER BY id DESC LIMIT ?""",
        (user_id, last_n)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def add_topic(user: dict, topic: str):
    """Add topic without duplicates."""
    if topic not in user["topics_discussed"]:
        user["topics_discussed"].append(topic)

def update_note(user: dict, key: str, value):
    user["personality_notes"][key] = value