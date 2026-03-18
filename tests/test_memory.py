# tests/test_memory.py
# Phase 1, Ex 2: Simulating VIVEK's memory — dicts + JSON
# This IS the memory system. Phase 3 just wraps it in SQLite.

import json
import os
from datetime import datetime

MEMORY_FILE = "data/memory/vivek_memory.json"

# ─────────────────────────────────────────────
# CORE MEMORY FUNCTIONS
# ─────────────────────────────────────────────

def load_memory() -> dict:
    """Load all user memory from disk. Returns empty dict if first run."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory: dict) -> None:
    """Persist memory dict to disk as JSON."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    print(f"  💾 Memory saved to {MEMORY_FILE}")


def get_or_create_user(memory: dict, user_id: str) -> dict:
    """
    Get existing user profile or create a fresh one.
    This is a nested dict — a dict inside a dict.
    memory = { "user_001": { ...profile... }, "user_002": { ... } }
    """
    if user_id not in memory:
        memory[user_id] = {
            "user_id": user_id,
            "name": None,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "total_messages": 0,
            "topics_discussed": [],       # list of strings
            "personality_notes": {},      # dict inside a dict!
            "chat_history": [],           # list of dicts
        }
        print(f"  🆕 New user created: {user_id}")
    else:
        memory[user_id]["last_seen"] = datetime.now().isoformat()
        print(f"  👋 Welcome back: {memory[user_id].get('name', user_id)}")
    return memory[user_id]


def add_message(user_profile: dict, role: str, content: str) -> None:
    """
    Append a message to chat history.
    role = "user" or "assistant"
    This list-of-dicts is EXACTLY what gets sent to Groq API later.
    """
    user_profile["chat_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    user_profile["total_messages"] += 1


def get_context_window(user_profile: dict, last_n: int = 6) -> list:
    """
    Return last N messages formatted for Groq API.
    Groq wants: [{"role": "user", "content": "..."}, ...]
    We strip timestamps — Groq doesn't need them.
    """
    history = user_profile["chat_history"][-last_n:]
    return [{"role": m["role"], "content": m["content"]} for m in history]


def update_personality_note(user_profile: dict, key: str, value) -> None:
    """
    Store something VIVEK learned about this user.
    e.g. update_personality_note(profile, "prefers_hindi", True)
         update_personality_note(profile, "expertise", "beginner Python")
    """
    user_profile["personality_notes"][key] = value
    print(f"  🧠 VIVEK noted: {key} = {value}")


def build_vivek_system_prompt(user_profile: dict) -> str:
    """
    Dynamically build VIVEK's system prompt using what he knows about the user.
    This is where memory becomes personality — a dict becomes a string for the LLM.
    """
    name = user_profile.get("name") or "yaar"
    notes = user_profile.get("personality_notes", {})
    topics = user_profile.get("topics_discussed", [])

    # Build memory context string from dict
    memory_lines = []
    if notes.get("expertise"):
        memory_lines.append(f"- Their expertise level: {notes['expertise']}")
    if notes.get("prefers_hindi"):
        memory_lines.append("- They enjoy Hindi/Hinglish — use it more!")
    if topics:
        memory_lines.append(f"- Topics we've discussed: {', '.join(topics[-5:])}")

    memory_context = "\n".join(memory_lines) if memory_lines else "- New user, learn about them!"

    return f"""You are VIVEK — विवेक — The Wise One.
You are witty, occasionally sarcastic, always warm, and have a desi Indian soul.
Use emojis naturally. Call users '{name}' or 'yaar' or 'bhai' sometimes.
Roast gently, encourage genuinely. Never be boring.
When you search the web, announce it dramatically.
When you find something in a document, be excited about it.

What you remember about this user:
{memory_context}

Respond in a conversational, fun way. Mix Hindi/Hinglish naturally when appropriate."""


# ─────────────────────────────────────────────
# SIMULATION — watch memory build across "sessions"
# ─────────────────────────────────────────────

print("=" * 50)
print("SESSION 1 — First time user meets VIVEK")
print("=" * 50)

memory = load_memory()
profile = get_or_create_user(memory, "yaar_001")

# User introduces themselves
profile["name"] = "Rahul"
update_personality_note(profile, "expertise", "intermediate Python")
update_personality_note(profile, "prefers_hindi", True)
profile["topics_discussed"].append("Python dictionaries")

add_message(profile, "user", "Hey VIVEK! I just started learning Python.")
add_message(profile, "assistant", "Arrey Rahul bhai! Welcome! Dictionaries samajh aaye? 😄")
add_message(profile, "user", "Yes! Dict comprehensions are cool.")
add_message(profile, "assistant", "Bilkul sahi! You'll be using them EVERYWHERE. 🔥")

save_memory(memory)

print("\n--- System prompt VIVEK would send to Groq ---")
print(build_vivek_system_prompt(profile))

print("\n--- Context window sent to Groq API ---")
context = get_context_window(profile, last_n=4)
for msg in context:
    print(f"  [{msg['role']}]: {msg['content']}")

print("\n" + "=" * 50)
print("SESSION 2 — User comes back next day")
print("=" * 50)

# Simulate app restart — reload from disk
memory = load_memory()
profile = get_or_create_user(memory, "yaar_001")  # Should say "Welcome back"

print(f"\n  User's name: {profile['name']}")
print(f"  Total messages so far: {profile['total_messages']}")
if "Python dictionaries" not in profile["topics_discussed"]:
    profile["topics_discussed"].append("Python dictionaries")
print(f"  Personality notes: {profile['personality_notes']}")

add_message(profile, "user", "VIVEK, explain embeddings to me!")
add_message(profile, "assistant", "Oho! Getting into the good stuff! 🧠 Embeddings are like...")
if "embeddings" not in profile["topics_discussed"]:
    profile["topics_discussed"].append("embeddings")

save_memory(memory)
print("\n✅ Memory persisted. VIVEK will remember this next session too.")

print("\n--- Updated system prompt (notice it knows more now) ---")
print(build_vivek_system_prompt(profile))