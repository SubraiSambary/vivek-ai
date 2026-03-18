# backend/gif_engine.py
# GIF responses — Giphy free tier (optional) with emoji fallback

import requests
import os

GIPHY_KEY = os.getenv("GIPHY_API_KEY", "")

# Emoji fallbacks when no Giphy key — VIVEK still feels expressive
EMOJI_MAP = {
    "excited":     "🎉🔥✨",
    "thinking":    "🤔💭🧠",
    "celebrating": "🎊🥳🙌",
    "confused":    "😵‍💫🤯😅",
    "cool":        "😎🕶️⚡",
    "sad":         "😢💔🥺",
    "love":        "❤️🥰💖",
    "funny":       "😂🤣💀",
    "search":      "🔍🌐📡",
    "docs":        "📄📚🗂️",
    "default":     "😄👍✨",
}

def get_gif_url(mood: str) -> str | None:
    """Fetch a GIF URL from Giphy. Returns None if no key or request fails."""
    if not GIPHY_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.giphy.com/v1/gifs/random",
            params={"api_key": GIPHY_KEY, "tag": mood, "rating": "g"},
            timeout=3,
        )
        data = resp.json()
        return data["data"]["images"]["fixed_height"]["url"]
    except Exception:
        return None

def get_reaction(mood: str) -> dict:
    """
    Returns {"type": "gif", "url": ...} or {"type": "emoji", "text": ...}
    VIVEK's frontend handles both gracefully.
    """
    gif_url = get_gif_url(mood)
    if gif_url:
        return {"type": "gif", "url": gif_url}
    return {"type": "emoji", "text": EMOJI_MAP.get(mood, EMOJI_MAP["default"])}

def detect_mood(text: str) -> str:
    """Detect what reaction mood fits the message."""
    t = text.lower()
    if any(w in t for w in ["great", "amazing", "excellent", "perfect", "wah"]):
        return "excited"
    if any(w in t for w in ["think", "hmm", "wonder", "maybe", "perhaps"]):
        return "thinking"
    if any(w in t for w in ["congrat", "well done", "awesome", "badhiya"]):
        return "celebrating"
    if any(w in t for w in ["confus", "don't understand", "unclear"]):
        return "confused"
    if any(w in t for w in ["haha", "lol", "funny", "joke"]):
        return "funny"
    if any(w in t for w in ["search", "looking", "find", "google"]):
        return "search"
    if any(w in t for w in ["document", "pdf", "file", "upload"]):
        return "docs"
    return "default"