# backend/web_search.py
# DuckDuckGo search — free, no API key, no rate limits for casual use

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

def web_search(query: str, max_results: int = 4) -> str:
    """
    Search DuckDuckGo and return formatted results.
    Returns empty string on failure — VIVEK degrades gracefully.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ""

        parts = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body",  "")
            href  = r.get("href",  "")
            parts.append(f"**{title}**\n{body}\nSource: {href}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        return f"Search failed: {str(e)}"

def should_search_web(message: str) -> bool:
    """
    Decide if VIVEK should search the web for this message.
    Simple keyword heuristic — good enough for Phase 3.
    """
    search_triggers = [
        "latest", "recent", "today", "news", "current", "2024", "2025",
        "who is", "what is", "how to", "where is", "when did", "price",
        "best", "top", "weather", "score", "results", "search", "find",
        "tell me about", "explain", "what happened",
    ]
    msg_lower = message.lower()
    return any(trigger in msg_lower for trigger in search_triggers)