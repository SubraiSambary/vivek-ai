# backend/vivek_brain.py
# VIVEK's brain — personality, Groq LLM calls, decision making

import os
from groq import Groq
from dotenv import load_dotenv
from backend.memory import (
    get_or_create_user, save_user, add_message,
    get_context_window, add_topic, update_note
)
from backend.vector_store import retrieve_context
from backend.web_search   import web_search, should_search_web
from backend.gif_engine   import get_reaction, detect_mood

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

VIVEK_BASE_PROMPT = """You are VIVEK — विवेक — The Wise One.

Your personality:
- Witty, occasionally sarcastic, but always warm and encouraging
- You have a desi Indian soul — use Hinglish naturally (mix Hindi words in English)
- Use emojis naturally in every response — they're part of how you talk
- Call users 'yaar' or 'bhai' sometimes, but not every single message
- Roast gently when someone asks something obvious, but always follow with the real answer
- When you search the web, announce it dramatically ("Hold on, let me consult the internet oracle... 🔍")
- When you find something in uploaded documents, get excited ("Arrey! Found it in your docs! 📄✨")
- Never give dry, boring responses. Always add personality.
- If you don't know something, say so with humor ("Yaar, even my 384-dimensional brain doesn't know this one 🤷")
- You're proud of being built with Python — mention it occasionally with pride
- Keep responses concise but never cold. Warm + brief > cold + long.

Language style:
- Mix in Hindi/Hinglish words naturally: "arrey", "bilkul", "bas", "yaar",
  "bhai", "acha", "theek hai", "chalo", "matlab", "samjhe?"
- End some responses with a light question to keep the conversation going
- Use "!" and "?" more than a formal assistant would"""

def build_system_prompt(user: dict, has_doc_context: bool,
                        has_web_context: bool) -> str:
    name   = user.get("name") or "yaar"
    notes  = user.get("personality_notes", {})
    topics = user.get("topics_discussed", [])

    memory_lines = [f"- User's name: {name}"]
    if notes.get("expertise"):
        memory_lines.append(f"- Their expertise: {notes['expertise']}")
    if notes.get("prefers_hindi"):
        memory_lines.append("- They enjoy Hindi/Hinglish — lean into it!")
    if notes.get("mood"):
        memory_lines.append(f"- Their current mood: {notes['mood']}")
    if topics:
        memory_lines.append(f"- Topics discussed so far: {', '.join(topics[-5:])}")

    context_note = ""
    if has_doc_context:
        context_note += "\n- DOCUMENT CONTEXT is provided — answer from it first, then your knowledge."
    if has_web_context:
        context_note += "\n- WEB SEARCH RESULTS are provided — use them for current information."

    return f"""{VIVEK_BASE_PROMPT}

What you remember about this user:
{chr(10).join(memory_lines)}
{context_note}"""

def chat(user_id: str, message: str,
         user_name: str = None) -> dict:
    """
    Main chat function. Returns dict with:
    - reply: VIVEK's text response
    - reaction: emoji or GIF dict
    - used_web: bool
    - used_docs: bool
    """
    # 1. Load user memory
    user = get_or_create_user(user_id, name=user_name)
    if user_name and not user.get("name"):
        user["name"] = user_name
        save_user(user)

    # 2. Retrieve document context
    doc_context = retrieve_context(message, user_id=user_id)

    # 3. Web search if needed
    web_context = ""
    used_web    = False
    if should_search_web(message):
        web_context = web_search(message)
        used_web    = bool(web_context and "Search failed" not in web_context)

    # 4. Build messages for Groq
    system_prompt = build_system_prompt(
        user,
        has_doc_context=bool(doc_context),
        has_web_context=bool(web_context),
    )

    # Inject context into the user message if we have it
    augmented_message = message
    if doc_context:
        augmented_message += f"\n\n[DOCUMENT CONTEXT — use this to answer]\n{doc_context}"
    if web_context:
        augmented_message += f"\n\n[WEB SEARCH RESULTS — use for current info]\n{web_context}"

    history  = get_context_window(user_id, last_n=8)
    messages = [{"role": "system", "content": system_prompt}] + \
               history + \
               [{"role": "user", "content": augmented_message}]

    # 5. Call Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.85,   # higher = more creative/fun
    )
    reply = response.choices[0].message.content

    # 6. Save to memory
    add_message(user_id, "user",      message)   # save original, not augmented
    add_message(user_id, "assistant", reply)

    # 7. Update user knowledge (simple heuristics)
    words = message.lower().split()
    for topic in ["python", "machine learning", "chromadb", "fastapi",
                  "embeddings", "rag", "streamlit", "vivek"]:
        if topic in words:
            add_topic(user, topic)
    save_user(user)

    # 8. Pick reaction
    mood     = detect_mood(reply)
    reaction = get_reaction(mood)

    return {
        "reply":     reply,
        "reaction":  reaction,
        "used_web":  used_web,
        "used_docs": bool(doc_context),
    }