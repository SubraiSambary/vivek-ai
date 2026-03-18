# frontend/app.py
# Streamlit chat UI — VIVEK's face to the world

import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="VIVEK — विवेक",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .vivek-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .vivek-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .vivek-sub {
        color: #888;
        font-size: 0.95rem;
        margin-top: -0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .badge-web  { background:#e8f4fd; color:#1a73e8; }
    .badge-doc  { background:#e8f8f0; color:#1a8a4a; }
    .status-bar {
        text-align: center;
        padding: 0.3rem;
        font-size: 0.8rem;
        color: #888;
        border-top: 1px solid #f0f0f0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div class="vivek-header">
    <div class="vivek-title">🧠 VIVEK</div>
    <div class="vivek-sub">विवेक · The Wise One · Always caffeinated ☕</div>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────
if "user_id"  not in st.session_state:
    st.session_state.user_id   = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages  = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Who are you, yaar?")
    name_input = st.text_input(
        "Your name",
        value=st.session_state.user_name or "",
        placeholder="e.g. Rahul, Priya..."
    )
    if name_input:
        st.session_state.user_name = name_input

    st.divider()
    st.markdown("### 📄 Upload documents")
    st.caption("VIVEK will answer questions from your files")
    uploaded = st.file_uploader(
        "PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )
    if uploaded:
        with st.spinner("VIVEK is reading your document... 📖"):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(),
                                uploaded.type)},
                data={"user_id": st.session_state.user_id},
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ {data['message']}")
            else:
                st.error("Upload failed. Is the backend running?")

    st.divider()

    # Doc count
    try:
        cnt = requests.get(
            f"{API_URL}/doc-count/{st.session_state.user_id}",
            timeout=2
        ).json().get("count", 0)
        st.caption(f"📚 {cnt} document chunks in VIVEK's memory")
    except Exception:
        st.caption("📡 Backend connecting...")

    st.divider()
    st.markdown("### ⚙️ Settings")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("VIVEK v1.0 · Built with ❤️ + Python")
    st.caption(f"Session ID: `{st.session_state.user_id}`")

# ── Chat history display ─────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"],
                         avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Show badges if assistant used web/docs
        if msg["role"] == "assistant":
            badges = ""
            if msg.get("used_web"):
                badges += '<span class="badge badge-web">🌐 web search</span>'
            if msg.get("used_docs"):
                badges += '<span class="badge badge-doc">📄 from your docs</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)

        # Show emoji reaction
        if msg.get("reaction"):
            r = msg["reaction"]
            if r["type"] == "gif":
                st.image(r["url"], width=200)
            else:
                st.markdown(f"<div style='font-size:1.4rem'>{r['text']}</div>",
                            unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Talk to VIVEK... ask anything, upload docs, search the web!"):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Call VIVEK
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("VIVEK is thinking... 🤔"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "user_id":   st.session_state.user_id,
                        "message":   prompt,
                        "user_name": st.session_state.user_name,
                    },
                    timeout=30,
                )
                data = resp.json()
                reply    = data.get("reply", "Yaar, something went wrong! 😅")
                reaction = data.get("reaction", {"type": "emoji", "text": "🤔"})
                used_web = data.get("used_web",  False)
                used_docs= data.get("used_docs", False)

            except Exception as e:
                reply    = f"Arrey! Backend se connection nahi ho raha 😅 Make sure FastAPI is running!\n\nError: {e}"
                reaction = {"type": "emoji", "text": "😵‍💫"}
                used_web = used_docs = False

        st.markdown(reply)

        # Badges
        badges = ""
        if used_web:  badges += '<span class="badge badge-web">🌐 web search</span>'
        if used_docs: badges += '<span class="badge badge-doc">📄 from your docs</span>'
        if badges:    st.markdown(badges, unsafe_allow_html=True)

        # Reaction
        if reaction["type"] == "gif":
            st.image(reaction["url"], width=200)
        else:
            st.markdown(
                f"<div style='font-size:1.4rem'>{reaction['text']}</div>",
                unsafe_allow_html=True
            )

    # Save to session
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   reply,
        "reaction":  reaction,
        "used_web":  used_web,
        "used_docs": used_docs,
    })

# ── Status bar ───────────────────────────────────────────
st.markdown(
    '<div class="status-bar">VIVEK · विवेक · Groq + ChromaDB + Streamlit · 100% Free 🚀</div>',
    unsafe_allow_html=True
)