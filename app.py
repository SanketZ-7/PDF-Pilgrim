"""
app.py — Streamlit frontend for the "Pilgrims of the Stars" RAG chatbot.

Run:
    streamlit run app.py

Requires the FAISS index to already be built:
    python ingest.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pilgrims of the Stars",
    page_icon="✦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --glass-bg: rgba(255, 255, 255, 0.06);
    --glass-bg-strong: rgba(255, 255, 255, 0.12);
    --glass-border: rgba(255, 255, 255, 0.15);
    --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    --text-main: #f5f5f7;
    --text-muted: rgba(255, 255, 255, 0.65);
    --accent: #0A84FF;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-main);
}

.stApp {
    background: radial-gradient(circle at top right, #2c2d3e 0%, #12131c 100%);
    background-attachment: fixed;
}

/* Ambient glowing orbs for the macOS wallpaper effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 10%;
    left: 15%;
    width: 400px;
    height: 400px;
    background: rgba(10, 132, 255, 0.35);
    filter: blur(120px);
    border-radius: 50%;
    z-index: -1;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: 10%;
    right: 15%;
    width: 350px;
    height: 350px;
    background: rgba(255, 55, 95, 0.25);
    filter: blur(120px);
    border-radius: 50%;
    z-index: -1;
}

.rag-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    margin-bottom: 1.5rem;
}
.rag-header .title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--text-main);
    margin: 0;
    line-height: 1.2;
}
.rag-header .subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    letter-spacing: 0.02em;
    margin-top: 0.4rem;
    font-weight: 400;
}
.rag-header .ornament {
    color: var(--text-muted);
    font-size: 1.2rem;
    letter-spacing: 0.5em;
    opacity: 0.3;
    display: block;
    margin: 0.6rem 0;
}

.chat-container { display: flex; flex-direction: column; gap: 1.2rem; padding: 0 0.5rem; }

.msg-user { display: flex; justify-content: flex-end; }
.msg-bot  { display: flex; justify-content: flex-start; }

.bubble-user {
    background: var(--accent);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px 20px 4px 20px;
    padding: 0.85rem 1.2rem;
    max-width: 72%;
    font-size: 1.1rem;
    color: #000000 !important;
    line-height: 1.5;
    box-shadow: var(--glass-shadow);
}
.bubble-bot {
    background: rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 20px 20px 20px 4px;
    padding: 0.85rem 1.35rem;
    max-width: 82%;
    font-size: 1.1rem;
    color: #000000 !important;
    line-height: 1.6;
    box-shadow: var(--glass-shadow);
}
.bubble-bot strong, .bubble-user strong, .bubble-bot *, .bubble-user * { color: #000000 !important; font-weight: 600; }

.avatar-user {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    text-align: right;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.avatar-bot {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.source-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.source-page {
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--text-main);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.source-score {
    background: var(--glass-bg-strong);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 0.1rem 0.5rem;
    font-size: 0.85rem;
    color: var(--text-main);
}
.source-text { color: var(--text-muted); font-size: 0.95rem; }

.stTextInput > div > div > input {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text-main) !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1.2rem !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.3) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
}

.stButton > button {
    background: var(--glass-bg-strong) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: var(--glass-shadow) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    transform: scale(1.02) !important;
}

hr { border-color: var(--glass-border) !important; }

.streamlit-expanderHeader {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--text-main) !important;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    background: rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    padding: 1rem !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

.welcome {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    box-shadow: var(--glass-shadow);
    margin-top: 2rem;
}
.welcome .star { font-size: 2.5rem; display: block; margin-bottom: 1rem; }
.welcome p {
    font-size: 1.15rem;
    color: var(--text-main);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.8;
    font-weight: 300;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
    .rag-header { padding: 1rem 0; margin-bottom: 1rem; }
    .rag-header .title { font-size: 2rem; }
    .rag-header .subtitle { font-size: 1rem; }
    .bubble-user { max-width: 90%; padding: 0.7rem 1rem; font-size: 1.05rem; }
    .bubble-bot { max-width: 95%; padding: 0.7rem 1rem; font-size: 1.05rem; }
    .source-card { padding: 0.6rem 0.8rem; font-size: 0.95rem; }
    .welcome { padding: 1.5rem 1rem; }
    .welcome p { font-size: 1.05rem; line-height: 1.5; }
    .stButton > button { padding: 0.4rem 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Late imports ──────────────────────────────────────────────────────────────
try:
    import config
    from retriever import retrieve, build_context
    from chat import generate_answer
    _imports_ok = True
    _import_error = ""
except ImportError as e:
    _imports_ok = False
    _import_error = str(e)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


# ── Configuration Variables ───────────────────────────────────────────────────
show_sources = True


# ── Main layout ───────────────────────────────────────────────────────────────
col_main = st.container()

with col_main:

    # ── Header ──
    st.markdown("""
    <div class="rag-header">
        <h1 class="title">PILGRIMS OF THE STARS</h1>
        <span class="ornament">· · ✦ · ·</span>
        <p class="subtitle">Autobiography of Two Yogis &nbsp;·&nbsp; Dilip Kumar Roy &amp; Indira Devi</p>
    </div>
    """, unsafe_allow_html=True)

    if not _imports_ok:
        st.error(f"Import error: `{_import_error}`\n\nRun: `pip install -r requirements.txt`")

    # ── Chat history ──
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <span class="star">✦</span>
            <p>Ask anything about the book — the spiritual journey of Dilip Kumar Roy
            and Indira Devi, their encounters with saints, their practice of yoga,
            or the philosophy of the pilgrimage of the soul.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                  <div>
                    <div class="avatar-user">YOU</div>
                    <div class="bubble-user">{msg["content"]}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                html_content = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div class="msg-bot">
                  <div style="width:100%">
                    <div class="avatar-bot">✦ ORACLE</div>
                    <div class="bubble-bot">{html_content}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                if show_sources and msg.get("sources"):
                    with st.expander(
                        f"✦  {len(msg['sources'])} source passages retrieved",
                        expanded=False,
                    ):
                        for i, src in enumerate(msg["sources"], 1):
                            snippet = src["text"][:300].replace("<", "&lt;")
                            score_pct = int(src["score"] * 100)
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-page">
                                    EXCERPT {i} &nbsp;·&nbsp; PAGE {src['page']}
                                    <span class="source-score">{score_pct}% match</span>
                                </div>
                                <div class="source-text">"{snippet}..."</div>
                            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Input row ──
    input_col, btn_col = st.columns([9, 1.4])
    with input_col:
        user_query = st.text_input(
            "query",
            placeholder="Ask a question about the book...",
            label_visibility="collapsed",
            key="query_input",
        )
    with btn_col:
        send = st.button("✦  Ask", use_container_width=True)

    # ── Example prompts ──
    st.markdown(
        '<div style="font-size:0.85rem; font-weight:600; '
        'letter-spacing:0.05em; color:var(--text-muted); margin:0.5rem 0 0.4rem; text-transform:uppercase;">'
        'EXAMPLE QUESTIONS</div>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(3)
    examples = [
        "Who is Dilip Kumar Roy?",
        "What is the role of kirtan in the book?",
        "Describe Indira Devi's spiritual experiences.",
    ]
    for col, ex in zip(ex_cols, examples):
        with col:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:10]}"):
                user_query = ex
                send = True


# ── Query handling ────────────────────────────────────────────────────────────
if send and user_query and user_query.strip():

    if not _imports_ok:
        st.error("Dependencies not installed. Run `pip install -r requirements.txt`.")
        st.stop()

    from pathlib import Path
    if not Path("faiss_index/index.faiss").exists():
        st.error("FAISS index not found. Run `python ingest.py` first.")
        st.stop()

    # Guard: API key missing
    if not config.GEMINI_API_KEY:
        st.error(
            "**GEMINI_API_KEY not found.**\n\n"
            "- **Locally:** add `GEMINI_API_KEY=your-key` to your `.env` file.\n"
            "- **Streamlit Cloud:** add it under App Settings → Secrets."
        )
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "sources": [],
    })

    with st.spinner("Searching the book..."):
        try:
            sources = retrieve(user_query, top_k=st.session_state.top_k)
            context = build_context(sources)
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            st.stop()

    with st.spinner("Consulting the oracle..."):
        try:
            answer = generate_answer(user_query, context)
        except Exception as exc:
            st.error(f"Generation error: {exc}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    st.rerun()
