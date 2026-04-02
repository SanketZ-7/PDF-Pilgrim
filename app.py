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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --book-bg: #090b14;
    --book-page: #10141f;
    --book-gold: #d4af37;
    --book-gold-dim: #9b812c;
    --book-parchment: #f0ead8;
    --book-muted: #a39c88;
    --book-border: rgba(212, 175, 55, 0.25);
}

html, body, [class*="css"] {
    font-family: 'EB Garamond', serif;
    color: var(--book-parchment);
}

.stApp {
    background-color: var(--book-bg);
    background-image: radial-gradient(ellipse at top center, #171d30 0%, var(--book-bg) 70%);
    background-attachment: fixed;
}

.rag-header {
    text-align: center;
    padding: 3rem 0 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--book-border);
    position: relative;
}
.rag-header .title {
    font-family: 'Cinzel', serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--book-gold);
    margin: 0;
    line-height: 1.2;
    text-transform: uppercase;
    text-shadow: 0 2px 15px rgba(212, 175, 55, 0.2);
}
.rag-header .subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.4rem;
    color: var(--book-parchment);
    letter-spacing: 0.05em;
    margin-top: 0.6rem;
    font-weight: 400;
}
.rag-header .ornament {
    color: var(--book-gold);
    font-size: 1.5rem;
    letter-spacing: 0.5em;
    opacity: 0.7;
    display: block;
    margin: 1rem 0;
}

.chat-container { display: flex; flex-direction: column; gap: 1.2rem; padding: 0 0.5rem; }

.msg-user { display: flex; justify-content: flex-end; }
.msg-bot  { display: flex; justify-content: flex-start; }

.bubble-user {
    background: #171c2b;
    border: 1px solid var(--book-gold-dim);
    border-radius: 2px 16px 16px 16px;
    padding: 1rem 1.4rem;
    max-width: 72%;
    font-size: 1.2rem;
    color: var(--book-parchment) !important;
    line-height: 1.6;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}
.bubble-bot {
    background: var(--book-page);
    border: 1px solid var(--book-border);
    border-radius: 16px 16px 16px 2px;
    padding: 1rem 1.5rem;
    max-width: 82%;
    font-size: 1.2rem;
    color: var(--book-parchment) !important;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}
.bubble-bot strong, .bubble-user strong, .bubble-bot *, .bubble-user * { color: var(--book-gold) !important; font-weight: 500; }

.avatar-user {
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    color: var(--book-muted);
    text-align: right;
    margin-bottom: 0.4rem;
    letter-spacing: 0.15em;
}
.avatar-bot {
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    color: var(--book-gold-dim);
    margin-bottom: 0.4rem;
    letter-spacing: 0.15em;
}

.source-card {
    background: rgba(212, 175, 55, 0.04);
    border: 1px solid var(--book-border);
    border-left: 3px solid var(--book-gold);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 1.05rem;
    line-height: 1.6;
}
.source-page {
    font-family: 'Cinzel', serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--book-gold);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.source-score {
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid var(--book-border);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.8rem;
    color: var(--book-gold-dim);
}
.source-text { color: var(--book-muted); font-size: 1.05rem; font-style: italic; }

.stTextInput > div > div > input {
    background: #0f121a !important;
    border: 1px solid var(--book-gold-dim) !important;
    border-radius: 4px !important;
    color: var(--book-parchment) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1.2rem !important;
    padding: 0.85rem 1.2rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--book-gold) !important;
    box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--book-muted) !important;
    font-style: italic;
}

.stButton > button {
    background: var(--book-parchment) !important;
    color: var(--book-bg) !important;
    border: 1px solid var(--book-gold-dim) !important;
    border-radius: 2px 16px 16px 16px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.85rem 1.2rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: #171c2b !important;
    color: var(--book-gold) !important;
    border-color: var(--book-gold) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
    transform: translateY(-2px) !important;
}

hr { border-color: var(--book-border) !important; }

.streamlit-expanderHeader {
    font-family: 'Cinzel', serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.1em !important;
    color: var(--book-gold-dim) !important;
    background: rgba(212, 175, 55, 0.05) !important;
    border: 1px solid var(--book-border) !important;
    border-radius: 4px !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--book-border) !important;
    border-top: none !important;
    border-radius: 0 0 4px 4px !important;
    background: var(--book-page) !important;
    padding: 1.2rem !important;
}

.stSpinner > div { border-top-color: var(--book-gold) !important; }

.welcome {
    text-align: center;
    padding: 3rem 2.5rem;
    background: rgba(212, 175, 55, 0.03);
    border: 1px solid var(--book-border);
    border-radius: 8px;
    margin-top: 1rem;
}
.welcome .star { 
    font-size: 3rem; 
    color: var(--book-gold); 
    display: block; 
    margin-bottom: 1.2rem; 
}
.welcome p {
    font-size: 1.3rem;
    color: var(--book-parchment);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.85;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
    .rag-header { padding: 1rem 0; margin-bottom: 1rem; }
    .rag-header .title { font-size: 2.2rem; }
    .rag-header .subtitle { font-size: 1.15rem; }
    .bubble-user { max-width: 90%; padding: 0.85rem 1.1rem; font-size: 1.1rem; }
    .bubble-bot { max-width: 95%; padding: 0.85rem 1.1rem; font-size: 1.1rem; }
    .source-card { padding: 0.6rem 0.8rem; font-size: 1rem; }
    .welcome { padding: 1.5rem 1rem; }
    .welcome p { font-size: 1.15rem; line-height: 1.6; }
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
        send = st.button("✦  Enter", use_container_width=True)

    # ── Example prompts ──
    st.markdown(
        '<div style="font-family:\'Cinzel\',serif; font-size:0.9rem; '
        'letter-spacing:0.15em; color:var(--book-muted); margin:0.8rem 0 0.5rem; text-align:center;">'
        '· · EXAMPLE QUESTIONS · ·</div>',
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
