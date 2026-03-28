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
import time
from typing import Any

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pilgrims of the Stars",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Cinzel:wght@400;600&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

/* ── Root variables ── */
:root {
    --midnight:   #0b0d1a;
    --deep-navy:  #0f1225;
    --ink:        #141729;
    --gold:       #c9a84c;
    --gold-light: #e8c97a;
    --gold-dim:   #7a6030;
    --mist:       #d4cfc5;
    --parchment:  #f0ead8;
    --star-white: #f8f5ee;
    --accent:     #7b5ea7;
    --accent-soft:#a07fc8;
    --border:     rgba(201,168,76,0.20);
    --glow:       rgba(201,168,76,0.08);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'EB Garamond', serif;
    background-color: var(--midnight);
    color: var(--mist);
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a1535 0%, var(--midnight) 60%),
                radial-gradient(ellipse at 80% 100%, #0d1a2e 0%, transparent 60%);
    background-blend-mode: screen;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1021 0%, #111428 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--mist) !important;
}

/* ── Header ── */
.rag-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.rag-header .title {
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--gold-light);
    text-shadow: 0 0 40px rgba(201,168,76,0.3);
    margin: 0;
    line-height: 1.2;
}

.rag-header .subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--gold-dim);
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

.rag-header .ornament {
    color: var(--gold);
    font-size: 1.2rem;
    letter-spacing: 0.5em;
    opacity: 0.6;
    display: block;
    margin: 0.6rem 0;
}

/* ── Chat messages ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding: 0 0.5rem;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
}

.bubble-user {
    background: linear-gradient(135deg, #2a1f4a 0%, #1e1840 100%);
    border: 1px solid rgba(123,94,167,0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.2rem;
    max-width: 72%;
    font-size: 1rem;
    color: var(--star-white);
    line-height: 1.65;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.bubble-bot {
    background: linear-gradient(135deg, #141a2e 0%, #0f1525 100%);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.35rem;
    max-width: 82%;
    font-size: 1rem;
    color: var(--mist);
    line-height: 1.75;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 0 40px var(--glow);
}

.bubble-bot strong, .bubble-user strong {
    color: var(--gold-light);
}

/* Avatar labels */
.avatar-user {
    font-family: 'Cinzel', serif;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: var(--accent-soft);
    text-align: right;
    margin-bottom: 0.3rem;
    opacity: 0.8;
}

.avatar-bot {
    font-family: 'Cinzel', serif;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: var(--gold-dim);
    margin-bottom: 0.3rem;
    opacity: 0.8;
}

/* ── Sources expander ── */
.source-card {
    background: rgba(201,168,76,0.04);
    border: 1px solid rgba(201,168,76,0.15);
    border-left: 3px solid var(--gold-dim);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    line-height: 1.6;
}

.source-page {
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--gold);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.source-score {
    background: rgba(201,168,76,0.12);
    border-radius: 20px;
    padding: 0.1rem 0.5rem;
    font-size: 0.65rem;
    color: var(--gold-dim);
}

.source-text {
    color: #a09a8f;
    font-style: italic;
    font-size: 0.88rem;
}

/* ── Input area ── */
.stTextInput > div > div > input {
    background: rgba(20, 26, 46, 0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--star-white) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    caret-color: var(--gold) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(201,168,76,0.5) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.08) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4a5070 !important;
    font-style: italic;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c 0%, #a0803a 100%) !important;
    color: #0b0d1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.5rem 1.2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders & selectboxes in sidebar ── */
[data-testid="stSidebar"] .stSlider > div { filter: hue-rotate(200deg) brightness(0.9); }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--gold-dim) !important;
    background: rgba(201,168,76,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    background: rgba(11,13,26,0.6) !important;
    padding: 0.75rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Sidebar info box ── */
.info-box {
    background: rgba(201,168,76,0.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #9a9585;
}
.info-box .label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--gold-dim);
    margin-bottom: 0.35rem;
}

/* ── Welcome message ── */
.welcome {
    text-align: center;
    padding: 3rem 2rem;
    opacity: 0.65;
}
.welcome .star { font-size: 2.5rem; display: block; margin-bottom: 1rem; }
.welcome p {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: var(--gold-dim);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.8;
}

/* ── Status pills ── */
.pill {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    font-family: 'Cinzel', serif;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.pill-green  { background: rgba(80,200,100,0.12); color: #5ec870; border: 1px solid rgba(80,200,100,0.25); }
.pill-red    { background: rgba(200,80,80,0.12);  color: #e07070; border: 1px solid rgba(200,80,80,0.25); }
.pill-yellow { background: rgba(200,168,76,0.12); color: var(--gold); border: 1px solid rgba(200,168,76,0.25); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Late imports (after page config) ─────────────────────────────────────────
try:
    import config
    from retriever import retrieve, build_context
    from chat import generate_answer
    _imports_ok = True
    _import_error = ""
except ImportError as e:
    _imports_ok = False
    _import_error = str(e)


# ── Session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {"role", "content", "sources"}
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = bool(os.environ.get("GEMINI_API_KEY", ""))
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 1rem;">
        <div style="font-family:'Cinzel',serif; font-size:1.05rem; color:#c9a84c;
                    letter-spacing:0.15em;">✦ PILGRIMS ✦</div>
        <div style="font-family:'Cormorant Garamond',serif; font-style:italic;
                    font-size:0.82rem; color:#7a6030; margin-top:0.2rem;">
            of the Stars
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── API Key input ──
    st.markdown('<div class="info-box"><div class="label">GEMINI API KEY</div>',
                unsafe_allow_html=True)
    api_input = st.text_input(
        "API Key",
        type="password",
        placeholder="AIza…",
        label_visibility="collapsed",
        value=os.environ.get("GEMINI_API_KEY", ""),
    )
    if api_input:
        os.environ["GEMINI_API_KEY"] = api_input
        if _imports_ok:
            import config as _cfg
            _cfg.GEMINI_API_KEY = api_input
        st.session_state.api_key_set = True

    status_html = (
        '<span class="pill pill-green">● CONNECTED</span>'
        if st.session_state.api_key_set else
        '<span class="pill pill-red">○ NOT SET</span>'
    )
    st.markdown(f"<div style='margin-top:0.4rem;'>{status_html}</div></div>",
                unsafe_allow_html=True)

    st.divider()

    # ── Retrieval settings ──
    st.markdown('<div style="font-family:\'Cinzel\',serif; font-size:0.68rem; '
                'letter-spacing:0.15em; color:#7a6030; margin-bottom:0.6rem;">'
                'RETRIEVAL SETTINGS</div>', unsafe_allow_html=True)

    top_k = st.slider("Passages to retrieve (top-k)", 1, 10,
                      st.session_state.top_k, key="top_k_slider")
    st.session_state.top_k = top_k

    show_sources = st.toggle("Show source passages", value=True)

    st.divider()

    # ── Book info ──
    st.markdown("""
    <div class="info-box">
        <div class="label">ABOUT THE BOOK</div>
        <em>Pilgrims of the Stars</em> is the spiritual autobiography of
        <strong style="color:#c9a84c">Dilip Kumar Roy</strong> and
        <strong style="color:#c9a84c">Indira Devi</strong> — two seekers whose
        journey led them from the world of music and art to the inner life of
        yoga and devotion.<br><br>
        384 pages · First published 1973
    </div>
    """, unsafe_allow_html=True)

    # ── Clear chat ──
    if st.button("✦  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Index status ──
    from pathlib import Path
    idx_exists = Path("faiss_index/index.faiss").exists()
    idx_pill = ('<span class="pill pill-green">● INDEX READY</span>'
                if idx_exists else
                '<span class="pill pill-yellow">○ RUN ingest.py</span>')
    st.markdown(f"<div style='margin-top:0.8rem; text-align:center;'>{idx_pill}</div>",
                unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
col_pad_l, col_main, col_pad_r = st.columns([0.5, 9, 0.5])

with col_main:
    # ── Header ──
    st.markdown("""
    <div class="rag-header">
        <h1 class="title">PILGRIMS OF THE STARS</h1>
        <span class="ornament">· · ✦ · ·</span>
        <p class="subtitle">Autobiography of Two Yogis &nbsp;·&nbsp; Dilip Kumar Roy &amp; Indira Devi</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Error banner if imports failed ──
    if not _imports_ok:
        st.error(f"⚠️ Import error: `{_import_error}`\n\n"
                 "Make sure all dependencies are installed:\n"
                 "```\npip install -r requirements.txt\n```")

    # ── Chat history ──
    chat_area = st.container()

    with chat_area:
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
                    # Convert newlines to <br> for HTML rendering
                    html_content = msg["content"].replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="msg-bot">
                      <div style="width:100%">
                        <div class="avatar-bot">✦ ORACLE</div>
                        <div class="bubble-bot">{html_content}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # Sources expander
                    if show_sources and msg.get("sources"):
                        with st.expander(
                            f"✦  {len(msg['sources'])} source passages retrieved",
                            expanded=False
                        ):
                            for i, src in enumerate(msg["sources"], 1):
                                snippet = src["text"][:300].replace("<", "&lt;")
                                score_pct = int(src["score"] * 100)
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-page">
                                        EXCERPT {i} &nbsp;·&nbsp; PAGE {src['page']}
                                        <span class="source-score">
                                            {score_pct}% match
                                        </span>
                                    </div>
                                    <div class="source-text">"{snippet}…"</div>
                                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Input row ──
    input_col, btn_col = st.columns([9, 1.4])

    with input_col:
        user_query = st.text_input(
            "query",
            placeholder="Ask a question about the book…",
            label_visibility="collapsed",
            key="query_input",
        )

    with btn_col:
        send = st.button("✦  Ask", use_container_width=True)

    # ── Example prompts ──
    st.markdown(
        "<div style='font-family:\"Cinzel\",serif; font-size:0.6rem; "
        "letter-spacing:0.12em; color:#4a5070; margin:0.5rem 0 0.4rem;'>"
        "EXAMPLE QUESTIONS</div>",
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

    # Guard: API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
        st.stop()

    # Guard: imports
    if not _imports_ok:
        st.error("Dependencies not installed. Run `pip install -r requirements.txt`.")
        st.stop()

    # Guard: index
    from pathlib import Path
    if not Path("faiss_index/index.faiss").exists():
        st.error("FAISS index not found. Run `python ingest.py` first.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "sources": [],
    })

    # RAG pipeline with spinner
    with st.spinner("Searching the book…"):
        try:
            sources = retrieve(user_query, top_k=st.session_state.top_k)
            context = build_context(sources)
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            st.stop()

    with st.spinner("Consulting the oracle…"):
        try:
            answer = generate_answer(user_query, context)
        except Exception as exc:
            st.error(f"Generation error: {exc}")
            st.stop()

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    st.rerun()
