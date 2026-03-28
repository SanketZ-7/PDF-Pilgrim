"""
config.py — Central configuration for the RAG chatbot.
Loads GEMINI_API_KEY from .env locally, or st.secrets on Streamlit Cloud.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Try st.secrets first (Streamlit Cloud), fall back to .env
try:
    import streamlit as st
    GEMINI_API_KEY: str = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# Embedding model
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

# Chat model
CHAT_MODEL: str = "models/gemini-3-pro-preview"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# ── FAISS / retrieval ────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = "faiss_index/index.faiss"
METADATA_PATH: str = "faiss_index/metadata.json"
TOP_K: int = 5

# ── Paths ────────────────────────────────────────────────────────────────────
PDF_PATH: str = "Pilgrims_of_the_Stars.pdf"

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are a knowledgeable assistant specialised in the book
"Pilgrims of the Stars: Autobiography of Two Yogis" by Dilip Kumar Roy and
Indira Devi.

RULES:
1. Answer ONLY using the context passages provided below.
2. Always cite the source with its page number, e.g. (p. 42), or chapter/section
   title when available.
3. If the provided context does not contain enough information to answer the
   question, say exactly: "I'm sorry, that information is not covered in the
   book's content I have access to."
4. Do NOT speculate, hallucinate, or draw on outside knowledge.
5. Keep answers concise yet complete; use bullet points for lists.

CONTEXT (retrieved excerpts from the book):
{context}

USER QUESTION:
{question}

ANSWER:"""
