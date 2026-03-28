"""
config.py — Central configuration for the RAG chatbot.
API key is loaded automatically from a .env file in the project root.
"""

import os
from dotenv import load_dotenv

load_dotenv()   # reads .env → populates os.environ silently

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# Embedding model — gemini-embedding-001 (GA, June 2025)
#   • 3 072-dimensional vectors (4x richer than text-embedding-004's 768-dim)
#   • 8 192-token input limit
#   • #1 on MTEB Multilingual leaderboard
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

# Chat / generation model — gemini-2.5-flash (GA, replaces retired 1.5-flash)
#   • gemini-1.5-flash was retired April 29 2025
#   • gemini-2.5-flash: best price-performance, low latency, strong reasoning
CHAT_MODEL: str = "gemini-2.5-flash"

# ── Chunking ─────────────────────────────────────────────────────────────────
# Chunk size: 800 characters (~150-180 words).
#   Rationale: "Pilgrims of the Stars" is narrative prose. 800 chars keeps
#   each chunk semantically coherent (a scene or argument) while staying well
#   within gemini-embedding-001's generous 8 192-token input limit.
#   Too small -> chunks lose meaning; too large -> retrieval becomes noisy.
CHUNK_SIZE: int = 800

# Overlap: 150 characters (~30 words, ~19% of chunk).
#   Rationale: Avoids hard cuts mid-sentence and preserves cross-boundary
#   context (e.g., a spiritual concept introduced at the end of one chunk
#   and elaborated at the start of the next).
CHUNK_OVERLAP: int = 150

# ── FAISS / retrieval ────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = "faiss_index/index.faiss"
METADATA_PATH: str = "faiss_index/metadata.json"
TOP_K: int = 5          # number of chunks returned per query

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
