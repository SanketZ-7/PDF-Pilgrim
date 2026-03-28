"""
retriever.py — Load the FAISS index and retrieve top-k relevant chunks
               for any natural-language query.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import google.generativeai as genai
import numpy as np
import faiss

import config


# ── Load index & metadata (cached after first load) ──────────────────────────

@lru_cache(maxsize=1)
def _load_index() -> faiss.Index:
    """Load the FAISS index from disk (once per process)."""
    if not Path(config.FAISS_INDEX_PATH).exists():
        sys.exit(
            f"[ERROR] FAISS index not found at '{config.FAISS_INDEX_PATH}'.\n"
            "Run  python ingest.py  first."
        )
    return faiss.read_index(config.FAISS_INDEX_PATH)


@lru_cache(maxsize=1)
def _load_metadata() -> list[dict[str, Any]]:
    """Load chunk metadata from disk (once per process)."""
    if not Path(config.METADATA_PATH).exists():
        sys.exit(
            f"[ERROR] Metadata not found at '{config.METADATA_PATH}'.\n"
            "Run  python ingest.py  first."
        )
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Query embedding ───────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string using Gemini.
    task_type="RETRIEVAL_QUERY" is the asymmetric counterpart to
    RETRIEVAL_DOCUMENT used during ingestion.
    Returns a float32 array of shape (1, dim).
    """
    genai.configure(api_key=config.GEMINI_API_KEY)

    for attempt in range(4):
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY",
            )
            vec = np.array([result["embedding"]], dtype=np.float32)
            faiss.normalize_L2(vec)    # must match how documents were indexed
            return vec
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[warn] Embedding attempt {attempt+1} failed: {exc}. "
                  f"Retrying in {wait}s …")
            time.sleep(wait)

    sys.exit("[ERROR] Could not embed query after 4 attempts.")


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for `query`.

    Returns a list of dicts:
        { "chunk_id": int, "page": int, "text": str, "score": float }
    sorted by descending similarity score.
    """
    index = _load_index()
    metadata = _load_metadata()

    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:          # FAISS returns -1 when fewer than k vectors exist
            continue
        chunk = metadata[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    return results


# ── Formatted context builder ─────────────────────────────────────────────────

def build_context(results: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered, page-cited context block
    ready to be injected into the prompt.
    """
    if not results:
        return "(No relevant passages found.)"

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[Excerpt {i} — p. {r['page']}]\n{r['text']}"
        )
    return "\n\n".join(lines)
