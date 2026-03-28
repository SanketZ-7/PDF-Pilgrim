"""
ingest.py — Extract text from the PDF, chunk it, embed with Gemini,
            and store in a FAISS index.

Run once (or re-run to refresh the index):
    python ingest.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
import numpy as np
import faiss
from pypdf import PdfReader
from tqdm import tqdm

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def validate_env() -> None:
    """Abort early with a helpful message if the API key is missing."""
    if not config.GEMINI_API_KEY:
        sys.exit(
            "\n[ERROR] GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export GEMINI_API_KEY='your-key-here'   # Linux / macOS\n"
            "  set GEMINI_API_KEY=your-key-here         # Windows CMD\n"
        )


# ── Step 1 · Extract text page-by-page ───────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    """
    Return a list of dicts:
        { "page": int, "text": str }
    Page numbers are 1-based to match the printed book.
    """
    reader = PdfReader(pdf_path)
    pages: list[dict[str, Any]] = []

    print(f"[1/4] Extracting text from {len(reader.pages)} pages …")
    for idx, page in enumerate(tqdm(reader.pages, unit="page"), start=1):
        raw = page.extract_text() or ""
        cleaned = " ".join(raw.split())        # collapse whitespace / newlines
        if cleaned:                            # skip blank / image-only pages
            pages.append({"page": idx, "text": cleaned})

    print(f"      → {len(pages)} pages contained extractable text.")
    return pages


# ── Step 2 · Chunk pages into overlapping windows ───────────────────────────

def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Slide a window of `chunk_size` characters with `overlap` characters of
    stride over the concatenated text of each page.

    Each chunk carries:
        { "chunk_id": int, "page": int, "text": str, "start_char": int }

    Chunking is done *within* each page (not across pages) so that page
    number attribution is unambiguous.
    """
    print(f"[2/4] Chunking (size={chunk_size} chars, overlap={overlap} chars) …")
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 30:           # ignore tiny trailing fragments
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": chunk_text,
                        "start_char": start,
                    }
                )
                chunk_id += 1

            if end >= len(text):
                break
            start += chunk_size - overlap      # advance by (size − overlap)

    print(f"      → {len(chunks)} chunks created.")
    return chunks


# ── Step 3 · Generate Gemini embeddings ─────────────────────────────────────

def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 20,
) -> np.ndarray:
    """
    Call Gemini Embedding API in batches.
    Returns float32 array of shape (N, embedding_dim).

    Gemini text-embedding-004 outputs 768-dimensional vectors.
    task_type="RETRIEVAL_DOCUMENT" optimises vectors for asymmetric
    retrieval (query ≠ document style).
    """
    genai.configure(api_key=config.GEMINI_API_KEY)
    all_embeddings: list[list[float]] = []

    print(f"[3/4] Embedding {len(chunks)} chunks via Gemini "
          f"({config.EMBEDDING_MODEL}) …")

    for i in tqdm(range(0, len(chunks), batch_size), unit="batch"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Retry loop — Gemini API occasionally rate-limits free-tier requests
        for attempt in range(5):
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=texts,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                all_embeddings.extend(result["embedding"])
                break
            except Exception as exc:
                wait = 2 ** attempt
                tqdm.write(f"  [warn] Batch {i//batch_size} attempt {attempt+1} "
                           f"failed: {exc}. Retrying in {wait}s …")
                time.sleep(wait)
        else:
            sys.exit(f"[ERROR] Embedding failed after 5 attempts for batch {i}.")

    vectors = np.array(all_embeddings, dtype=np.float32)
    print(f"      → Embedding matrix shape: {vectors.shape}")
    return vectors


# ── Step 4 · Build & persist FAISS index ─────────────────────────────────────

def build_faiss_index(
    vectors: np.ndarray,
    chunks: list[dict[str, Any]],
) -> None:
    """
    Index type: IndexFlatIP (exact inner-product / cosine similarity).
    Gemini embeddings are L2-normalised, so inner product == cosine similarity.
    For very large corpora (>100 k chunks) consider IndexIVFFlat instead.
    """
    print("[4/4] Building FAISS index …")

    # Normalise so inner-product == cosine similarity
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist index and metadata
    index_dir = Path(config.FAISS_INDEX_PATH).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, config.FAISS_INDEX_PATH)
    print(f"      → FAISS index saved  → {config.FAISS_INDEX_PATH}  "
          f"({index.ntotal} vectors)")

    # Metadata: strip the text to save space; keep it for prompt construction
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "text": c["text"],
        }
        for c in chunks
    ]
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      → Metadata saved     → {config.METADATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    validate_env()

    if not Path(config.PDF_PATH).exists():
        sys.exit(
            f"[ERROR] PDF not found at '{config.PDF_PATH}'.\n"
            "Place the PDF in the project root or update PDF_PATH in config.py."
        )

    pages = extract_pages(config.PDF_PATH)
    chunks = chunk_pages(pages)
    vectors = embed_chunks(chunks)
    build_faiss_index(vectors, chunks)

    print("\n✅ Ingestion complete. Run  python chat.py  to start chatting.\n")


if __name__ == "__main__":
    main()
