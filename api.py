"""
api.py — Optional FastAPI REST interface for the RAG chatbot.

Install extra dependency:  pip install fastapi uvicorn
Run:  uvicorn api:app --reload --port 8000

Endpoints:
  POST /chat        → { "question": "..." }  →  { "answer": "...", "sources": [...] }
  GET  /healthz     → { "status": "ok" }
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from retriever import retrieve, build_context
from chat import generate_answer, validate_env


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pilgrims of the Stars — RAG API",
    description="Retrieval-Augmented Generation over the book by Dilip Kumar Roy & Indira Devi.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Natural-language question about the book.")
    top_k: int = Field(config.TOP_K, ge=1, le=20,
                       description="Number of chunks to retrieve.")


class SourceChunk(BaseModel):
    chunk_id: int
    page: int
    text: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    validate_env()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        sources = retrieve(req.question, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    context = build_context(sources)

    try:
        answer = generate_answer(req.question, context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")

    return ChatResponse(
        question=req.question,
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
    )
