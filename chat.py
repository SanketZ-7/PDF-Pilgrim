"""
chat.py — Interactive RAG chatbot for "Pilgrims of the Stars".

Usage:
    python chat.py

Commands inside the chat:
    /quit  or  /exit  — end the session
    /sources          — show source chunks from the last answer
    /reset            — clear conversation history
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any

import google.generativeai as genai

import config
from retriever import retrieve, build_context


# ── Validate environment ──────────────────────────────────────────────────────

def validate_env() -> None:
    if not config.GEMINI_API_KEY:
        sys.exit(
            "\n[ERROR] GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export GEMINI_API_KEY='your-key-here'   # Linux / macOS\n"
            "  set GEMINI_API_KEY=your-key-here         # Windows CMD\n"
        )


# ── Gemini generation ─────────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Build the RAG prompt and call the Gemini chat model.
    Uses a stateless call (no multi-turn history sent to Gemini) because
    context is always re-retrieved fresh per turn, which keeps answers
    grounded.  Conversation display history is maintained locally for UX.
    """
    prompt = config.SYSTEM_PROMPT.format(context=context, question=question)

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.CHAT_MODEL)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,        # low temperature → factual, grounded answers
            max_output_tokens=8192,
        ),
    )
    return response.text.strip()


# ── Pretty printing ───────────────────────────────────────────────────────────

WIDTH = 80

def _hr(char: str = "─") -> None:
    print(char * WIDTH)

def _wrap(text: str, prefix: str = "") -> None:
    for line in text.splitlines():
        if line.strip():
            print(textwrap.fill(line, width=WIDTH, initial_indent=prefix,
                                subsequent_indent=" " * len(prefix)))
        else:
            print()

def _show_sources(sources: list[dict[str, Any]]) -> None:
    _hr()
    print("📖  SOURCES")
    _hr("·")
    for i, s in enumerate(sources, start=1):
        print(f"  [{i}] Page {s['page']}  (similarity: {s['score']:.3f})")
        snippet = s["text"][:200].replace("\n", " ")
        print(f'      "{snippet}..."')
    _hr()


# ── Main chat loop ────────────────────────────────────────────────────────────

def main() -> None:
    validate_env()

    banner = textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║      📚  PILGRIMS OF THE STARS — RAG Chatbot  (Gemini + FAISS)         ║
    ║      Ask anything about the book.  Type /quit to exit.                 ║
    ║      Commands: /sources · /reset · /quit                               ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    print(banner)

    history: list[dict[str, str]] = []   # local display history
    last_sources: list[dict[str, Any]] = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye! 🙏")
            break

        if user_input.lower() == "/sources":
            if last_sources:
                _show_sources(last_sources)
            else:
                print("  (No sources yet — ask a question first.)")
            continue

        if user_input.lower() == "/reset":
            history.clear()
            last_sources.clear()
            print("  ✓ Conversation history cleared.")
            continue

        # ── RAG pipeline ─────────────────────────────────────────────────────
        print("  -> Retrieving relevant passages ...", end="\r")

        try:
            sources = retrieve(user_input, top_k=config.TOP_K)
        except Exception as exc:
            print(f"\n[ERROR] Retrieval failed: {exc}")
            continue

        context = build_context(sources)

        print("  -> Generating answer ...             ", end="\r")

        try:
            answer = generate_answer(user_input, context)
        except Exception as exc:
            print(f"\n[ERROR] Generation failed: {exc}")
            continue

        # ── Display ───────────────────────────────────────────────────────────
        print(" " * 50, end="\r")    # clear spinner line
        _hr()
        _wrap(answer, prefix="Bot > ")
        _hr()

        # ── Update state ──────────────────────────────────────────────────────
        last_sources = sources
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        print("  (Type /sources to see retrieved passages)")
        print()


if __name__ == "__main__":
    main()
