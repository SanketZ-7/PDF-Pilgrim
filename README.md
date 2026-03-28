# 📚 Pilgrims of the Stars — RAG Chatbot

> Retrieval-Augmented Generation over *Pilgrims of the Stars: Autobiography of Two Yogis*  
> by Dilip Kumar Roy & Indira Devi  
> Powered by **Google Gemini** (embeddings + LLM) and **FAISS** (vector search)

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                       │
│  (run once — offline)                                            │
│                                                                  │
│  PDF  ──pypdf──▶  Page texts  ──slide-window──▶  Chunks         │
│                                                     │            │
│                                          Gemini Embedding API    │
│                                          (text-embedding-004)    │
│                                                     │            │
│                                          float32 vectors (768-d) │
│                                                     │            │
│                                          FAISS IndexFlatIP       │
│                                          ├─ index.faiss          │
│                                          └─ metadata.json        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│  (real-time — per user message)                                  │
│                                                                  │
│  User query                                                      │
│      │                                                           │
│      ▼                                                           │
│  Gemini Embedding (RETRIEVAL_QUERY)                              │
│      │                                                           │
│      ▼                                                           │
│  FAISS.search(query_vec, top_k=5)  ──▶  top-k chunk IDs         │
│      │                                                           │
│      ▼                                                           │
│  metadata.json lookup  ──▶  chunk texts + page numbers          │
│      │                                                           │
│      ▼                                                           │
│  Prompt assembly                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │ SYSTEM PROMPT + CONTEXT (excerpts) + QUESTION │              │
│  └───────────────────────────────────────────────┘              │
│      │                                                           │
│      ▼                                                           │
│  Gemini 3.1 Pro Preview (generate_content, temp=0.2)            │
│      │                                                           │
│      ▼                                                           │
│  Grounded answer with page citations  ──▶  User                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag_chatbot/
├── config.py          # All tunable parameters (chunk size, models, paths)
├── ingest.py          # PDF extraction → chunking → embedding → FAISS index
├── retriever.py       # FAISS loading, query embedding, top-k retrieval
├── chat.py            # Interactive CLI chatbot
├── api.py             # Optional FastAPI REST interface
├── requirements.txt
├── README.md
├── Pilgrims_of_the_Stars.pdf   ← place your PDF here
└── faiss_index/                ← created by ingest.py
    ├── index.faiss
    └── metadata.json
```

---

## Chunking Strategy — Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `CHUNK_SIZE` | **800 chars** (~150 words) | Narrative prose benefits from medium-sized chunks that capture a complete thought or scene. Too small → loses semantic coherence. Too large → retrieval becomes noisy and context window fills up. |
| `CHUNK_OVERLAP` | **150 chars** (~30 words, 19 %) | Cross-boundary context is preserved; a spiritual concept or character name mentioned at the end of one chunk isn't orphaned. |
| Chunking scope | **Within-page** | Each chunk is tagged with an unambiguous page number, enabling precise citations. |

---

## Setup Instructions

### 1. Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Get API key"** → **"Create API key"**
3. Copy the key

### 2. Set the Environment Variable

```bash
# Linux / macOS
export GEMINI_API_KEY="AIza..."

# Windows (Command Prompt)
set GEMINI_API_KEY=AIza...

# Windows (PowerShell)
$env:GEMINI_API_KEY="AIza..."

# Persist across sessions — add to ~/.bashrc or ~/.zshrc:
echo 'export GEMINI_API_KEY="AIza..."' >> ~/.bashrc
source ~/.bashrc
```

> ⚠️ Never commit your API key to version control.  
> Use a `.env` file + `python-dotenv` for team projects (see below).

### Optional: `.env` file approach

```bash
# .env  (add this file to .gitignore!)
GEMINI_API_KEY=AIza...
```

Add to the top of `config.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

Install: `pip install python-dotenv`

---

### 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Place the PDF

Copy `Pilgrims_of_the_Stars.pdf` into the `rag_chatbot/` directory,  
or update `PDF_PATH` in `config.py` to point to its location.

### 5. Run Ingestion (one-time)

```bash
python ingest.py
```

Expected output:
```
[1/4] Extracting text from 384 pages … 100%|████████| 384/384
      → 371 pages contained extractable text.
[2/4] Chunking (size=800 chars, overlap=150 chars) …
      → 1842 chunks created.
[3/4] Embedding 1842 chunks via Gemini (models/text-embedding-004) …
      → Embedding matrix shape: (1842, 768)
[4/4] Building FAISS index …
      → FAISS index saved  → faiss_index/index.faiss  (1842 vectors)
      → Metadata saved     → faiss_index/metadata.json

✅ Ingestion complete. Run  python chat.py  to start chatting.
```

### 6. Start Chatting

```bash
python chat.py
```

---

## Example Session

```
You › Who is Dilip Kumar Roy?

Bot › Dilip Kumar Roy was a celebrated Indian musician, singer, and spiritual
     seeker, and one of the two co-authors of this autobiography. He was a
     devoted disciple of Sri Aurobindo and later of Swami Ramdas. The book
     describes his inner journey from artistic life to spiritual realization
     (p. 1, p. 14).

  (Type /sources to see retrieved passages)

You › /sources

────────────────────────────────────────────────────────────────────────────────
📖  SOURCES
·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
  [1] Page 1  (similarity: 0.891)
      "PILGRIMS OF THE STARS AUTOBIOGRAPHY OF TWO YOGIS Dilip Kumar Roy AND …"
  [2] Page 14  (similarity: 0.843)
      "…his early musical career and eventual turn towards spirituality…"
────────────────────────────────────────────────────────────────────────────────
```

---

## Streamlit UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Features
- 🔑 Enter your Gemini API key directly in the sidebar (no env var needed)
- 💬 Chat history with styled user / assistant bubbles  
- 📖 Expandable **source passages** panel with page numbers and similarity scores  
- ✦ Example question buttons to get started instantly  
- ⚙️ Live `top-k` slider to tune retrieval depth  
- 🗑️ One-click conversation reset  
- Index status indicator (green = ready / yellow = run `ingest.py`)

---

## REST API (Optional)

```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the significance of kirtan in the book?"}'
```

Response:
```json
{
  "question": "What is the significance of kirtan in the book?",
  "answer": "Kirtan (devotional singing) is presented as a central ...",
  "sources": [
    { "chunk_id": 312, "page": 87, "text": "...", "score": 0.921 },
    ...
  ]
}
```

---

## Configuration Reference (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | env var | Your Google AI Studio API key |
| `EMBEDDING_MODEL` | `text-embedding-004` | Gemini embedding model (768-dim) |
| `CHAT_MODEL` | `gemini-3.1-pro-preview` | Gemini generative model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `FAISS_INDEX_PATH` | `faiss_index/index.faiss` | Where index is saved |
| `METADATA_PATH` | `faiss_index/metadata.json` | Where chunk metadata is saved |
| `PDF_PATH` | `Pilgrims_of_the_Stars.pdf` | Input PDF path |

---

## Design Decisions

### Why FAISS `IndexFlatIP` (not IVF)?
The book produces ~1 800 chunks. `IndexFlatIP` does exact cosine search in milliseconds at this scale. For > 100 k chunks, switch to `IndexIVFFlat` with `nlist=256`.

### Why `temperature=0.2`?
Low temperature makes Gemini produce factual, grounded answers rather than creative elaborations — critical for a book-grounded chatbot.

### Why within-page chunking?
Splitting across page boundaries makes page-number attribution ambiguous. Keeping chunks within pages gives clean, citable sources.

### Why asymmetric embeddings (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`)?
Queries are short and telegraphic; passages are full prose. Gemini's asymmetric task types optimise the embedding space for this mismatch, improving recall significantly over symmetric embedding.
