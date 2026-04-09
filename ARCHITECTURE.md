# Architecture

## Overview

Resume for ATS is a full-stack web application that takes a resume (PDF or LaTeX) and a job description, then uses a **LangGraph multi-agent AI pipeline** with **parallel rewriting and a conditional refinement loop** to produce an ATS-optimised version of the resume while preserving the original file's formatting and layout.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           Browser                                │
│                                                                  │
│   React 19 + Vite (TypeScript) + Zustand + Tailwind CSS 4       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐│
│   │  Upload  │→│    JD    │→│ Generate │→│  Review  │→│Preview││
│   │  Resume  │ │  Input   │ │(loading) │ │  Diffs   │ │  PDF  ││
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘│
│                      fetch /api/*  +  SSE streams                │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (proxied by Vite → :8000)
┌────────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend (:8000)                        │
│                                                                  │
│  POST /api/parse-resume         → parser.py / latex_parser.py    │
│  POST /api/scrape-jd            → scraper.py                     │
│  POST /api/generate-resume      → agents/graph.py (LangGraph)    │
│  POST /api/generate-resume-stream → SSE streaming pipeline       │
│  POST /api/preview              → preview pipeline (no PDF)      │
│  POST /api/preview-stream       → SSE streaming preview          │
│  POST /api/confirm              → apply replacements → PDF       │
│  POST /api/generate-cover-letter→ cover_letter.py                │
│  GET  /api/pipeline-runs        → db.py (MongoDB)                │
│  GET  /api/pipeline-runs/{id}/pdf → compiled PDF download        │
│                                                                  │
└────────────┬─────────────────────────────────┬───────────────────┘
             │                                 │
   ┌─────────▼──────────┐             ┌────────▼────────┐
   │  LLM Provider      │             │    MongoDB       │
   │  (via litellm)     │             │  (optional)      │
   │  Groq · Gemini     │             └─────────────────┘
   │  OpenAI · Anthropic│
   │  Ollama · DeepSeek │
   │  OpenRouter        │
   └────────────────────┘
```

---

## Frontend

**Stack:** React 19 · Vite 8 · TypeScript · Tailwind CSS 4 · Zustand · react-router-dom

### Pages & Components

```
src/
├── App.tsx                  # Router: "/" → HomePage, "/info" → InfoPage
├── pages/
│   └── InfoPage.tsx         # Pipeline run inspector (debug view)
├── components/
│   ├── StepIndicator.tsx    # 5-step progress bar
│   ├── ResumeUpload.tsx     # File drag-drop + /api/parse-resume call
│   ├── JDInput.tsx          # JD paste or URL + /api/scrape-jd call
│   ├── DiffPreview.tsx      # Side-by-side old→new replacement diffs
│   ├── ResumePreview.tsx    # PDF viewer, ATS score diff, download button
│   └── CoverLetterPanel.tsx # Cover letter + LinkedIn message generator
├── store/
│   └── appStore.ts          # Zustand store with localStorage persistence
├── api/
│   └── client.ts            # Typed fetch wrappers + SSE stream helpers
└── types/
    └── resume.ts            # Shared TypeScript interfaces (ResumeData, etc.)
```

### User Flow (5-step wizard)

```
Step 1 (Upload Resume)
  → ResumeUpload calls POST /api/parse-resume
  → stores resumeText + fileB64 + fileType + fileName in Zustand store

Step 2 (Job Description)
  → JDInput accepts pasted text or calls POST /api/scrape-jd for URL
  → on submit, triggers Step 3

Step 3 (Generate — loading with SSE agent progress)
  → App calls POST /api/preview-stream via SSE
  → streams per-agent progress events (agent name, scores, replacement counts)
  → on complete → Step 4

Step 4 (Review Diffs)
  → DiffPreview shows proposed old→new replacements
  → user can review before confirming
  → on confirm → POST /api/confirm → Step 5

Step 5 (Preview & Download)
  → ResumePreview renders the returned PDF (base64) in an <iframe>
  → displays before/after ATS scores and matched keywords
  → provides download link + optional cover letter generation
```

---

## Backend

**Stack:** Python · FastAPI · LangGraph · litellm · PyMuPDF · xelatex/pdflatex · pymongo

See [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) for full backend detail.

### Routers → Services mapping

| Endpoint | Router | Service(s) |
|----------|--------|-----------|
| `POST /api/parse-resume` | `routers/resume.py` | `parser.py`, `latex_parser.py` |
| `POST /api/scrape-jd` | `routers/jd.py` | `scraper.py` |
| `POST /api/generate-resume` | `routers/generate.py` | `agents/graph.py` |
| `POST /api/generate-resume-stream` | `routers/stream.py` | `agents/graph.py` (SSE) |
| `POST /api/preview` | `routers/generate.py` | `agents/graph.py` (no PDF) |
| `POST /api/preview-stream` | `routers/generate.py` | `agents/graph.py` (SSE, no PDF) |
| `POST /api/confirm` | `routers/generate.py` | `agents/pdf_compiler.py` |
| `POST /api/generate-cover-letter` | `routers/generate.py` | `agents/cover_letter.py` |
| `GET /api/pipeline-runs` | `routers/pipeline.py` | `db.py` |
| `GET /api/pipeline-runs/{id}` | `routers/pipeline.py` | `db.py` |
| `GET /api/pipeline-runs/{id}/pdf` | `routers/pipeline.py` | `db.py` |
| `GET /api/pipeline-runs/status` | `routers/pipeline.py` | `db.py` |
| `GET /api/health` | `main.py` | — |

### AI Pipeline (LangGraph — parallel rewriters + conditional refinement)

```
resume_text + jd_text
       │
       ▼
┌──────────────────────┐
│ 1. extract_keywords  │  → 30–60 categorised JD keywords
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. analyse_resume    │  → section map, keyword gaps
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. score_before      │  → baseline ATS score (algorithmic + LLM)
└──────────┬───────────┘
           ▼
   ┌───────┼───────┐        ← parallel fan-out
   ▼       ▼       ▼
┌──────┐┌──────┐┌──────┐
│skills││summ. ││exper.│    → 3 section-specific rewriters (parallel)
└──┬───┘└──┬───┘└──┬───┘
   └───────┼───────┘        ← fan-in (all three must complete)
           ▼
┌──────────────────────┐
│ 5. qa_deduplicate    │  → validated, deduplicated replacements
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. score_extract     │  → ATS score + structured ResumeData
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  score≥90    score<90 & pass==0
     │            │
     │    ┌───────▼───────┐
     │    │refine_rewrite │  → targeted missing-keyword injection
     │    └───────┬───────┘
     │            ▼
     │    ┌───────────────┐
     │    │  refine_qa    │  → validate refinement replacements
     │    └───────┬───────┘
     │            │
     └─────┬──────┘
           ▼
┌──────────────────────┐
│ 7. compile_pdf       │  → apply replacements to original file → PDF bytes
└──────────────────────┘
           │
           ▼
  (ResumeData, rewritten_file_b64)
```

All agents share an `AgentState` TypedDict. PDF resumes use PyMuPDF in-place text redaction; LaTeX resumes patch the source and invoke `xelatex`/`pdflatex`.

---

## Data Flow (end-to-end)

### Full Generate Flow

```
1. User uploads file
   Browser → multipart POST /api/parse-resume
   Backend extracts text (PyMuPDF or latex_parser), returns { text, html, file_b64, file_type }

2. User submits job description
   Browser → POST /api/scrape-jd (if URL) or uses pasted text directly

3. User clicks Generate
   Browser → POST /api/generate-resume-stream (SSE)
   Backend → streams per-agent progress events
           → returns { resume: ResumeData, rewritten_file_b64: string }

4. User previews and downloads
   Browser renders base64 PDF in <iframe>, offers download
```

### Preview / Confirm Flow (two-phase)

```
1–2. Same as above

3. User clicks Generate (preview mode)
   Browser → POST /api/preview-stream (SSE, no PDF compilation)
   Backend → streams progress, returns { replacements, scores, keywords }

4. User reviews old→new diffs
   DiffPreview component shows proposed changes

5. User confirms
   Browser → POST /api/confirm { replacements, resume_file_b64, ... }
   Backend → compile_pdf only → returns { rewritten_file_b64 }
```

---

## Infrastructure

| Concern | Solution |
|---------|---------|
| API proxy (dev) | Vite `server.proxy` → `localhost:8000` |
| CORS | FastAPI `CORSMiddleware` (configured in `main.py` via `ALLOWED_ORIGINS`) |
| LLM provider | `LLM_PROVIDER` env var — 7 providers via **litellm** |
| PDF processing | PyMuPDF (`fitz`) for read/write; `xelatex`/`pdflatex`/`lualatex` for LaTeX |
| ATS scoring | Algorithmic (`rapidfuzz`) — exact, synonym, fuzzy matching with section weighting |
| Persistence | MongoDB via **pymongo** (sync) — optional; pipeline tracking degrades gracefully |
| State management | Zustand with `persist` middleware (localStorage, excludes large blobs) |
| Deployment | `render.yaml` for Render.com (backend as web service) |

---

## Key Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes* | Groq API key for LLM calls |
| `GROQ_API_KEYS` | No | Comma-separated Groq keys for round-robin fallback |
| `GEMINI_API_KEY` | Yes* | Google Gemini API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
| `DEEPSEEK_API_KEY` | Yes* | DeepSeek API key |
| `OPENROUTER_API_KEY` | Yes* | OpenRouter API key |
| `LLM_PROVIDER` | No | `groq` (default), `gemini`, `openai`, `anthropic`, `ollama`, `deepseek`, `openrouter` |
| `MONGODB_URL` | No | MongoDB connection string for pipeline run tracking |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (default: `http://localhost:5173`) |
| `LATEX_COMPILER_PATH` | No | Override path to xelatex/pdflatex binary |

\* At least one API key is required, matching the selected `LLM_PROVIDER`.

---

## Local Development

```bash
# Backend (port 8000)
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Frontend (port 5173, proxies /api → :8000)
cd frontend
npm run dev

# Or use Make
make dev       # starts both
```

Or use `make dev` to start both simultaneously.
