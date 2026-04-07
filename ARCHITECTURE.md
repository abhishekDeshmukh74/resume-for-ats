# Architecture

## Overview

Resume for ATS is a full-stack web application that takes a resume (PDF or LaTeX) and a job description, then uses a **LangGraph multi-agent AI pipeline** to produce an ATS-optimised version of the resume while preserving the original file's formatting and layout.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│                                                             │
│   React 19 + Vite (TypeScript)                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │  Upload  │→ │    JD    │→ │ Generate │→ │ Preview  │    │
│   │  Resume  │  │  Input   │  │(loading) │  │Download  │    │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                      fetch /api/*                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (proxied by Vite → :8000)
┌───────────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend (:8000)                    │
│                                                             │
│  POST /api/parse-resume   → parser.py / latex_parser.py     │
│  POST /api/scrape-jd      → scraper.py                      │
│  POST /api/generate-resume→ agents/graph.py (LangGraph)     │
│  GET  /api/pipeline-runs  → db.py (MongoDB)                 │
│                                                             │
└───────────┬─────────────────────────────────┬───────────────┘
            │                                 │
   ┌────────▼────────┐               ┌────────▼────────┐
   │  Groq / Gemini  │               │    MongoDB       │
   │   LLM APIs      │               │  (optional)      │
   └─────────────────┘               └─────────────────┘
```

---

## Frontend

**Stack:** React 19 · Vite 8 · TypeScript · Tailwind CSS 4 · react-router-dom

### Pages & Components

```
src/
├── App.tsx                  # Router: "/" → HomePage, "/info" → InfoPage
├── pages/
│   └── InfoPage.tsx         # Pipeline run inspector (debug view)
├── components/
│   ├── StepIndicator.tsx    # 4-step progress bar
│   ├── ResumeUpload.tsx     # File drag-drop + /api/parse-resume call
│   ├── JDInput.tsx          # JD paste or URL + /api/scrape-jd call
│   └── ResumePreview.tsx    # PDF viewer, ATS score diff, download button
├── api/
│   └── client.ts            # Typed fetch wrappers for all API endpoints
└── types/
    └── resume.ts            # Shared TypeScript interfaces (ResumeData, etc.)
```

### User Flow

```
Step 1 (Upload Resume)
  → ResumeUpload calls POST /api/parse-resume
  → stores resumeText + resumeFileB64 + resumeFileType in App state

Step 2 (Job Description)
  → JDInput accepts pasted text or calls POST /api/scrape-jd for URL
  → on submit, triggers Step 3

Step 3 (Generate — loading)
  → App calls POST /api/generate-resume with all state
  → on success → Step 4

Step 4 (Preview & Download)
  → ResumePreview renders the returned PDF (base64) in an <iframe>
  → displays before/after ATS scores
  → provides a download link for the rewritten file
```

---

## Backend

**Stack:** Python · FastAPI · LangGraph · PyMuPDF · xelatex/pdflatex · Motor (async MongoDB)

See [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) for full backend detail.

### Routers → Services mapping

| Endpoint | Router | Service(s) |
|----------|--------|-----------|
| `POST /api/parse-resume` | `routers/resume.py` | `parser.py`, `latex_parser.py` |
| `POST /api/scrape-jd` | `routers/jd.py` | `scraper.py` |
| `POST /api/generate-resume` | `routers/generate.py` | `agents/graph.py`, `rewriter.py`, `latex_rewriter.py` |
| `GET /api/pipeline-runs` | `routers/pipeline.py` | `db.py` |
| `GET /api/pipeline-runs/{id}` | `routers/pipeline.py` | `db.py` |

### AI Pipeline (LangGraph — 7 sequential agents)

```
resume_text + jd_text
       │
       ▼
┌─────────────────────┐
│ 1. extract_keywords │  → 30–60 categorised JD keywords
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 2. analyse_resume   │  → section map, keyword gaps
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 3. score_before     │  → baseline ATS score
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 4. rewrite_sections │  → raw old→new text replacements
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 5. qa_deduplicate   │  → validated, deduplicated replacements
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 6. score_extract    │  → final ATS score + structured ResumeData
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 7. compile_pdf      │  → apply replacements to original file → PDF bytes
└─────────────────────┘
       │
       ▼
(ResumeData, rewritten_file_b64)
```

All agents share an `AgentState` TypedDict. PDF resumes use PyMuPDF in-place text redaction; LaTeX resumes patch the source and invoke `xelatex`/`pdflatex`.

---

## Data Flow (end-to-end)

```
1. User uploads file
   Browser → multipart POST /api/parse-resume
   Backend extracts text (PyMuPDF or latex_parser), returns { text, html, file_b64, file_type }

2. User submits job description
   Browser → POST /api/scrape-jd (if URL) or uses pasted text directly

3. User clicks Generate
   Browser → POST /api/generate-resume { resume_text, jd_text, resume_file_b64, resume_file_type }
   Backend  → runs 7-agent LangGraph pipeline
            → returns { resume: ResumeData, rewritten_file_b64: string }

4. User previews and downloads
   Browser renders base64 PDF in <iframe>, offers download
```

---

## Infrastructure

| Concern | Solution |
|---------|---------|
| API proxy (dev) | Vite `server.proxy` → `localhost:8000` |
| CORS | FastAPI `CORSMiddleware` (configured in `main.py`) |
| LLM provider | `LLM_PROVIDER` env var — `groq` (default) or `gemini` |
| PDF processing | PyMuPDF (`fitz`) for read/write; `xelatex` for LaTeX compilation |
| Persistence | MongoDB via Motor — optional; pipeline tracking degrades gracefully if unavailable |
| Deployment | `render.yaml` for Render.com (backend as web service) |

---

## Key Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (or Gemini) | Groq API key for LLM calls |
| `GEMINI_API_KEY` | Yes (or Groq) | Google Gemini API key |
| `LLM_PROVIDER` | No | `groq` (default) or `gemini` |
| `MONGODB_URL` | No | MongoDB connection string for pipeline run tracking |

---

## Local Development

```bash
# Backend (port 8000)
.venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000

# Frontend (port 5173, proxies /api → :8000)
cd frontend
npm run dev
```

Or use `make dev` to start both simultaneously.
