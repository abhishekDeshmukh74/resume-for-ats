# Backend Architecture

## Overview

FastAPI backend for the Resume for ATS resume tailor. Receives a PDF or LaTeX resume and job description, runs a **LangGraph multi-agent pipeline** with **3 parallel section rewriters and a conditional refinement loop** to rewrite resume content for ATS optimisation, and returns a modified PDF preserving the original layout.

## Directory Structure

```
backend/
├── main.py              # FastAPI app, CORS, router registration, JSON logging
├── models.py            # Pydantic models (request/response schemas)
├── routers/
│   ├── resume.py        # POST /api/parse-resume — PDF + LaTeX upload + text extraction
│   ├── jd.py            # POST /api/scrape-jd — URL → JD text scraper
│   ├── generate.py      # POST /api/generate-resume, /preview, /preview-stream, /confirm, /generate-cover-letter
│   ├── stream.py        # POST /api/generate-resume-stream — SSE full pipeline streaming
│   └── pipeline.py      # GET  /api/pipeline-runs — run list, detail, PDF download, status
└── services/
    ├── parser.py         # PDF text extraction (PyMuPDF, span-dict HTML)
    ├── latex_parser.py   # LaTeX (.tex) text extraction
    ├── scraper.py        # URL scraping for job descriptions
    ├── rewriter.py       # In-place PDF text replacement (search_for + redact)
    ├── latex_rewriter.py # LaTeX source patching + compilation (xelatex)
    ├── db.py             # MongoDB persistence for pipeline run tracking (pymongo, sync)
    └── agents/           # LangGraph multi-agent pipeline
        ├── __init__.py             # Re-exports generate_resume()
        ├── state.py                # AgentState TypedDict (shared pipeline state)
        ├── llm.py                  # Multi-provider LLM (litellm) + parse_llm_json()
        ├── keyword_extractor.py    # Agent 1: Extract JD keywords
        ├── resume_analyser.py      # Agent 2: Section analysis + gap identification
        ├── scorer.py               # Agents 3 & 6: ATS scoring (algorithmic + LLM) + extraction
        ├── keyword_matcher.py      # Algorithmic scoring engine (rapidfuzz, synonyms, section weights)
        ├── skills_rewriter.py      # Agent 4a: Skills section rewriter (parallel)
        ├── summary_rewriter.py     # Agent 4b: Summary section rewriter (parallel)
        ├── experience_rewriter.py  # Agent 4c: Experience bullets rewriter (parallel)
        ├── rewriter_agent.py       # Legacy monolithic rewriter (unused in graph, kept as reference)
        ├── qa_agent.py             # Agent 5: Validate, deduplicate, clean AI phrases
        ├── refinement_agent.py     # Agent 6b: Targeted second-pass keyword injection
        ├── pdf_compiler.py         # Agent 7: Apply replacements + compile PDF
        ├── cover_letter.py         # Standalone: cover letter + LinkedIn message generator
        └── graph.py                # StateGraph wiring, pipeline tracking, public API
```

## Request Flow (Generate — full pipeline)

```
Frontend                    Backend
   │                          │
   ├─ POST /api/generate ────►│
   │  { resume_text,          │
   │    jd_text,              │
   │    resume_file_b64 }     │
   │                          ├─ agents.generate_resume()
   │                          │   → LangGraph pipeline:
   │                          │     1. extract_keywords  → JD keywords
   │                          │     2. analyse_resume    → gap analysis
   │                          │     3. score_before      → baseline ATS score
   │                          │     4. rewrite_skills ┐
   │                          │        rewrite_summary├ (parallel) → raw replacements
   │                          │        rewrite_experience┘
   │                          │     5. qa_deduplicate    → validated replacements
   │                          │     6. score_extract     → final ATS score + structured data
   │                          │     [if score<90: refine_rewrite → refine_qa → score again]
   │                          │     7. compile_pdf       → apply replacements → PDF
   │                          │   → (ResumeData, compiled_pdf_b64)
   │                          │
   │◄─ { resume, b64_pdf } ──┤
```

## Request Flow (Preview / Confirm — two-phase)

```
Frontend                       Backend
   │                             │
   ├─ POST /api/preview-stream ─►│  (SSE, no PDF compilation)
   │                             ├─ runs pipeline agents 1–6
   │◄─ SSE: agent_complete ──────┤  (per-agent progress events)
   │◄─ SSE: complete ───────────┤  → { replacements, scores, keywords }
   │                             │
   │ User reviews diffs          │
   │                             │
   ├─ POST /api/confirm ────────►│
   │  { replacements,            ├─ compile_pdf only
   │    resume_file_b64 }        │
   │◄─ { rewritten_file_b64 } ──┤
```

## LangGraph Pipeline

The AI logic uses 3 parallel section rewriters and a conditional refinement loop:

| # | Agent | Node Name(s) | Purpose |
|---|-------|---------------|---------|
| 1 | Keyword Extractor | `extract_keywords` | Extract 30–60 JD keywords, categorised into required/preferred |
| 2 | Resume Analyser | `analyse_resume` | Map resume sections, find keyword gaps |
| 3 | Pre-Rewrite Scorer | `score_before` | Baseline ATS score (algorithmic via rapidfuzz + LLM; takes conservative min) |
| 4 | Section Rewriters | `rewrite_skills`, `rewrite_summary`, `rewrite_experience` | 3 parallel agents: each generates section-specific old→new replacements |
| 5 | QA Agent | `qa_deduplicate` | Validate old text accuracy, deduplicate keywords, clean AI phrases |
| 6 | Final Scorer | `score_extract` | Algorithmic ATS score + structured data extraction via LLM |
| 6b | Refinement Writer | `refine_rewrite` | (conditional) Inject still-missing keywords if score < 90 |
| 6c | Refinement QA | `refine_qa` | (conditional) Validate refinement replacements |
| 7 | PDF Compiler | `compile_pdf` | Apply replacements to original file, produce PDF |

All agents share an `AgentState` TypedDict. See `backend/services/agents/AGENTS.md` for details.

## Pipeline Run Tracking

Every pipeline execution is tracked in MongoDB (best-effort — failures never break the pipeline):

1. **`graph.py`** creates a pipeline run in MongoDB at the start via `db.create_pipeline_run()`
2. Each agent is wrapped by `_tracked()`, which records:
   - Agent name, execution duration (ms)
   - Input state summary (relevant keys only, via `_AGENT_INPUT_KEYS` mapping)
   - Output data (serialised, truncated for storage)
3. On completion: `db.complete_pipeline_run()` saves ATS scores, replacement count, and name
4. Compiled PDF bytes are stored as BSON Binary via `db.save_compiled_pdf()`
5. On failure: `db.fail_pipeline_run()` saves the error message
6. Run ID is stored in a `contextvars.ContextVar` for thread-safe tracking

**Inspection endpoints**:
- `GET /api/pipeline-runs` — list runs with summary (status, timestamps, agent names)
- `GET /api/pipeline-runs/{id}` — full detail with per-agent timing, I/O data
- `GET /api/pipeline-runs/{id}/pdf` — download the compiled PDF as binary
- `GET /api/pipeline-runs/status` — check MongoDB connectivity

## Key Data Models (models.py)

- **`TextReplacement`**: `{old: str, new: str}` — a single find-and-replace pair
- **`ResumeData`**: Full structured resume + `replacements: list[TextReplacement]` + `ats_score` + `ats_score_before`
- **`GenerateRequest`**: `{resume_text, jd_text, resume_file_b64, resume_file_type}`
- **`GenerateResponse`**: `{resume: ResumeData, rewritten_file_b64: str}`
- **`PreviewRequest`**: `{resume_text, jd_text}` — runs pipeline without PDF compilation
- **`PreviewResponse`**: `{replacements, ats_score_before, ats_score, matched_keywords, still_missing_keywords}`
- **`ConfirmRequest`**: `{resume_text, replacements, resume_file_b64, resume_file_type}` — apply and compile
- **`ConfirmResponse`**: `{rewritten_file_b64: str}`
- **`CoverLetterRequest`**: `{resume_text, jd_text, company_name?}`
- **`CoverLetterResponse`**: `{cover_letter, suggested_job_title, linkedin_message}`

## Critical Path: Text Matching

The most fragile part of the pipeline is matching AI-generated `old` strings against PDF/LaTeX text:

### PDF Flow
1. **`parser.py`** extracts text using `page.get_text("text")` with normalisation (spacing artefact fixes).
2. **Section rewriters** (Agents 4a/4b/4c) return `old` strings that should be verbatim substrings.
3. **`agents/qa_agent.py`** (Agent 5) validates that each `old` string exists in the original resume text.
4. **`rewriter.py`** uses `page.search_for()` to locate `old` text in the PDF, then redacts and re-inserts.

### LaTeX Flow
1. **`latex_parser.py`** strips LaTeX commands to produce plain text.
2. Section rewriters work identically on the plain text.
3. **`latex_rewriter.py`** applies replacements to the `.tex` source (with flexible pattern matching for line-wrapping and LaTeX escapes), then compiles to PDF via xelatex.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `groq` (default), `gemini`, `openai`, `anthropic`, `ollama`, `deepseek`, `openrouter` |
| `GROQ_API_KEY` | Yes* | Groq API key for AI calls |
| `GROQ_API_KEYS` | No | Comma-separated Groq keys for round-robin fallback |
| `GEMINI_API_KEY` | Yes* | Google Gemini API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
| `DEEPSEEK_API_KEY` | Yes* | DeepSeek API key |
| `OPENROUTER_API_KEY` | Yes* | OpenRouter API key |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `GROQ_MODEL` | No | Groq model name (default: `llama-3.3-70b-versatile`) |
| `GEMINI_MODEL` | No | Gemini model name (default: `gemini-2.0-flash`) |
| `OPENAI_MODEL` | No | OpenAI model name (default: `gpt-4o`) |
| `ANTHROPIC_MODEL` | No | Anthropic model name (default: `claude-sonnet-4-20250514`) |
| `OLLAMA_MODEL` | No | Ollama model name (default: `llama3.1`) |
| `DEEPSEEK_MODEL` | No | DeepSeek model name (default: `deepseek-chat`) |
| `OPENROUTER_MODEL` | No | OpenRouter model name (default: `meta-llama/llama-3-70b`) |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (default: `http://localhost:5173`) |
| `MONGODB_URL` | No | MongoDB connection string for pipeline run tracking |
| `LATEX_COMPILER_PATH` | No | Override path to xelatex/pdflatex binary |

\* At least one API key is required, matching the selected `LLM_PROVIDER`.

## Running

```bash
uvicorn backend.main:app --reload --port 8000
```
