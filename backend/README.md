# Backend — Resume for ATS API

FastAPI backend that parses PDF and LaTeX resumes, scrapes job descriptions, runs a **LangGraph multi-agent pipeline** (3 parallel section rewriters + conditional refinement loop) to tailor resume content for ATS, and produces a final PDF with updated text.

## Architecture

```
backend/
├── main.py              # FastAPI app, CORS, router registration, JSON logging
├── models.py            # Pydantic request/response schemas
├── routers/
│   ├── resume.py        # POST /api/parse-resume (PDF + LaTeX)
│   ├── jd.py            # POST /api/scrape-jd
│   ├── generate.py      # POST /api/generate-resume, /preview, /preview-stream, /confirm, /generate-cover-letter
│   ├── stream.py        # POST /api/generate-resume-stream (SSE full pipeline)
│   └── pipeline.py      # GET  /api/pipeline-runs (list, detail, PDF download, status)
└── services/
    ├── parser.py         # PDF text + HTML extraction (PyMuPDF)
    ├── latex_parser.py   # LaTeX (.tex) text extraction
    ├── scraper.py        # URL → plain text (httpx + BeautifulSoup)
    ├── rewriter.py       # In-place PDF text replacement (PyMuPDF)
    ├── latex_rewriter.py # LaTeX source patching + compilation
    ├── db.py             # MongoDB persistence for pipeline run tracking (pymongo, sync)
    └── agents/           # LangGraph multi-agent pipeline
        ├── graph.py                # StateGraph wiring + pipeline tracking + public API
        ├── state.py                # Shared AgentState TypedDict
        ├── llm.py                  # Multi-provider LLM (litellm) + JSON parser
        ├── keyword_extractor.py    # Agent 1: JD keyword extraction
        ├── resume_analyser.py      # Agent 2: Section analysis + gap identification
        ├── scorer.py               # Agents 3 & 6: ATS scoring + structured extraction
        ├── keyword_matcher.py      # Algorithmic scoring (rapidfuzz, synonyms, section weights)
        ├── skills_rewriter.py      # Agent 4a: Skills section rewriter (parallel)
        ├── summary_rewriter.py     # Agent 4b: Summary section rewriter (parallel)
        ├── experience_rewriter.py  # Agent 4c: Experience bullets rewriter (parallel)
        ├── qa_agent.py             # Agent 5: Validation + deduplication + AI phrase cleanup
        ├── refinement_agent.py     # Agent 6b: Targeted second-pass keyword injection
        ├── pdf_compiler.py         # Agent 7: Apply replacements + compile PDF
        └── cover_letter.py         # Standalone: cover letter + LinkedIn message generator
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `groq` (default), `gemini`, `openai`, `anthropic`, `ollama`, `deepseek`, `openrouter` |
| `GROQ_API_KEY` | Yes* | API key from [console.groq.com](https://console.groq.com/) |
| `GROQ_API_KEYS` | No | Comma-separated Groq keys for round-robin fallback |
| `GEMINI_API_KEY` | Yes* | API key from [aistudio.google.com](https://aistudio.google.com/) |
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

## API Endpoints

### `POST /api/parse-resume`

Upload a PDF or LaTeX (.tex) file. Returns extracted text, styled HTML preview, and the original file as base64.

**Request**: `multipart/form-data` with field `file` (PDF or `.tex`, max 10 MB)

**Response** (`ParsedResumeResponse`):
```json
{
  "text": "plain text extracted from PDF",
  "html": "<div class='pdf-document'>…styled HTML…</div>",
  "file_b64": "base64-encoded original PDF bytes",
  "file_type": "pdf"
}
```

**Errors**: `415` unsupported type · `413` file too large · `422` parse failure or empty text

> For PDF uploads, text is extracted via PyMuPDF and HTML is generated from span dicts.
> For LaTeX uploads, text is produced by stripping LaTeX commands; the raw source is preserved for rewriting.

---

### `POST /api/scrape-jd`

Fetch a URL and extract the main text content (job description).

**Request**:
```json
{ "url": "https://example.com/job-posting" }
```

**Response** (`TextResponse`):
```json
{ "text": "extracted job description text" }
```

**Errors**: `400` invalid URL · `502` fetch/scrape failure · `422` empty extracted text

---

### `POST /api/generate-resume`

Core endpoint. Takes the resume text, job description, and original file (base64). Runs the full LangGraph pipeline to tailor the resume, then produces a rewritten PDF.

**Request** (`GenerateRequest`):
```json
{
  "resume_text": "plain text of the resume",
  "jd_text": "job description text",
  "resume_file_b64": "base64-encoded original PDF or .tex",
  "resume_file_type": "pdf"
}
```

**Response** (`GenerateResponse`):
```json
{
  "resume": {
    "name": "John Doe",
    "email": "john@example.com",
    "summary": "Tailored professional summary…",
    "skills": ["Python", "React", "AWS"],
    "experience": [{ "job_title": "…", "company": "…", "bullets": ["…"] }],
    "education": [{ "degree": "…", "institution": "…" }],
    "certifications": [{ "name": "…", "issuer": "…" }],
    "replacements": [{ "old": "original text", "new": "rewritten text" }],
    "ats_score_before": 45,
    "ats_score": 88,
    "matched_keywords": ["Python", "React", "CI/CD"]
  },
  "rewritten_file_b64": "base64-encoded rewritten PDF"
}
```

**Errors**: `400` empty inputs · `502` AI generation failure · `500` PDF rewrite failure

---

### `POST /api/generate-resume-stream`

SSE streaming version of the full generate pipeline. Streams per-agent progress events as they complete.

**Request** (`StreamRequest`):
```json
{
  "resume_text": "…", "jd_text": "…",
  "resume_file_b64": "…", "resume_file_type": "pdf"
}
```

**SSE Events**:
- `started` — `{ "run_id": "..." }`
- `agent_complete` — `{ "agent": "extract_keywords", ... }` (per-agent with relevant scores/counts)
- `complete` — `{ "resume": ResumeData, "rewritten_file_b64": "..." }`
- `error` — `{ "detail": "error message" }`

---

### `POST /api/preview`

Run the pipeline WITHOUT PDF compilation — returns proposed replacements for user review.

**Request** (`PreviewRequest`):
```json
{ "resume_text": "…", "jd_text": "…" }
```

**Response** (`PreviewResponse`):
```json
{
  "replacements": [{ "old": "…", "new": "…" }],
  "ats_score_before": 45,
  "ats_score": 88,
  "matched_keywords": ["Python", "React"],
  "still_missing_keywords": ["Kubernetes"]
}
```

---

### `POST /api/preview-stream`

SSE streaming version of the preview pipeline. Same events as `/generate-resume-stream` but without PDF compilation; final `complete` event contains replacements and scores.

---

### `POST /api/confirm`

Apply user-approved replacements and compile the final PDF.

**Request** (`ConfirmRequest`):
```json
{
  "resume_text": "…",
  "replacements": [{ "old": "…", "new": "…" }],
  "resume_file_b64": "base64-encoded original file",
  "resume_file_type": "pdf"
}
```

**Response** (`ConfirmResponse`):
```json
{ "rewritten_file_b64": "base64-encoded compiled PDF" }
```

---

### `POST /api/generate-cover-letter`

Generate a cover letter, suggested job title, and LinkedIn outreach message.

**Request** (`CoverLetterRequest`):
```json
{
  "resume_text": "…", "jd_text": "…",
  "company_name": "Acme Corp"
}
```

**Response** (`CoverLetterResponse`):
```json
{
  "cover_letter": "Dear Hiring Manager,\n\n…",
  "suggested_job_title": "Senior Software Engineer",
  "linkedin_message": "Hi, I noticed your team is…"
}
```

---

### `GET /api/pipeline-runs`

List recent pipeline runs.

**Query params**: `limit` (default 20, max 100), `skip` (default 0)

**Response**: Array of run summaries with `_id`, `status`, `created_at`, `completed_at`, agent names/durations, and final result summary.

---

### `GET /api/pipeline-runs/{run_id}`

Full detail of a single pipeline run including all agent steps with timing, inputs, and outputs.

**Errors**: `404` run not found

---

### `GET /api/pipeline-runs/{run_id}/pdf`

Download the compiled PDF for a pipeline run as binary (`application/pdf`).

**Errors**: `404` no compiled PDF found

---

### `GET /api/pipeline-runs/status`

Check whether the pipeline run database is connected.

**Response**: `{ "db_connected": true }`

---

### `GET /api/health`

Health check. Returns `{ "status": "ok" }`.

## Data Flow

```
Upload PDF/TeX ─► parse_pdf() / parse_tex() ─► text + file_b64
                                    │
Enter JD ───────────────────────┤
                                    ▼
                            agents.generate_resume()
                            (LangGraph pipeline with parallel rewriters)
                                    │
                                    ▼
                            ResumeData + compiled PDF (base64)
```

## Services

### `parser.py`

Uses **PyMuPDF** (`fitz`) to extract plain text and styled HTML from each page. Text is normalised to fix common PDF extraction spacing artefacts (collapsed whitespace, camelCase word boundaries, missing post-punctuation spaces). HTML is generated from span dicts for reliable font/colour extraction. Returns base64-encoded original bytes for later rewriting. See [PARSER.md](services/PARSER.md).

### `latex_parser.py`

Strips LaTeX markup to produce plain text for the AI pipeline. Handles `\textbf`, `\href`, `\section`, `\item`, and other common commands. Preserves the raw `.tex` source as base64 for later rewriting by `latex_rewriter.py`.

### `scraper.py`

Uses **httpx** to fetch URLs and **BeautifulSoup** to extract text. Strips noise tags (`script`, `style`, `nav`, `header`, `footer`, `aside`, `iframe`). Prefers content from `<main>`, `<article>`, or `#job-details` elements.

### `rewriter.py`

Rewrites the original PDF in-place using AI-generated `{old, new}` replacement pairs. Uses `page.search_for()` to locate text, captures font/size/colour from the matched region, redacts the old text, and re-inserts new text via `TextWriter` with matched styling. Embedded fonts are extracted per-page and reused when available; falls back to Base-14 fonts. See [REWRITER.md](services/REWRITER.md).

### `latex_rewriter.py`

Applies AI replacements to `.tex` source code and compiles to PDF via xelatex (falls back to pdflatex/lualatex). Includes flexible pattern matching that tolerates `.tex` line-wrapping and LaTeX escape sequences (`\%`, `\&`). Auto-detects MiKTeX/TeX Live installations. Supports `LATEX_COMPILER_PATH` env var override and auto-installs missing LaTeX packages via `tlmgr`.

### `db.py`

MongoDB persistence for pipeline run tracking using **pymongo** (synchronous). All operations are **best-effort** — failures are logged but never break the pipeline. Stores run metadata, per-agent timing/IO, final results, and compiled PDF bytes (as BSON Binary). See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline tracking design.

### `agents/`

LangGraph multi-agent pipeline with 3 parallel section rewriters and a conditional refinement loop. Uses **litellm** for multi-provider LLM access (Groq, Gemini, OpenAI, Anthropic, Ollama, DeepSeek, OpenRouter). See [agents/AGENTS.md](services/agents/AGENTS.md).

## Running

```bash
# from project root
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
uvicorn backend.main:app --reload --port 8000
```

## Dependencies

See [requirements.txt](requirements.txt). Key packages:

- **FastAPI** + **uvicorn** — web framework + ASGI server
- **PyMuPDF** (`fitz`) — PDF parsing and in-place text rewriting
- **LangGraph** — multi-agent pipeline graph orchestration
- **litellm** — unified LLM interface (Groq, Gemini, OpenAI, Anthropic, Ollama, DeepSeek, OpenRouter)
- **rapidfuzz** — algorithmic keyword matching (fuzzy, synonym, exact)
- **httpx** + **beautifulsoup4** — HTTP client + HTML scraping
- **pymongo** — MongoDB driver for pipeline run tracking
- **pydantic** — data validation
- **python-json-logger** — structured JSON logging
