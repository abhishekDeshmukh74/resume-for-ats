# Backend — pass-ats API

FastAPI backend that parses PDF resumes, scrapes job descriptions, runs a **LangGraph multi-agent pipeline** (6 sequential AI agents) to tailor resume content for ATS, and rewrites the original PDF in-place with updated text.

## Architecture

```
backend/
├── main.py              # FastAPI app, CORS, router registration, logging
├── models.py            # Pydantic request/response schemas
├── routers/
│   ├── resume.py        # POST /api/parse-resume
│   ├── jd.py            # POST /api/scrape-jd
│   ├── generate.py      # POST /api/generate-resume
│   └── pipeline.py      # GET  /api/pipeline-runs (list + detail)
└── services/
    ├── parser.py         # PDF text + HTML extraction (PyMuPDF)
    ├── scraper.py        # URL → plain text (httpx + BeautifulSoup)
    ├── rewriter.py       # In-place PDF text replacement (PyMuPDF)
    ├── db.py             # MongoDB persistence for pipeline run tracking
    └── agents/           # LangGraph multi-agent pipeline
        ├── graph.py              # StateGraph wiring + pipeline tracking
        ├── state.py              # Shared AgentState TypedDict
        ├── llm.py                # ChatGroq instance + JSON parser
        ├── keyword_extractor.py  # Agent 1: JD keyword extraction
        ├── resume_analyser.py    # Agent 2: Section analysis + gap identification
        ├── scorer.py             # Agents 3 & 6: ATS scoring + structured extraction
        ├── rewriter_agent.py     # Agent 4: old→new replacement generation
        └── qa_agent.py           # Agent 5: Validation + deduplication
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key from [console.groq.com](https://console.groq.com/) |
| `ALLOWED_ORIGINS` | Yes | Comma-separated CORS origins (e.g. `http://localhost:5173`) |
| `MONGODB_URL` | No | MongoDB connection string for pipeline run tracking |

## API Endpoints

### `POST /api/parse-resume`

Upload a PDF file. Returns extracted text, styled HTML preview, and the original file as base64.

**Request**: `multipart/form-data` with field `file` (PDF, max 10 MB)

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

Core endpoint. Takes the resume text, job description, and original PDF (base64). Runs the LangGraph multi-agent pipeline to tailor the resume, then rewrites the original PDF in-place.

**Request** (`GenerateRequest`):
```json
{
  "resume_text": "plain text of the resume",
  "jd_text": "job description text",
  "resume_file_b64": "base64-encoded original PDF",
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

### `GET /api/pipeline-runs`

List recent pipeline runs.

**Query params**: `limit` (default 20, max 100), `skip` (default 0)

**Response**: Array of run summaries with `id`, `status`, `created_at`, `completed_at`, agent names, and final result summary.

---

### `GET /api/pipeline-runs/{run_id}`

Full detail of a single pipeline run including all agent steps with timing, inputs, and outputs.

**Errors**: `404` run not found

---

### `GET /api/health`

Health check. Returns `{ "status": "ok" }`.

## Data Flow

```
Upload PDF ──► parse_pdf() ──► text + file_b64
                                    │
Enter JD ───────────────────────────┤
                                    ▼
                            agents.generate_resume()
                            (LangGraph 6-agent pipeline)
                                    │
                                    ▼
                            ResumeData (tailored JSON + replacements)
                                    │
                                    ▼
                            rewriter.rewrite_pdf()
                                    │
                                    ▼
                            rewritten PDF (base64) + ResumeData
```

## Services

### `parser.py`

Uses **PyMuPDF** (`fitz`) to extract plain text (`page.get_text("text")`) and styled HTML (`page.get_text("html")`) from each page. Returns base64-encoded original bytes for later rewriting. See [PARSER.md](services/PARSER.md).

### `scraper.py`

Uses **httpx** to fetch URLs and **BeautifulSoup** to extract text. Strips noise tags (`script`, `style`, `nav`, `header`, `footer`, `aside`, `iframe`). Prefers content from `<main>`, `<article>`, or `#job-details` elements.

### `rewriter.py`

Rewrites the original PDF in-place using AI-generated `{old, new}` replacement pairs. Extracts embedded fonts from the PDF and reuses them for replacement text. Uses block-scoped, line-aware matching with Unicode normalisation, and inserts text at original baselines via `TextWriter`. See [REWRITER.md](services/REWRITER.md).

### `db.py`

MongoDB persistence for pipeline run tracking. All operations are **best-effort** — failures are logged but never break the pipeline. Stores run metadata, per-agent timing/IO, and final results. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline tracking design.

### `agents/`

LangGraph multi-agent pipeline with 6 sequential agents: keyword extraction → resume analysis → pre-rewrite scoring → rewriting → QA/dedup → final scoring. See [agents/AGENTS.md](services/agents/AGENTS.md).

### `groq_service.py` (superseded)

Original monolithic AI service, now replaced by the LangGraph agents pipeline. Kept for reference. See [GROQ_SERVICE.md](services/GROQ_SERVICE.md).

## Running

```bash
# from project root
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
uvicorn backend.main:app --reload --port 8000
```

## Dependencies

See [requirements.txt](requirements.txt). Key packages:

- **FastAPI** + **uvicorn** — web framework + ASGI server
- **PyMuPDF** (`fitz`) — PDF parsing and in-place text rewriting
- **LangGraph** + **langchain-groq** — multi-agent AI pipeline
- **httpx** + **beautifulsoup4** — HTTP client + HTML scraping
- **pymongo** — MongoDB driver for pipeline run tracking
- **pydantic** — data validation
