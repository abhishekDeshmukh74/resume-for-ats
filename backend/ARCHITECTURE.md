# Backend Architecture

## Overview

FastAPI backend for the pass-ats resume tailor. Receives a PDF or LaTeX resume and job description, runs a **LangGraph 13-node multi-agent pipeline** (10 AI agents + 3 internal nodes) to rewrite resume content for ATS optimisation, and returns a modified PDF preserving the original layout.

## Directory Structure

```
backend/
├── main.py              # FastAPI app, CORS, router registration, logging setup
├── models.py            # Pydantic models (request/response schemas)
├── routers/
│   ├── resume.py        # POST /api/parse-resume — PDF + LaTeX upload + text extraction
│   ├── jd.py            # POST /api/scrape-jd — URL → JD text scraper
│   ├── generate.py      # POST /api/generate-resume — AI rewrite + PDF output
│   └── pipeline.py      # GET  /api/pipeline-runs — run list + detail inspection
└── services/
    ├── parser.py         # PDF text extraction (PyMuPDF, span-dict HTML)
    ├── latex_parser.py   # LaTeX (.tex) text extraction
    ├── scraper.py        # URL scraping for job descriptions
    ├── rewriter.py       # In-place PDF text replacement (search_for + redact)
    ├── latex_rewriter.py # LaTeX source patching + compilation (xelatex)
    ├── db.py             # MongoDB persistence for pipeline run tracking
    └── agents/           # LangGraph multi-agent pipeline
        ├── __init__.py           # Re-exports generate_resume()
        ├── state.py              # ResumeGraphState TypedDict (shared pipeline state)
        ├── llm.py                # Multi-provider LLM (Groq/Gemini) + JSON parsing + sanitisation
        ├── tools.py              # Deterministic scoring utilities (no LLM)
        ├── intake_parser.py      # Agent 1: Parse resume into structured JSON
        ├── jd_analyzer.py        # Agent 2: Extract JD signals and ATS keywords
        ├── gap_analyzer.py       # Agent 3: Resume vs JD gap analysis
        ├── bullet_rewriter.py    # Agent 5: Rewrite experience bullets (Action+Tech+Scope+Result)
        ├── summary_optimizer.py  # Agent 6: Generate ATS-optimized summary
        ├── skills_optimizer.py   # Agent 7: Normalise and align skills
        ├── truth_guard.py        # Agent 8: Truthfulness verification (safety)
        ├── scorer.py             # Agents 4 & 11: Hybrid ATS scoring (baseline + final)
        ├── critic.py             # Agent 9: Quality gate with revision routing
        ├── formatter.py          # Agent 10: Multi-format export + replacement generation
        ├── pdf_compiler.py       # Agent 11: Apply replacements + compile PDF
        ├── graph.py              # 13-node StateGraph wiring + public generate_resume()
        └── AGENTS.md             # Detailed pipeline documentation
```

## Request Flow (Generate)

```
Frontend                    Backend
   │                          │
   ├─ POST /api/generate ────►│
   │  { resume_text,          │
   │    jd_text,              │
   │    resume_file_b64 }     │
   │                          ├─ agents.generate_resume()
   │                          │   → LangGraph pipeline (13 nodes):
   │                          │     1.  parse_resume      → structured resume JSON
   │                          │     2.  analyze_jd        → JD signals & keywords
   │                          │     3.  compute_gap       → gap report
   │                          │     4.  baseline_score    → pre-optimisation ATS score
   │                          │     5.  optimize_summary  → ATS summary
   │                          │     6.  optimize_skills   → normalised skills
   │                          │     7.  optimize_experience → rewritten bullets
   │                          │     8.  merge_resume      → combined draft
   │                          │     9.  truth_guard       → truthfulness check
   │                          │    10.  critic            → quality gate
   │                          │    11.  final_score       → post-optimisation ATS score
   │                          │    12.  export            → text + markdown + replacements
   │                          │    13.  compile_pdf       → rewritten PDF
   │                          │   → (ResumeData, compiled_pdf_b64)
   │                          │
   │◄─ { resume, b64_pdf } ──┤
```

## LangGraph Pipeline

The AI logic is split into 13 nodes using LangGraph's `StateGraph`, with conditional revision loops (max 2 revisions):

| # | Agent | Node Name | File | Purpose |
|---|-------|-----------|------|---------|
| 1 | Intake Parser | `parse_resume` | `intake_parser.py` | Extract structured JSON from raw resume |
| 2 | JD Analyzer | `analyze_jd` | `jd_analyzer.py` | Extract JD signals, keywords, requirements |
| 3 | Gap Analyzer | `compute_gap` | `gap_analyzer.py` | Resume vs JD gap analysis |
| 4 | Baseline Scorer | `baseline_score` | `scorer.py` | Hybrid ATS score before optimisation |
| 5 | Summary Optimizer | `optimize_summary` | `summary_optimizer.py` | Generate 3–4 sentence ATS summary |
| 6 | Skills Optimizer | `optimize_skills` | `skills_optimizer.py` | Normalise and align skills list |
| 7 | Bullet Rewriter | `optimize_experience` | `bullet_rewriter.py` | Rewrite bullets (Action+Tech+Scope+Result) |
| 8 | — | `merge_resume` | `graph.py` | Combine optimized sections into draft |
| 9 | Truth Guard | `truth_guard` | `truth_guard.py` | Verify claims against original resume |
| 10 | Critic | `critic` | `critic.py` | Quality gate — route to revision or proceed |
| — | — | `rewrite_router` | `graph.py` | Route revision to specific optimizer |
| 11 | Final Scorer | `final_score` | `scorer.py` | Hybrid ATS score after optimisation |
| 12 | Formatter | `export` | `formatter.py` | Text + markdown + old→new replacements |
| 13 | PDF Compiler | `compile_pdf` | `pdf_compiler.py` | Apply replacements to original file |

All agents share a `ResumeGraphState` TypedDict. See `backend/services/agents/AGENTS.md` for detailed agent documentation.

## Scoring Formula

ATS score = weighted composite of deterministic + LLM sub-scores:

```
0.30 × keyword_coverage  +  0.25 × semantic_score  +  0.20 × section_quality
+  0.15 × ats_format  +  0.10 × truthfulness  −  (5 × stuffed_keyword_count)
```

## Pipeline Run Tracking

Every pipeline execution is tracked in MongoDB (best-effort — failures never break the pipeline):

1. **`graph.py`** creates a pipeline run in MongoDB at the start via `db.create_pipeline_run()`
2. Each agent is wrapped by `_tracked()`, which records:
   - Agent name, execution duration (ms)
   - Input state summary (relevant keys only, via `_AGENT_INPUT_KEYS` mapping)
   - Output data (serialised, truncated for storage)
3. On completion: `db.complete_pipeline_run()` saves ATS scores, replacement count, and name
4. On failure: `db.fail_pipeline_run()` saves the error message
5. Run ID is stored in a `contextvars.ContextVar` for thread-safe tracking

**Inspection endpoints**:
- `GET /api/pipeline-runs` — list runs with summary (status, timestamps, agent names)
- `GET /api/pipeline-runs/{id}` — full detail with per-agent timing, I/O data

## Key Data Models (models.py)

- **`TextReplacement`**: `{old: str, new: str}` — a single find-and-replace pair
- **`ResumeData`**: Full structured resume + `replacements: list[TextReplacement]` + `ats_score` + `ats_score_before`
- **`GenerateRequest`**: `{resume_text, jd_text, resume_file_b64, resume_file_type}`
- **`GenerateResponse`**: `{resume: ResumeData, rewritten_file_b64: str}`

## Critical Path: Text Matching

The most fragile part of the pipeline is matching AI-generated `old` strings against PDF/LaTeX text:

### PDF Flow
1. **`parser.py`** extracts text using `page.get_text("text")` with normalisation.
2. **Formatter** (`formatter.py`) generates `old` strings that should be verbatim substrings.
3. **`rewriter.py`** uses `page.search_for()` to locate `old` text in the PDF, then redacts and re-inserts.

### LaTeX Flow
1. **`latex_parser.py`** strips LaTeX commands to produce plain text.
2. Agents work on the plain text.
3. **`latex_rewriter.py`** applies replacements to the `.tex` source (with flexible pattern matching for line-wrapping and LaTeX escapes), then compiles to PDF via xelatex.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes* | Groq API key for AI calls |
| `GROQ_API_KEYS` | No | Comma-separated Groq keys for failover |
| `GEMINI_API_KEY` | Yes* | Google Gemini API key for AI calls |
| `LLM_PROVIDER` | No | `"groq"` (default) or `"gemini"` |
| `GROQ_MODEL` | No | Groq model name (default: `llama-3.3-70b-versatile`) |
| `GEMINI_MODEL` | No | Gemini model name (default: `gemini-2.0-flash`) |
| `ALLOWED_ORIGINS` | Yes | Comma-separated CORS origins |
| `MONGODB_URL` | No | MongoDB connection string for pipeline run tracking |

\* At least one of `GROQ_API_KEY` or `GEMINI_API_KEY` is required, depending on `LLM_PROVIDER`.

## Running

```bash
uvicorn backend.main:app --reload --port 8000
```
