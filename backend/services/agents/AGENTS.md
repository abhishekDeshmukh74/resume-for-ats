# LangGraph Multi-Agent Pipeline

## Overview

The AI logic uses **LangGraph** to orchestrate a **multi-agent pipeline with parallel section rewriters and a conditional refinement loop**. Each agent is a standalone node that reads from and writes to a shared `AgentState` TypedDict.

All agents use a multi-provider LLM via **litellm** (7 providers, configured via `LLM_PROVIDER` env var) with temperature 0.2 via `llm.py`.

## Pipeline Flow

```
extract_keywords → analyse_resume → score_before
  → ┌ rewrite_skills  ┐
    │ rewrite_summary  │ (parallel fan-out / fan-in)
    └ rewrite_experience┘
  → qa_deduplicate → score_extract
  → ┌─ [score ≥ 90 OR pass > 0] → compile_pdf → END
    └─ [score < 90 AND pass == 0] → refine_rewrite → refine_qa → compile_pdf → END
```

```mermaid
graph TD
    START((Start)) --> extract_keywords

    extract_keywords["🔑 Extract Keywords\n(keyword_extractor.py)\n→ jd_keywords, required/preferred"]
    analyse_resume["📋 Analyse Resume\n(resume_analyser.py)\n→ sections, gap_analysis"]
    score_before["📊 Score Before\n(scorer.py)\n→ ats_score_before, missing_keywords"]
    rewrite_skills["✏️ Rewrite Skills\n(skills_rewriter.py)\n→ raw_replacements"]
    rewrite_summary["✏️ Rewrite Summary\n(summary_rewriter.py)\n→ raw_replacements"]
    rewrite_experience["✏️ Rewrite Experience\n(experience_rewriter.py)\n→ raw_replacements"]
    qa_deduplicate["✅ QA & Deduplicate\n(qa_agent.py)\n→ replacements\n+ AI phrase cleanup"]
    score_extract["📈 Score & Extract\n(scorer.py)\n→ ats_score, still_missing_keywords\nalgorithmic (rapidfuzz)"]
    refine_rewrite["🔄 Refine Rewrite\n(refinement_agent.py)\n→ raw_replacements (appended)"]
    refine_qa["✅ Refine QA\n(qa_agent.py)\n→ replacements (appended)"]
    compile_pdf["📄 Compile PDF\n(pdf_compiler.py)\n→ compiled_pdf_b64"]
    END_NODE((End))

    extract_keywords --> analyse_resume
    analyse_resume --> score_before
    score_before --> rewrite_skills
    score_before --> rewrite_summary
    score_before --> rewrite_experience
    rewrite_skills --> qa_deduplicate
    rewrite_summary --> qa_deduplicate
    rewrite_experience --> qa_deduplicate
    qa_deduplicate --> score_extract

    score_extract -->|"score ≥ 90\nOR pass > 0"| compile_pdf
    score_extract -->|"score < 90\nAND pass == 0\nAND missing keywords"| refine_rewrite

    refine_rewrite --> refine_qa
    refine_qa --> compile_pdf

    compile_pdf --> END_NODE

    style extract_keywords fill:#e8f5e9,stroke:#2e7d32
    style analyse_resume fill:#e3f2fd,stroke:#1565c0
    style score_before fill:#fff3e0,stroke:#e65100
    style rewrite_skills fill:#fce4ec,stroke:#c62828
    style rewrite_summary fill:#fce4ec,stroke:#c62828
    style rewrite_experience fill:#fce4ec,stroke:#c62828
    style qa_deduplicate fill:#f3e5f5,stroke:#6a1b9a
    style score_extract fill:#fff3e0,stroke:#e65100
    style refine_rewrite fill:#ffebee,stroke:#b71c1c
    style refine_qa fill:#f3e5f5,stroke:#6a1b9a
    style compile_pdf fill:#e0f2f1,stroke:#00695c
```

## Key Improvements

1. **Enhanced Algorithmic Scoring** — Uses `rapidfuzz` for multi-signal keyword matching: exact word-boundary, synonym/abbreviation expansion (50+ tech aliases), and fuzzy matching for typos/variations. LLM is used only for structured data extraction, not scoring.
2. **Section-Aware Scoring** — Keywords found in high-value sections (title 1.4×, summary 1.3×, skills 1.2×) receive placement bonuses, reflecting real ATS behaviour.
3. **Keyword Stuffing Detection** — Penalises resumes that repeat keywords excessively (>4 occurrences), deducting up to 15 points.
4. **Priority Keywords** — Extracts `required_skills` vs `preferred_skills` from JD so the rewriter prioritises must-have keywords.
5. **Missing Keyword Focus** — The rewriter receives explicit lists of MISSING keywords (not all keywords) with priority levels (REQUIRED / PREFERRED / OTHER).
6. **Conditional Refinement Loop** — If the first-pass score is below 90, a focused refinement agent injects still-missing keywords into the already-rewritten resume.
7. **AI Phrase Cleanup** — Replaces AI-sounding buzzwords ("spearheaded", "leveraged", "synergized") with simpler human-sounding alternatives.

## Agent Details

### Agent 1 — Keyword Extractor (`keyword_extractor.py`)

**Input**: `jd_text`
**Output**: `jd_keywords`, `keyword_categories`, `required_keywords`, `preferred_keywords`

Extracts 30–60 unique keywords from the job description and categorises them into:
- `technical_skills`, `soft_skills`, `tools_platforms`
- `domain_knowledge`, `certifications`, `action_verbs`
- `required_skills` — must-have keywords from core requirements
- `preferred_skills` — nice-to-have keywords from secondary mentions

### Agent 2 — Resume Analyser (`resume_analyser.py`)

**Input**: `resume_text`, `jd_keywords`
**Output**: `resume_sections`, `gap_analysis`

- Identifies resume sections (summary, skills, each experience entry, education)
- Maps which keywords are already present vs missing
- Produces a gap analysis with specific placement recommendations

### Agent 3 — Pre-Rewrite Scorer (`scorer.py`)

**Input**: `resume_text`, `jd_keywords`, `keyword_categories`, `resume_sections`
**Output**: `ats_score_before`, `missing_keywords`

Scores the **original** resume using **multi-signal algorithmic matching** (via `rapidfuzz`):
- **Exact**: word-boundary regex matching
- **Synonym**: 50+ tech abbreviation expansions (e.g. `k8s` → `kubernetes`, `JS` → `javascript`)
- **Fuzzy**: rapidfuzz token matching (≥80% similarity threshold) for typos/variations
- **Category weights**: tech skills 1.5×, certifications 1.25×
- **Section placement**: keywords in title/summary/skills get bonus weight
- **Stuffing penalty**: -3 pts per keyword appearing >4 times (max -15)
- LLM provides a secondary score; final = conservative min(LLM, algorithmic)
- Also identifies which keywords are missing from the original resume

### Agent 4 — Section Rewriters (parallel)

Three specialised rewriters run **in parallel** via LangGraph fan-out/fan-in. Each targets a specific resume section with section-appropriate rewriting strategies. All three receive the same inputs and write to the shared `raw_replacements` list (merge reducer).

#### Agent 4a — Skills Rewriter (`skills_rewriter.py`)

**Input**: `resume_text`, `keyword_categories`, `gap_analysis`, `missing_keywords`, `required_keywords`, `preferred_keywords`, `resume_sections`
**Output**: `raw_replacements` (list of `{old, new}` dicts, merged)

Rewrites the **skills** section:
- Targets comma-separated skill lines
- Appends missing keywords naturally within existing skill groupings
- Preserves original formatting (commas, pipes, bullets)
- Prioritises REQUIRED → PREFERRED → OTHER keywords
- Each keyword appears at most 2 times across all replacements

#### Agent 4b — Summary Rewriter (`summary_rewriter.py`)

**Input**: `resume_text`, `keyword_categories`, `gap_analysis`, `missing_keywords`, `required_keywords`, `preferred_keywords`, `resume_sections`
**Output**: `raw_replacements` (list of `{old, new}` dicts, merged)

Rewrites the **summary/objective** section:
- Integrates broad JD keywords into natural prose
- Maintains professional tone and readability
- Keeps replacement length within ±20% of original
- Uses synonyms and variations to avoid keyword stuffing

#### Agent 4c — Experience Rewriter (`experience_rewriter.py`)

**Input**: `resume_text`, `keyword_categories`, `gap_analysis`, `missing_keywords`, `required_keywords`, `preferred_keywords`, `resume_sections`
**Output**: `raw_replacements` (list of `{old, new}` dicts, merged)

Rewrites **experience** bullet points:
- Preserves quantified metrics and achievements
- Weaves specific technical keywords into action-oriented bullets
- Keeps replacement length within ±20% of original
- `old` must be verbatim from the original resume

### Agent 5 — QA Agent (`qa_agent.py`)

**Input**: `resume_text`, `jd_keywords`, `raw_replacements`
**Output**: `replacements` (list of `TextReplacement` Pydantic models)

Validates and fixes:
1. Checks each `old` text exists in the original resume
2. Counts keyword frequency across all `new` texts, flags overuse (>2)
3. Instructs LLM to fix duplicates with synonyms
4. Programmatic dedup safety net (removes duplicate `old` entries)
5. Enforces skills section format consistency
6. **AI phrase cleanup**: replaces 30+ AI-sounding phrases with simpler alternatives (e.g. "spearheaded" → "led", "leveraged" → "used")

### Agent 6 — Final Scorer (`scorer.py`)

**Input**: `resume_text`, `jd_keywords`, `jd_text`, `replacements`, `keyword_categories`, `resume_sections`
**Output**: `ats_score`, `matched_keywords`, `algorithmic_score`, `still_missing_keywords`, `name`, `email`, `skills`, `experience`, etc.

- Applies replacements to produce the "final resume" text
- Runs **multi-signal algorithmic scoring** via `rapidfuzz`:
  - Exact word-boundary + synonym expansion + fuzzy matching
  - Category weights (tech 1.5×) + section placement bonuses
  - Keyword stuffing penalty (up to -15 pts)
- LLM is used **only for structured data extraction** (name, skills, experience, etc.) — not for scoring
- Reports `still_missing_keywords` used by the refinement loop
- Match details logged: each keyword shows match type (exact/synonym/fuzzy) and confidence

### Conditional Refinement (if score < 90)

### Agent 6b — Refinement Writer (`refinement_agent.py`)

**Input**: `resume_text`, `replacements`, `still_missing_keywords`, `required_keywords`
**Output**: `raw_replacements` (appended), `rewrite_pass` = 1

Targeted second-pass keyword injection:
- Applies existing replacements to get current resume text
- Focuses specifically on `still_missing_keywords`, prioritising required ones
- Targets skills section and summary as highest-impact areas
- Uses exact JD phrasing for ATS compatibility
- Only runs once (pass 0 → pass 1, no further loops)

### Agent 6c — Refinement QA (`qa_agent.py`)

Same QA agent as Agent 5, validates the refinement replacements.

### Agent 7 — PDF Compiler (`pdf_compiler.py`)

**Input**: `replacements`, `resume_file_b64`, `resume_file_type`
**Output**: `compiled_pdf_b64`

Applies the validated replacements to the original file and produces the final PDF:
- **PDF uploads** → `rewriter.py` (PyMuPDF in-place text replacement)
- **LaTeX uploads** → `latex_rewriter.py` (source patching + xelatex/pdflatex compilation)

If no original file is available, returns an empty string (the pipeline can still return structured `ResumeData` without a compiled file).

## Pipeline Run Tracking

Every pipeline execution is tracked in MongoDB via `db.py` (best-effort):

1. A run is created at the start of `generate_resume()` with status `"running"`
2. Each agent is wrapped by `_tracked()` in `graph.py`, which records:
   - Agent name and execution duration (ms)
   - Input summary (relevant state keys only)
   - Output data (serialised, truncated for storage)
3. On success: final result saved with ATS scores, replacement count, and name
4. On failure: error message saved

The run ID is stored in a `contextvars.ContextVar` for thread-safe tracking.

## Shared State (`state.py`)

`AgentState` is a `TypedDict` with annotated reducers:
- **List fields** use `_merge_lists` (extend, not replace)
- **Scalar/dict fields** use `_overwrite` (last-write-wins)

Key state fields:

| Category | Fields |
|----------|--------|
| Inputs | `resume_text`, `jd_text`, `resume_file_b64`, `resume_file_type` |
| Agent 1 | `jd_keywords` (list, merge), `keyword_categories` (dict, overwrite), `required_keywords` (list, overwrite), `preferred_keywords` (list, overwrite) |
| Agent 2 | `resume_sections` (dict, overwrite), `gap_analysis` (str, overwrite) |
| Agent 3 | `ats_score_before` (int, overwrite), `missing_keywords` (list, overwrite) |
| Agents 4a/4b/4c | `raw_replacements` (list, merge) — all three parallel rewriters append to this |
| Agent 5 | `replacements` (list[TextReplacement], merge) |
| Agent 6 | `ats_score` (int), `matched_keywords` (list, overwrite), `algorithmic_score` (float, overwrite), `still_missing_keywords` (list, overwrite), structured fields |
| Refinement | `rewrite_pass` (int, overwrite) — tracks which pass we're on (0 = initial, 1 = refinement) |
| Agent 7 | `compiled_pdf_b64` (str, overwrite) |

## LLM Configuration (`llm.py`)

The LLM provider is selected by the `LLM_PROVIDER` env var. All providers are accessed through **litellm**, a unified interface that normalises the API across providers.

| Provider | Model | Env Var | litellm Prefix |
|----------|-------|---------|----------------|
| Groq (default) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `groq/` |
| Google Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` | `gemini/` |
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` | (none) |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | `anthropic/` |
| DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` | `deepseek/` |
| OpenRouter | `meta-llama/llama-3-70b` | `OPENROUTER_API_KEY` | `openrouter/` |
| Ollama (local) | `llama3.1` | `OLLAMA_BASE_URL` | `ollama/` |

**Common settings** (all providers): temperature `0.2`, max_tokens `8192`.

- `_LiteLLMChat` is a custom wrapper that conforms to LangChain's `BaseChatModel` interface so it can be used as a drop-in within LangGraph chains.
- Groq supports `GROQ_API_KEYS` (comma-separated) for round-robin API key rotation across calls.
- `GROQ_MODEL` and `GEMINI_MODEL` env vars override the default model for their respective providers.

`parse_llm_json()` safely extracts JSON from LLM responses, handling markdown code fences. Includes a multi-pass JSON repair pipeline for truncated or malformed output (trailing comma removal → bracket closure → regex object extraction).

## Public API (`graph.py`)

### Full Pipeline

```python
from backend.services.agents import generate_resume

resume_data, compiled_pdf_b64 = generate_resume(
    resume_text, jd_text,
    resume_file_b64="<base64-encoded original file>",
    resume_file_type="pdf",  # or "tex"
)
```

Returns `(ResumeData, compiled_pdf_b64)`.

### Preview Pipeline (no PDF compilation)

```python
from backend.services.agents.graph import preview_resume

resume_data = preview_resume(resume_text, jd_text)
```

Returns `ResumeData` with proposed replacements and scores but no compiled PDF. Used by the `POST /api/preview` endpoint.

### Streaming Pipeline

```python
from backend.services.agents.graph import generate_resume_stream

async for event in generate_resume_stream(resume_text, jd_text, ...):
    # event is an SSE dict: {"agent": "...", "status": "..."} or final result
    ...
```

Yields SSE events as each agent completes. Used by `POST /api/generate-resume-stream` and `POST /api/preview-stream`.

### Cover Letter (standalone)

```python
from backend.services.agents.cover_letter import generate_cover_letter

result = generate_cover_letter(resume_text, jd_text)
# result has: cover_letter, suggested_title, linkedin_message
```
