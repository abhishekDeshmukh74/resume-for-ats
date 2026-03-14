# Groq AI Service — `groq_service.py` (Superseded)

> **This module has been replaced by the LangGraph multi-agent pipeline in
> `backend/services/agents/`.** The file still exists for reference but is no
> longer imported by the generate router. See
> [agents/AGENTS.md](agents/AGENTS.md) for the current architecture.

## Purpose (Historical)

Called the Groq API (LLaMA 3.3 70B) in a single monolithic prompt to generate an ATS-optimised resume given the original resume text and a target job description.

## Why It Was Replaced

The single-prompt approach had several limitations:

1. **Unreliable `old` text** — the LLM frequently paraphrased instead of copy-pasting verbatim substrings, causing PDF replacement failures.
2. **No validation step** — keyword overuse and duplicate replacements went unchecked.
3. **No baseline scoring** — no way to show a before/after ATS score comparison.
4. **Monolithic prompt** — a single large prompt made it hard to debug which part of the pipeline failed.

The multi-agent pipeline addresses all of these by splitting the work into 7 focused agents with dedicated QA validation, dual scoring, and a PDF compilation step.

## How It Worked

1. Receives `resume_text` (plain text from PDF extraction) and `jd_text` (job description).
2. Sends a structured prompt to Groq with a system message and user message.
3. Parses the JSON response into a `ResumeData` Pydantic model.

## Prompt Design

The prompt has two critical goals:

### Goal 1: High-Quality Replacements (Most Important)

The AI must return a `replacements` array of `{old, new}` pairs where:

- **`old`** is a **verbatim, character-for-character** substring from the original resume text.
  - This is critical because the rewriter (`rewriter.py`) uses these strings to locate text in the PDF.
  - If the `old` value doesn't match the PDF text, the replacement silently fails.
- **`new`** is the ATS-optimised rewrite incorporating JD keywords.

### Goal 2: ATS Score > 90%

The AI must aggressively rewrite:
- **Every** experience bullet — weave in JD keywords, action verbs, and technologies.
- The **summary** — tailor to the target role.
- **Skills** — reorder to prioritise JD-relevant skills, add standard aliases.

### What Must NOT Change

- Job titles, company names, employment dates
- Degree names, institution names, graduation dates
- Certifications
- Contact information
- Number of entries (no adding/removing/merging)

## Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `llama-3.3-70b-versatile` | Best balance of speed and quality on Groq |
| Temperature | `0.2` | Low for deterministic, faithful output (especially verbatim `old` copying) |
| Max tokens | `8192` | Enough for full resume + all replacements |
| Response format | `json_object` | Forces valid JSON output |

## Response Schema

```json
{
  "replacements": [{"old": "...", "new": "..."}],
  "name": "string",
  "email": "string | null",
  "phone": "string | null",
  "linkedin": "string | null",
  "github": "string | null",
  "location": "string | null",
  "summary": "rewritten summary",
  "skills": ["skill1", "skill2"],
  "experience": [{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": []}],
  "education": [{"degree": "", "institution": "", "location": "", "graduation_date": "", "details": []}],
  "certifications": [{"name": "", "issuer": "", "date": ""}],
  "ats_score": 90,
  "matched_keywords": ["keyword1", "keyword2"]
}
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `old` text doesn't match PDF | LLM paraphrased instead of copy-pasting | Strengthen prompt; lower temperature; add explicit examples |
| Empty replacements array | LLM didn't understand the instruction | Check prompt; ensure resume_text is not empty |
| ats_score always low | LLM not rewriting aggressively enough | Prompt must explicitly say "above 90%" |
| JSON parse error | LLM returned markdown fences | `response_format=json_object` prevents this; if it still happens, strip fences |

## Dependencies

- **`groq`** Python SDK — API client
- **`GROQ_API_KEY`** environment variable — must be set
- **`backend.models.ResumeData`** — Pydantic model for the response
