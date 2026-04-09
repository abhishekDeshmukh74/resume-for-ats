# Resume for ATS Frontend

React + TypeScript SPA that drives the 5-step resume tailoring workflow with SSE streaming, a diff review phase, and a dedicated pipeline runs inspector page.

## Tech Stack

- **React 19** with TypeScript
- **Vite 8** — dev server with proxy (`/api → http://localhost:8000`)
- **Tailwind CSS 4** — utility-first styling
- **Zustand 5** — state management with `persist` middleware (localStorage)
- **react-router-dom 7** — client-side routing (`/` home, `/info` pipeline runs)

## Architecture

```
src/
├── App.tsx                  # Root — routing (/ and /info) + 5-step wizard + SSE streaming
├── main.tsx                 # React DOM entry
├── index.css                # Tailwind base styles
├── api/
│   └── client.ts            # Typed fetch wrappers + SSE stream helpers
├── components/
│   ├── StepIndicator.tsx    # Progress bar (steps 1–5)
│   ├── ResumeUpload.tsx     # Step 1 — PDF / LaTeX (.tex) upload + parse
│   ├── JDInput.tsx          # Step 2 — Job description input / scrape
│   ├── DiffPreview.tsx      # Step 4 — Side-by-side old→new replacement diffs
│   ├── ResumePreview.tsx    # Step 5 — PDF preview (iframe) + download + ATS scores
│   └── CoverLetterPanel.tsx # Cover letter + LinkedIn outreach generator
├── store/
│   └── appStore.ts          # Zustand store (persist middleware, excludes large blobs)
├── pages/
│   └── InfoPage.tsx         # Pipeline runs inspector (list + detail view)
└── types/
    └── resume.ts            # Shared TypeScript interfaces
```

## Pages

### Home (`/`)

The 5-step resume tailoring wizard:

| Step | Component | Backend Call | Description |
|------|-----------|-------------|-------------|
| 1 | `ResumeUpload` | `POST /api/parse-resume` | Upload PDF or LaTeX (.tex), receive parsed text + base64 |
| 2 | `JDInput` | `POST /api/scrape-jd` | Paste or scrape a job description |
| 3 | (auto / loading) | `POST /api/preview-stream` (SSE) | Stream per-agent progress, get proposed replacements |
| 4 | `DiffPreview` | — | Review old→new text diffs before confirming |
| 5 | `ResumePreview` | `POST /api/confirm` | Compile final PDF, preview, ATS scores, download, cover letter |

### Pipeline Runs (`/info`)

Full pipeline inspection page (`InfoPage.tsx`):
- Lists recent pipeline runs with status badges (running / completed / failed)
- Run detail view with:
  - ATS score panel (before → after with delta badge)
  - Per-agent expandable cards with duration bars
  - Input/output toggle tabs with recursive JSON tree viewer
  - Matched keywords and replacement count
  - Error display for failed runs

## State Management (`src/store/appStore.ts`)

Zustand store with `persist` middleware (localStorage):

**State fields**: `step`, `resumeText`, `jdText`, `fileB64`, `fileType`, `fileName`, `generatedResume`, `rewrittenFileB64`, `completedAgents`, `currentAgent`, `previewData`, `genError`

**Key actions**: `setStep`, `setResumeUploaded`, `setJdText`, `setGenerating`, `addCompletedAgent`, `setCurrentAgent`, `setComplete`, `setPreviewData`, `setGenError`, `setConfirmResult`, `reset`

**Persistence exclusions**: `resumeFileB64` and `rewrittenFileB64` are excluded from localStorage due to size.

## API Client (`src/api/client.ts`)

Uses `fetch` (no Axios) for all backend calls.

| Function | Method | Endpoint | Returns |
|----------|--------|----------|---------|
| `parseResume(file)` | POST | `/api/parse-resume` | `ParsedResume` |
| `scrapeJd(url)` | POST | `/api/scrape-jd` | `{ text: string }` |
| `generateResume(...)` | POST | `/api/generate-resume` | `GenerateResponse` |
| `previewResume(...)` | POST | `/api/preview` | `PreviewResponse` |
| `confirmResume(...)` | POST | `/api/confirm` | `{ rewritten_file_b64: string }` |
| `generateCoverLetter(...)` | POST | `/api/generate-cover-letter` | `CoverLetterResponse` |
| `generateResumeStream(...)` | POST | `/api/generate-resume-stream` | SSE → callbacks |
| `previewResumeStream(...)` | POST | `/api/preview-stream` | SSE → callbacks |

### SSE Streaming

Both `generateResumeStream` and `previewResumeStream` return an `AbortController` for cancellation and accept callback objects:

- `onStarted(data)` — pipeline started with `run_id`
- `onAgentComplete(data)` — per-agent progress (scores, replacement counts)
- `onComplete(data)` — final result (replacements, scores, or full resume + PDF)
- `onError(detail)` — error message

## Key Types

### `src/types/resume.ts`

```ts
ResumeData          // Full structured resume (name, contact, summary, experience,
                    //   education, skills, certifications, ats_score, ats_score_before,
                    //   matched_keywords)
ExperienceItem      // job_title, company, location, start_date, end_date, bullets
EducationItem       // degree, institution, location, graduation_date, details
CertificationItem   // name, issuer, date
GenerateResponse    // { resume: ResumeData; rewritten_file_b64: string }
```

### `src/api/client.ts`

```ts
ParsedResume        // { text, html, file_b64, file_type }
TextReplacement     // { old: string, new: string }
PreviewResponse     // { replacements, ats_score_before, ats_score, matched_keywords, still_missing_keywords }
CoverLetterResponse // { cover_letter, suggested_job_title, linkedin_message }
SSEAgentEvent       // { agent, ats_score_before?, ats_score?, matched_keywords?, replacements_count?, has_pdf? }
SSECompleteEvent    // { resume: ResumeData, rewritten_file_b64: string }
```

## Running

```bash
npm install
npm run dev          # http://localhost:5173, proxies /api → :8000
npm run build        # production build to dist/
npm run lint         # ESLint check
npm run preview      # preview production build
```
