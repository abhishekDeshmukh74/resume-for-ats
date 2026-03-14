# Pass-ATS Frontend

React + TypeScript SPA that drives the 4-step resume tailoring workflow, with a dedicated pipeline runs inspector page.

## Tech Stack

- **React 19** with TypeScript
- **Vite 8** — dev server with proxy (`/api → http://localhost:8000`)
- **Tailwind CSS 4** — utility-first styling
- **react-router-dom** — client-side routing (`/` home, `/info` pipeline runs)

## Architecture

```
src/
├── App.tsx                  # Root — routing (/ and /info) + step wizard + data flow
├── main.tsx                 # React DOM entry
├── index.css                # Tailwind base styles
├── api/
│   └── client.ts            # Axios HTTP client (all backend calls)
├── components/
│   ├── ResumeUpload.tsx     # Step 1 — PDF upload + parse
│   ├── JDInput.tsx          # Step 2 — Job description input / scrape
│   ├── ResumePreview.tsx    # Step 4 — PDF preview (iframe) + download + ATS scores
│   └── StepIndicator.tsx    # Progress bar (steps 1–4)
├── pages/
│   └── InfoPage.tsx         # Pipeline runs inspector (list + detail view)
└── types/
    └── resume.ts            # Shared TypeScript interfaces
```

## Pages

### Home (`/`)

The 4-step resume tailoring wizard:

| Step | Component | Backend Call | Description |
|------|-----------|-------------|-------------|
| 1 | `ResumeUpload` | `POST /api/parse-resume` | Upload PDF, receive parsed text + base64 |
| 2 | `JDInput` | `POST /api/scrape-jd` | Paste or scrape a job description |
| 3 | (auto) | `POST /api/generate-resume` | AI-tailor resume → structured data + rewritten PDF |
| 4 | `ResumePreview` | — | Preview PDF, ATS score before/after, download |

### Pipeline Runs (`/info`)

Full pipeline inspection page (`InfoPage.tsx`):
- Lists recent pipeline runs with status badges (running / completed / failed)
- Run detail view with:
  - ATS score panel (before → after with delta badge)
  - Per-agent expandable cards with duration bars
  - Input/output toggle tabs with recursive JSON tree viewer
  - Matched keywords and replacement count
  - Error display for failed runs

## API Client (`src/api/client.ts`)

| Function | Method | Endpoint | Returns |
|----------|--------|----------|---------|
| `parseResume(file)` | POST | `/api/parse-resume` | `ParsedResumeResponse` |
| `scrapeJd(url)` | POST | `/api/scrape-jd` | `TextResponse` |
| `generateResume(...)` | POST | `/api/generate-resume` | `GenerateResponse` |
| `getPipelineRuns(limit, skip)` | GET | `/api/pipeline-runs` | `PipelineRun[]` |
| `getPipelineRun(id)` | GET | `/api/pipeline-runs/{id}` | `PipelineRun` |

## Key Types (`src/types/resume.ts`)

```ts
ResumeData          // Full structured resume (name, contact, summary, experience,
                    //   education, skills, certifications, ats_score, ats_score_before,
                    //   matched_keywords)
ExperienceItem      // company, title, location, dates, bullets
EducationItem       // institution, degree, details, dates
CertificationItem   // name, issuer, date
GenerateResponse    // { resume: ResumeData; rewritten_file_b64: string }
ParsedResumeResponse // { resume_text, resume_html, resume_file_b64, resume_file_type }
TextResponse        // { text: string }
```

Additional types in `client.ts`:

```ts
AgentStep           // { agent_name, duration_ms, input_summary, output_data }
PipelineRun         // { id, status, created_at, completed_at, resume_text_preview,
                    //   jd_text_preview, agents, final_result, error }
```

## Running

```bash
npm install
npm run dev          # http://localhost:5173, proxies /api → :8000
npm run build        # production build to dist/
npm run lint         # ESLint check
```
