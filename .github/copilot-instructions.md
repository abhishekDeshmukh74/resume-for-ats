# Copilot Instructions — Resume for ATS

## Project Overview

AI-powered resume tailoring tool. Users upload a PDF/LaTeX resume, provide a job description, and receive an ATS-optimised resume via a LangGraph multi-agent pipeline. The output preserves original formatting — only text content changes.

## Architecture

- **Backend**: Python 3.11+ / FastAPI at `backend/`
- **Frontend**: React 19 / Vite 8 / TypeScript / Tailwind CSS 4 at `frontend/`
- **AI Pipeline**: LangGraph StateGraph with parallel section rewriters at `backend/services/agents/`
- **Database**: MongoDB (optional, for pipeline run tracking) via pymongo (sync)

## Key Technical Decisions

### LLM Provider
- All LLM calls go through **litellm** (`backend/services/agents/llm.py`), NOT langchain-groq or langchain-google-genai
- 7 providers: Groq (default), Gemini, OpenAI, Anthropic, DeepSeek, OpenRouter, Ollama
- Custom `_LiteLLMChat` wrapper conforms to LangChain `BaseChatModel` interface
- Temperature 0.2, max_tokens 8192 for all providers

### Pipeline Architecture
- 3 section rewriters run **in parallel** (skills, summary, experience) via LangGraph fan-out
- Conditional refinement loop: if score < 90 AND pass == 0, run refinement agent + QA again
- ATS scoring is **purely algorithmic** via `rapidfuzz` (exact + synonym + fuzzy matching + section weighting + stuffing penalty). LLM is used only for structured data extraction
- State uses merge reducers for lists (raw_replacements, replacements, jd_keywords) and overwrite for scalars

### Frontend
- State management: **Zustand 5** with persist middleware (`frontend/src/store/appStore.ts`)
- API client uses **fetch** (not Axios) at `frontend/src/api/client.ts`
- SSE streaming for real-time agent progress updates
- 5-step wizard: Upload → JD → Generate → Review Diffs → Preview PDF

### PDF Processing
- PDF text replacement: PyMuPDF in-place redact+rewrite (`backend/services/rewriter.py`)
- LaTeX: source patching + xelatex/pdflatex compilation (`backend/services/latex_rewriter.py`)
- Text extraction: PyMuPDF for PDF, regex-based for LaTeX (`backend/services/parser.py`, `latex_parser.py`)

## File Layout Quick Reference

| Path | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry, CORS, router registration, JSON logging |
| `backend/models.py` | All Pydantic request/response models |
| `backend/routers/generate.py` | Main endpoints: generate, preview, confirm, cover letter |
| `backend/routers/stream.py` | SSE streaming endpoints |
| `backend/routers/pipeline.py` | Pipeline run inspection + PDF download |
| `backend/services/agents/graph.py` | LangGraph StateGraph wiring, tracking, public API |
| `backend/services/agents/state.py` | AgentState TypedDict with annotated reducers |
| `backend/services/agents/llm.py` | litellm wrapper + JSON parsing utilities |
| `backend/services/agents/scorer.py` | ATS scoring (Agents 3 & 6) |
| `backend/services/agents/keyword_matcher.py` | Algorithmic scoring engine (rapidfuzz) |
| `backend/services/agents/skills_rewriter.py` | Parallel Agent 4a |
| `backend/services/agents/summary_rewriter.py` | Parallel Agent 4b |
| `backend/services/agents/experience_rewriter.py` | Parallel Agent 4c |
| `backend/services/agents/refinement_agent.py` | Conditional refinement (Agent 6b) |
| `backend/services/agents/cover_letter.py` | Cover letter + LinkedIn message generation |
| `frontend/src/App.tsx` | Root component — routing, 5-step wizard, SSE streaming |
| `frontend/src/store/appStore.ts` | Zustand store (persisted to localStorage) |
| `frontend/src/api/client.ts` | All backend API calls (fetch + SSE helpers) |

## Common Tasks

### Adding a new LLM provider
1. Add provider config in `backend/services/agents/llm.py` (`_PROVIDER_CONFIG` dict)
2. Add env var to `.env.example` and document in `backend/README.md`

### Adding a new pipeline agent
1. Create agent file in `backend/services/agents/`
2. Add node to the StateGraph in `graph.py`
3. Update `AgentState` in `state.py` if new fields are needed
4. Wrap with `_tracked()` in `graph.py` for pipeline run tracking

### Adding a new API endpoint
1. Add route handler in the appropriate `backend/routers/` file
2. Add Pydantic models in `backend/models.py`
3. Add corresponding client function in `frontend/src/api/client.ts`

### Running the project
```bash
make install   # install all deps
make dev       # start backend + frontend
```

## Conventions

- Backend uses **sync** MongoDB (pymongo), not async Motor
- All agent LLM calls return JSON; use `parse_llm_json()` from `llm.py` for safe extraction
- Replacement lists use `_merge_lists` reducer (extend, not replace) in AgentState
- Frontend components are in `frontend/src/components/`, pages in `frontend/src/pages/`
- Environment variables are documented in `backend/README.md` (full table)

## Documentation Map

| Document | Scope |
|----------|-------|
| `ARCHITECTURE.md` (root) | High-level system overview |
| `backend/ARCHITECTURE.md` | Backend internals, request flows, data models |
| `backend/README.md` | API endpoints, env vars, dependencies |
| `backend/services/agents/AGENTS.md` | Full pipeline agent documentation |
| `backend/services/PARSER.md` | PDF parser internals |
| `backend/services/REWRITER.md` | PDF rewriter internals |
| `frontend/README.md` | Frontend architecture, components, store, API client |
