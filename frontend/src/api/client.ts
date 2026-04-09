import type { GenerateResponse } from '../types/resume';

const BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const err = await res.json();
      message = err.detail || JSON.stringify(err);
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export interface ParsedResume {
  text: string;      // plain text used by the AI pipeline
  html: string;      // styled HTML preserving fonts, colours and layout
  file_b64: string;  // base64-encoded original file bytes
  file_type: string; // "pdf" or "docx"
}

export async function parseResume(file: File): Promise<ParsedResume> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/parse-resume`, { method: 'POST', body: form });
  return handleResponse<ParsedResume>(res);
}

export async function scrapeJd(url: string): Promise<{ text: string }> {
  const res = await fetch(`${BASE}/scrape-jd`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  return handleResponse<{ text: string }>(res);
}

export async function generateResume(
  resume_text: string,
  jd_text: string,
  resume_file_b64: string,
  resume_file_type: string,
): Promise<GenerateResponse> {
  const res = await fetch(`${BASE}/generate-resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, jd_text, resume_file_b64, resume_file_type }),
  });
  return handleResponse<GenerateResponse>(res);
}

// ── Preview / Confirm (two-phase) ───────────────────────────────────────

export interface TextReplacement {
  old: string;
  new: string;
}

export interface PreviewResponse {
  replacements: TextReplacement[];
  ats_score_before: number;
  ats_score: number;
  matched_keywords: string[];
  still_missing_keywords: string[];
}

export async function previewResume(
  resume_text: string,
  jd_text: string,
): Promise<PreviewResponse> {
  const res = await fetch(`${BASE}/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, jd_text }),
  });
  return handleResponse<PreviewResponse>(res);
}

export async function confirmResume(
  resume_text: string,
  replacements: TextReplacement[],
  resume_file_b64: string,
  resume_file_type: string,
): Promise<{ rewritten_file_b64: string }> {
  const res = await fetch(`${BASE}/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, replacements, resume_file_b64, resume_file_type }),
  });
  return handleResponse<{ rewritten_file_b64: string }>(res);
}

// ── Cover letter ────────────────────────────────────────────────────────

export interface CoverLetterResponse {
  cover_letter: string;
  suggested_job_title: string;
  linkedin_message: string;
}

export async function generateCoverLetter(
  resume_text: string,
  jd_text: string,
  company_name?: string,
): Promise<CoverLetterResponse> {
  const res = await fetch(`${BASE}/generate-cover-letter`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, jd_text, company_name }),
  });
  return handleResponse<CoverLetterResponse>(res);
}

// ── SSE streaming ───────────────────────────────────────────────────────

export interface SSEAgentEvent {
  agent: string;
  ats_score_before?: number;
  ats_score?: number;
  matched_keywords?: string[];
  still_missing_keywords?: string[];
  replacements_count?: number;
  has_pdf?: boolean;
}

export interface SSECompleteEvent {
  resume: import('../types/resume').ResumeData;
  rewritten_file_b64: string;
}

export function generateResumeStream(
  resume_text: string,
  jd_text: string,
  resume_file_b64: string,
  resume_file_type: string,
  callbacks: {
    onStarted?: (data: { run_id: string }) => void;
    onAgentComplete?: (data: SSEAgentEvent) => void;
    onComplete?: (data: SSECompleteEvent) => void;
    onError?: (detail: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/generate-resume-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, jd_text, resume_file_b64, resume_file_type }),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      callbacks.onError?.(err.detail || `HTTP ${res.status}`);
      return;
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const lines = part.split('\n');
        let eventType = '';
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7);
          else if (line.startsWith('data: ')) data = line.slice(6);
        }
        if (!eventType || !data) continue;

        try {
          const parsed = JSON.parse(data);
          switch (eventType) {
            case 'started': callbacks.onStarted?.(parsed); break;
            case 'agent_complete': callbacks.onAgentComplete?.(parsed); break;
            case 'complete': callbacks.onComplete?.(parsed); break;
            case 'error': callbacks.onError?.(parsed.detail); break;
          }
        } catch {
          // skip malformed events
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      callbacks.onError?.(err.message || 'Stream connection failed');
    }
  });

  return controller;
}

// ── Preview SSE streaming ───────────────────────────────────────────────

export function previewResumeStream(
  resume_text: string,
  jd_text: string,
  callbacks: {
    onAgentComplete?: (data: SSEAgentEvent) => void;
    onComplete?: (data: PreviewResponse) => void;
    onError?: (detail: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/preview-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text, jd_text }),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      callbacks.onError?.(err.detail || `HTTP ${res.status}`);
      return;
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const lines = part.split('\n');
        let eventType = '';
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7);
          else if (line.startsWith('data: ')) data = line.slice(6);
        }
        if (!eventType || !data) continue;
        try {
          const parsed = JSON.parse(data);
          switch (eventType) {
            case 'agent_complete': callbacks.onAgentComplete?.(parsed); break;
            case 'complete': callbacks.onComplete?.(parsed); break;
            case 'error': callbacks.onError?.(parsed.detail || 'Preview failed'); break;
          }
        } catch {
          // skip malformed events
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      callbacks.onError?.(err.message || 'Preview stream connection failed');
    }
  });

  return controller;
}

// ── Pipeline runs (for /info page) ──────────────────────────────────────

export interface AgentStep {
  name: string;
  duration_ms: number;
  input_summary: Record<string, unknown>;
  output: Record<string, unknown>;
}

export interface PipelineRun {
  _id: string;
  created_at: string;
  completed_at?: string;
  status: 'running' | 'completed' | 'failed';
  resume_summary: string;
  jd_summary: string;
  agents: AgentStep[];
  final_result?: {
    ats_score_before?: number;
    ats_score: number;
    matched_keywords: string[];
    replacements_count: number;
    name: string;
  };
  has_compiled_pdf?: boolean;
  error?: string;
}

export async function getPipelineRunsStatus(): Promise<{ db_connected: boolean }> {
  const res = await fetch(`${BASE}/pipeline-runs/status`);
  return handleResponse<{ db_connected: boolean }>(res);
}

export async function getPipelineRuns(limit = 20, skip = 0): Promise<PipelineRun[]> {
  const res = await fetch(`${BASE}/pipeline-runs?limit=${limit}&skip=${skip}`);
  return handleResponse<PipelineRun[]>(res);
}

export async function getPipelineRun(id: string): Promise<PipelineRun> {
  const res = await fetch(`${BASE}/pipeline-runs/${encodeURIComponent(id)}`);
  return handleResponse<PipelineRun>(res);
}

export function getPipelineRunPdfUrl(id: string): string {
  return `${BASE}/pipeline-runs/${encodeURIComponent(id)}/pdf`;
}
