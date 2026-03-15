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
