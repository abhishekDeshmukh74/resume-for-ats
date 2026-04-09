import { useEffect, useState, useCallback } from 'react';
import { getPipelineRuns, getPipelineRun, getPipelineRunPdfUrl, getPipelineRunsStatus } from '../api/client';
import type { PipelineRun, AgentStep } from '../api/client';
import { Link } from 'react-router-dom';

const AGENT_LABELS: Record<string, string> = {
  extract_keywords: 'Keyword Extractor',
  analyse_resume: 'Resume Analyser',
  score_before: 'Pre-Rewrite Score',
  rewrite_sections: 'Rewriter',
  qa_deduplicate: 'QA & Dedup',
  score_extract: 'ATS Scorer',
};

const AGENT_COLOURS: Record<string, string> = {
  extract_keywords: 'bg-purple-500',
  analyse_resume: 'bg-blue-500',
  score_before: 'bg-cyan-500',
  rewrite_sections: 'bg-amber-500',
  qa_deduplicate: 'bg-emerald-500',
  score_extract: 'bg-rose-500',
};

const STATUS_BADGE: Record<string, string> = {
  running: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  completed: 'bg-green-100 text-green-700 border-green-300',
  failed: 'bg-red-100 text-red-700 border-red-300',
};

function formatDate(iso: string) {
  return new Date(iso).toLocaleString();
}

function formatDuration(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ── JSON code block with copy ────────────────────────────────────────────

function JsonBlock({ value }: { value: unknown }) {
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(value, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(json).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="relative">
      <button
        onClick={handleCopy}
        title="Copy JSON"
        className="absolute top-2 right-2 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs font-medium transition-colors z-10"
      >
        {copied ? (
          <>
            <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            Copied!
          </>
        ) : (
          <>
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-4 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Copy
          </>
        )}
      </button>
      <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 pt-10 overflow-auto max-h-96 text-xs leading-relaxed font-mono whitespace-pre">
        {json}
      </pre>
    </div>
  );
}

// ── Agent step card ──────────────────────────────────────────────────────

function AgentCard({ step, maxDuration }: { step: AgentStep; maxDuration: number }) {
  const [expanded, setExpanded] = useState<'input' | 'output' | null>(null);
  const barWidth = maxDuration > 0 ? (step.duration_ms / maxDuration) * 100 : 0;

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(expanded ? null : 'output')}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors text-left"
      >
        <div className={`w-3 h-3 rounded-full shrink-0 ${AGENT_COLOURS[step.name] ?? 'bg-gray-400'}`} />
        <span className="font-medium text-sm text-gray-800 w-40 shrink-0">
          {AGENT_LABELS[step.name] ?? step.name}
        </span>
        {/* duration bar */}
        <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden relative">
          <div
            className={`h-full rounded-full ${AGENT_COLOURS[step.name] ?? 'bg-gray-400'} opacity-70 transition-all`}
            style={{ width: `${barWidth}%` }}
          />
          <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-gray-600">
            {formatDuration(step.duration_ms)}
          </span>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="border-t border-gray-200 bg-gray-50">
          {/* Toggle tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setExpanded('input')}
              className={`px-4 py-2 text-xs font-semibold border-b-2 transition-colors ${
                expanded === 'input'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Input
            </button>
            <button
              onClick={() => setExpanded('output')}
              className={`px-4 py-2 text-xs font-semibold border-b-2 transition-colors ${
                expanded === 'output'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Output
            </button>
          </div>
          <div className="p-4">
            <JsonBlock
              value={expanded === 'input' ? step.input_summary : step.output}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Run detail panel ─────────────────────────────────────────────────────

function RunDetail({ runId, onBack }: { runId: string; onBack: () => void }) {
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPipelineRun(runId)
      .then((data) => { if (!cancelled) setRun(data); })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [runId]);

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-sm text-blue-600 hover:underline">&larr; Back to runs</button>
        <p className="text-red-600">{error ?? 'Run not found.'}</p>
      </div>
    );
  }

  const maxDuration = Math.max(...run.agents.map((a) => a.duration_ms), 1);
  const totalDuration = run.agents.reduce((s, a) => s + a.duration_ms, 0);

  return (
    <div className="space-y-6">
      <button onClick={onBack} className="text-sm text-blue-600 hover:underline">&larr; Back to runs</button>

      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="text-lg font-bold text-gray-800">
            {run.final_result?.name ?? 'Pipeline Run'}
          </h2>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${STATUS_BADGE[run.status]}`}>
            {run.status}
          </span>
          <span className="text-xs text-gray-400">Total: {formatDuration(totalDuration)}</span>
        </div>
        <p className="text-xs text-gray-400">{formatDate(run.created_at)}</p>
      </div>

      {/* ATS score */}
      {run.final_result && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-blue-800">ATS Score</p>
              <p className="text-xs text-blue-500">
                {run.final_result.replacements_count} replacements &middot;{' '}
                {run.final_result.matched_keywords?.length ?? 0} keywords matched
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-blue-700">{run.final_result.ats_score}%</span>
              {run.has_compiled_pdf && (
                <a
                  href={getPipelineRunPdfUrl(run._id)}
                  download
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Download Resume
                </a>
              )}
            </div>
          </div>
          {run.final_result.ats_score_before != null && (
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-gray-500">Before:</span>
                <span className="font-semibold text-gray-700">{run.final_result.ats_score_before}%</span>
              </div>
              <span className="text-gray-300">&rarr;</span>
              <div className="flex items-center gap-2">
                <span className="text-gray-500">After:</span>
                <span className="font-semibold text-blue-700">{run.final_result.ats_score}%</span>
              </div>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                run.final_result.ats_score - run.final_result.ats_score_before > 0
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-500'
              }`}>
                {run.final_result.ats_score - run.final_result.ats_score_before > 0 ? '+' : ''}
                {run.final_result.ats_score - run.final_result.ats_score_before}%
              </span>
            </div>
          )}
        </div>
      )}

      {run.error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
          {run.error}
        </div>
      )}

      {/* Summaries */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border border-gray-200 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Resume (preview)</p>
          <p className="text-sm text-gray-600 line-clamp-4">{run.resume_summary}</p>
        </div>
        <div className="border border-gray-200 rounded-xl p-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Job Description (preview)</p>
          <p className="text-sm text-gray-600 line-clamp-4">{run.jd_summary}</p>
        </div>
      </div>

      {/* Agent timeline */}
      <div>
        <h3 className="text-sm font-bold text-gray-700 mb-3">Agent Pipeline ({run.agents.length} steps)</h3>
        <div className="space-y-2">
          {run.agents.map((step, i) => (
            <AgentCard key={i} step={step} maxDuration={maxDuration} />
          ))}
        </div>
      </div>

      {/* Matched keywords */}
      {run.final_result?.matched_keywords && run.final_result.matched_keywords.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Matched Keywords</p>
          <div className="flex flex-wrap gap-1.5">
            {run.final_result.matched_keywords.map((kw) => (
              <span key={kw} className="px-2 py-0.5 bg-green-50 text-green-700 text-xs rounded-full border border-green-200">
                {kw}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main InfoPage ────────────────────────────────────────────────────────

const InfoPage = () => {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dbConnected, setDbConnected] = useState<boolean | null>(null);

  const fetchRuns = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([getPipelineRuns(), getPipelineRunsStatus()])
      .then(([data, status]) => {
        setRuns(data);
        setDbConnected(status.db_connected);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  if (selectedId) {
    return (
      <div className="min-h-screen bg-linear-to-br from-slate-50 to-blue-50 flex flex-col">
        <Header />
        <main className="flex-1 flex flex-col items-center py-10 px-4">
          <div className="w-full max-w-3xl bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
            <RunDetail runId={selectedId} onBack={() => setSelectedId(null)} />
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 to-blue-50 flex flex-col">
      <Header />
      <main className="flex-1 flex flex-col items-center py-10 px-4">
        <div className="w-full max-w-3xl space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-gray-800">Pipeline Runs</h1>
              {dbConnected !== null && (
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${
                  dbConnected
                    ? 'bg-green-100 text-green-700 border-green-300'
                    : 'bg-red-100 text-red-600 border-red-300'
                }`}>
                  {dbConnected ? 'DB connected' : 'DB not connected'}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={fetchRuns}
                disabled={loading}
                className="text-sm text-blue-600 hover:underline disabled:opacity-50"
              >
                Refresh
              </button>
              <Link to="/" className="text-sm text-blue-600 hover:underline">
                &larr; Back to app
              </Link>
            </div>
          </div>

          {loading && (
            <div className="flex justify-center py-20">
              <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
            </div>
          )}

          {error && (
            <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg p-3">{error}</p>
          )}

          {!loading && !error && runs.length === 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-10 text-center space-y-2">
              <p className="text-gray-500">No pipeline runs yet.</p>
              {dbConnected === false && (
                <p className="text-xs text-red-500">
                  MongoDB is not connected — set <code className="bg-red-50 px-1 rounded">MONGODB_URL</code> in your <code className="bg-red-50 px-1 rounded">.env</code> to enable run tracking.
                </p>
              )}
              {dbConnected === true && (
                <p className="text-xs text-gray-400">Generate a resume to see agent I/O here.</p>
              )}
            </div>
          )}

          {runs.map((run) => {
            const totalMs = run.agents?.reduce((s, a) => s + a.duration_ms, 0) ?? 0;
            return (
              <button
                key={run._id}
                onClick={() => setSelectedId(run._id)}
                className="w-full text-left bg-white rounded-2xl shadow-sm border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-gray-800">
                        {run.final_result?.name ?? 'Pipeline Run'}
                      </span>
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${STATUS_BADGE[run.status]}`}>
                        {run.status}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400">{formatDate(run.created_at)}</p>
                    <p className="text-sm text-gray-500 truncate">{run.jd_summary}</p>
                  </div>
                  <div className="text-right shrink-0 space-y-1">
                    {run.final_result?.ats_score != null && (
                      <p className="text-2xl font-bold text-blue-600">{run.final_result.ats_score}%</p>
                    )}
                    <p className="text-xs text-gray-400">
                      {run.agents?.length ?? 0} agents &middot; {formatDuration(totalMs)}
                    </p>
                  </div>
                </div>
                {/* Mini agent bar */}
                {run.agents && run.agents.length > 0 && (
                  <div className="flex gap-0.5 mt-3 h-1.5 rounded-full overflow-hidden bg-gray-100">
                    {run.agents.map((a, i) => (
                      <div
                        key={i}
                        className={`${AGENT_COLOURS[a.name] ?? 'bg-gray-400'} opacity-70`}
                        style={{ flex: a.duration_ms }}
                        title={`${AGENT_LABELS[a.name] ?? a.name}: ${formatDuration(a.duration_ms)}`}
                      />
                    ))}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </main>
    </div>
  );
};

// ── Shared header ────────────────────────────────────────────────────────

function Header() {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3 shadow-sm">
      <Link to="/" className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <span className="text-lg font-bold text-gray-800">Resume for ATS</span>
      </Link>
      <span className="text-xs text-gray-400 ml-1">AI-Powered Resume Tailor</span>
      <div className="flex-1" />
      <Link to="/info" className="text-sm text-blue-600 font-medium hover:underline">
        Pipeline Runs
      </Link>
    </header>
  );
}

export default InfoPage;
