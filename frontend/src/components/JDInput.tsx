import { useState } from 'react';
import { scrapeJd } from '../api/client';

interface JDInputProps {
  onDone: (jdText: string) => void;
  onPreview?: (jdText: string) => void;
}

type Tab = 'paste' | 'url';

const TAB_LABELS: Record<Tab, string> = {
  paste: 'Paste Text',
  url: 'From URL',
};

const JDInput = ({ onDone, onPreview }: JDInputProps) => {
  const [tab, setTab] = useState<Tab>('paste');
  const [pasteText, setPasteText] = useState('');
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePasteSubmit = () => {
    if (!pasteText.trim()) {
      setError('Please paste the job description text.');
      return;
    }
    onDone(pasteText.trim());
  };

  const handleUrlSubmit = async () => {
    if (!url.trim()) {
      setError('Please enter a URL.');
      return;
    }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      setError('URL must start with http:// or https://');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const { text } = await scrapeJd(url.trim());
      if (onPreview) {
        onPreview(text);
      } else {
        onDone(text);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to scrape URL.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-800">Add Job Description</h2>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {(['paste', 'url'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => { setTab(t); setError(null); }}
            className={`px-5 py-2 text-sm font-medium border-b-2 transition-colors
              ${tab === t
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      {tab === 'paste' && (
        <div className="space-y-3">
          <textarea
            rows={10}
            className="w-full rounded-xl border border-gray-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            placeholder="Paste the full job description here…"
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
          />
          {onPreview ? (
            <button
              onClick={() => {
                if (!pasteText.trim()) {
                  setError('Please paste the job description text.');
                  return;
                }
                onPreview(pasteText.trim());
              }}
              className="w-full py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors"
            >
              Preview Changes
            </button>
          ) : (
            <button
              onClick={handlePasteSubmit}
              className="w-full py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors"
            >
              Continue
            </button>
          )}
        </div>
      )}

      {tab === 'url' && (
        <div className="space-y-3">
          <input
            type="url"
            className="w-full rounded-xl border border-gray-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="https://www.linkedin.com/jobs/view/..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
          />
          <button
            onClick={handleUrlSubmit}
            disabled={loading}
            className="w-full py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading && (
              <div className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            )}
            {loading ? 'Fetching JD…' : 'Fetch & Preview'}
          </button>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
          {error}
        </p>
      )}
    </div>
  );
};

export default JDInput;
