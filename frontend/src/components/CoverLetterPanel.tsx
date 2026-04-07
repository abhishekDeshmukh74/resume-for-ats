import { useState } from 'react';
import { generateCoverLetter, type CoverLetterResponse } from '../api/client';

interface CoverLetterPanelProps {
  resumeText: string;
  jdText: string;
}

const CoverLetterPanel = ({ resumeText, jdText }: CoverLetterPanelProps) => {
  const [result, setResult] = useState<CoverLetterResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [companyName, setCompanyName] = useState('');

  const handleGenerate = async () => {
    setError(null);
    setLoading(true);
    try {
      const data = await generateCoverLetter(resumeText, jdText, companyName || undefined);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Generation failed.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (!result) {
    return (
      <div className="space-y-3 border border-gray-200 rounded-xl p-4 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700">Cover Letter & Outreach</h3>
        <input
          type="text"
          placeholder="Company name (optional)"
          value={companyName}
          onChange={(e) => setCompanyName(e.target.value)}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full py-2 rounded-xl bg-purple-600 text-white text-sm font-semibold hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Generating&hellip;
            </>
          ) : (
            'Generate Cover Letter & Outreach'
          )}
        </button>
        {error && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4 border border-gray-200 rounded-xl p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Cover Letter & Outreach</h3>
        <button
          onClick={() => setResult(null)}
          className="text-xs text-gray-500 hover:underline"
        >
          Regenerate
        </button>
      </div>

      {/* Suggested Job Title */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-gray-500 uppercase">Suggested Title:</span>
        <span className="text-sm font-medium text-blue-700">{result.suggested_job_title}</span>
      </div>

      {/* Cover Letter */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-500 uppercase">Cover Letter</span>
          <button
            onClick={() => copyToClipboard(result.cover_letter)}
            className="text-xs text-blue-600 hover:underline"
          >
            Copy
          </button>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-sm text-gray-700 whitespace-pre-wrap max-h-60 overflow-y-auto">
          {result.cover_letter}
        </div>
      </div>

      {/* LinkedIn Message */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-500 uppercase">LinkedIn Outreach</span>
          <button
            onClick={() => copyToClipboard(result.linkedin_message)}
            className="text-xs text-blue-600 hover:underline"
          >
            Copy
          </button>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-sm text-gray-700 whitespace-pre-wrap">
          {result.linkedin_message}
        </div>
      </div>
    </div>
  );
};

export default CoverLetterPanel;
