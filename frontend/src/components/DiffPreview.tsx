import { useState } from 'react';
import type { TextReplacement } from '../api/client';

interface DiffPreviewProps {
  replacements: TextReplacement[];
  atsScoreBefore: number;
  atsScore: number;
  matchedKeywords: string[];
  stillMissing: string[];
  onConfirm: (approved: TextReplacement[]) => void;
  onCancel: () => void;
  loading?: boolean;
}

const DiffPreview = ({
  replacements,
  atsScoreBefore,
  atsScore,
  matchedKeywords,
  stillMissing,
  onConfirm,
  onCancel,
  loading,
}: DiffPreviewProps) => {
  const [checked, setChecked] = useState<boolean[]>(() => replacements.map(() => true));
  const [edits, setEdits] = useState<string[]>(() => replacements.map((r) => r.new));

  const toggleAll = (on: boolean) => setChecked(replacements.map(() => on));

  const handleConfirm = () => {
    const approved: TextReplacement[] = [];
    for (let i = 0; i < replacements.length; i++) {
      if (checked[i]) {
        approved.push({ old: replacements[i].old, new: edits[i] });
      }
    }
    onConfirm(approved);
  };

  const approvedCount = checked.filter(Boolean).length;

  return (
    <div className="space-y-5">
      {/* Score summary */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Before:</span>
          <span className="font-bold text-gray-700">{atsScoreBefore}%</span>
        </div>
        <span className="text-gray-400">&rarr;</span>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">After:</span>
          <span className="font-bold text-blue-600">{atsScore}%</span>
        </div>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
          atsScore - atsScoreBefore > 0
            ? 'bg-green-100 text-green-700'
            : 'bg-gray-100 text-gray-500'
        }`}>
          {atsScore - atsScoreBefore > 0 ? '+' : ''}{atsScore - atsScoreBefore}%
        </span>
      </div>

      {/* Keywords */}
      {matchedKeywords.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Matched Keywords</p>
          <div className="flex flex-wrap gap-1">
            {matchedKeywords.map((kw) => (
              <span key={kw} className="px-2 py-0.5 bg-green-50 text-green-700 text-xs rounded-full border border-green-200">{kw}</span>
            ))}
          </div>
        </div>
      )}
      {stillMissing.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Still Missing</p>
          <div className="flex flex-wrap gap-1">
            {stillMissing.map((kw) => (
              <span key={kw} className="px-2 py-0.5 bg-red-50 text-red-600 text-xs rounded-full border border-red-200">{kw}</span>
            ))}
          </div>
        </div>
      )}

      {/* Replacement list */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">
            Proposed Changes ({approvedCount}/{replacements.length} approved)
          </h3>
          <div className="flex gap-2 text-xs">
            <button onClick={() => toggleAll(true)} className="text-blue-600 hover:underline">Select All</button>
            <button onClick={() => toggleAll(false)} className="text-gray-500 hover:underline">Deselect All</button>
          </div>
        </div>

        <div className="max-h-[50vh] overflow-y-auto space-y-3 pr-1">
          {replacements.map((r, i) => (
            <div key={i} className={`rounded-xl border p-3 transition-colors ${checked[i] ? 'border-blue-200 bg-blue-50/30' : 'border-gray-200 bg-gray-50/50 opacity-60'}`}>
              <label className="flex items-start gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={checked[i]}
                  onChange={(e) => {
                    const next = [...checked];
                    next[i] = e.target.checked;
                    setChecked(next);
                  }}
                  className="mt-1 rounded"
                />
                <div className="flex-1 min-w-0 space-y-2 text-sm">
                  <div>
                    <span className="text-xs font-semibold text-red-500 uppercase">Old</span>
                    <p className="bg-red-50 border border-red-100 rounded-lg px-2 py-1 text-red-800 break-words whitespace-pre-wrap">{r.old}</p>
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-green-600 uppercase">New</span>
                    <textarea
                      value={edits[i]}
                      onChange={(e) => {
                        const next = [...edits];
                        next[i] = e.target.value;
                        setEdits(next);
                      }}
                      rows={Math.max(2, edits[i].split('\n').length)}
                      className="w-full bg-green-50 border border-green-100 rounded-lg px-2 py-1 text-green-800 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-green-400"
                    />
                  </div>
                </div>
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onCancel}
          disabled={loading}
          className="flex-1 py-2.5 rounded-xl border border-gray-300 text-sm text-gray-600 hover:bg-gray-50 transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={handleConfirm}
          disabled={loading || approvedCount === 0}
          className="flex-1 py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Compiling PDF&hellip;
            </>
          ) : (
            `Confirm ${approvedCount} Change${approvedCount !== 1 ? 's' : ''}`
          )}
        </button>
      </div>
    </div>
  );
};

export default DiffPreview;
