import type { ResumeData } from '../types/resume';
import { downloadPdf } from '../api/client';
import { useState } from 'react';

interface Props {
  resume: ResumeData;
  onStartOver: () => void;
}

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 90 ? 'text-green-700 bg-green-50 border-green-200' :
    score >= 70 ? 'text-blue-700 bg-blue-50 border-blue-200' :
    score >= 50 ? 'text-yellow-700 bg-yellow-50 border-yellow-200' :
                  'text-red-700 bg-red-50 border-red-200';
  const bar =
    score >= 90 ? 'bg-green-500' :
    score >= 70 ? 'bg-blue-500' :
    score >= 50 ? 'bg-yellow-400' :
                  'bg-red-500';
  const label =
    score >= 90 ? 'Excellent' :
    score >= 70 ? 'Strong' :
    score >= 50 ? 'Moderate' :
                  'Weak';

  return (
    <div className={`rounded-xl border p-4 ${color}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold">ATS Match Score</span>
        <span className="text-2xl font-bold">{score}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
        <div className={`h-2 rounded-full transition-all ${bar}`} style={{ width: `${score}%` }} />
      </div>
      <p className="text-xs mt-1 opacity-75">{label} — resume keyword coverage against the job description</p>
    </div>
  );
}

export default function ResumePreview({ resume, onStartOver }: Props) {
  const [downloading, setDownloading] = useState(false);
  const [dlError, setDlError] = useState<string | null>(null);

  const handleDownload = async () => {
    setDlError(null);
    setDownloading(true);
    try {
      const blob = await downloadPdf(resume);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const safeName = resume.name.replace(/\s+/g, '_') || 'resume';
      a.href = url;
      a.download = `${safeName}_resume.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setDlError(e instanceof Error ? e.message : 'PDF download failed.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Action bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-semibold text-gray-800">Your Tailored Resume</h2>
        <div className="flex gap-2">
          <button
            onClick={onStartOver}
            className="px-4 py-2 rounded-xl border border-gray-300 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Start Over
          </button>
          <button
            onClick={handleDownload}
            disabled={downloading}
            className="px-4 py-2 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {downloading && (
              <div className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            )}
            {downloading ? 'Generating…' : 'Download PDF'}
          </button>
        </div>
      </div>

      {dlError && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
          {dlError}
        </p>
      )}

      {/* ATS Score panel */}
      {resume.ats_score != null && (
        <div className="space-y-3">
          <ScoreBadge score={resume.ats_score} />
          {resume.matched_keywords && resume.matched_keywords.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Matched Keywords</p>
              <div className="flex flex-wrap gap-1.5">
                {resume.matched_keywords.map((kw, i) => (
                  <span key={i} className="px-2 py-0.5 bg-green-50 text-green-700 text-xs rounded-full border border-green-200">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Resume card */}
      <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-8 max-w-3xl mx-auto font-serif text-gray-900">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-3xl font-bold text-blue-700">{resume.name}</h1>
          <p className="text-sm text-gray-500 mt-1">
            {[resume.email, resume.phone, resume.location, resume.linkedin, resume.github]
              .filter(Boolean)
              .join('  |  ')}
          </p>
        </div>

        <hr className="border-gray-200 mb-4" />

        {/* Summary */}
        {resume.summary && (
          <Section title="Professional Summary">
            <p className="text-sm leading-relaxed text-gray-700">{resume.summary}</p>
          </Section>
        )}

        {/* Experience */}
        {resume.experience.length > 0 && (
          <Section title="Experience">
            {resume.experience.map((exp, i) => (
              <div key={i} className="mb-4">
                <div className="flex flex-wrap justify-between items-baseline">
                  <span className="font-semibold text-sm">{exp.job_title} — {exp.company}</span>
                  <span className="text-xs text-gray-500">{exp.start_date} – {exp.end_date}</span>
                </div>
                {exp.location && <p className="text-xs text-gray-400">{exp.location}</p>}
                <ul className="mt-1 space-y-1 list-disc list-inside">
                  {exp.bullets.map((b, j) => (
                    <li key={j} className="text-sm text-gray-700">{b}</li>
                  ))}
                </ul>
              </div>
            ))}
          </Section>
        )}

        {/* Education */}
        {resume.education.length > 0 && (
          <Section title="Education">
            {resume.education.map((edu, i) => (
              <div key={i} className="mb-3">
                <div className="flex flex-wrap justify-between items-baseline">
                  <span className="font-semibold text-sm">{edu.degree} — {edu.institution}</span>
                  <span className="text-xs text-gray-500">{edu.graduation_date}</span>
                </div>
                {edu.location && <p className="text-xs text-gray-400">{edu.location}</p>}
                {edu.details?.map((d, j) => (
                  <p key={j} className="text-sm text-gray-600">• {d}</p>
                ))}
              </div>
            ))}
          </Section>
        )}

        {/* Skills */}
        {resume.skills.length > 0 && (
          <Section title="Skills">
            <div className="flex flex-wrap gap-2">
              {resume.skills.map((s, i) => (
                <span key={i} className="px-2.5 py-0.5 bg-blue-50 text-blue-700 text-xs rounded-full border border-blue-100">
                  {s}
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* Certifications */}
        {resume.certifications.length > 0 && (
          <Section title="Certifications">
            {resume.certifications.map((c, i) => (
              <p key={i} className="text-sm text-gray-700">
                <span className="font-medium">{c.name}</span>
                {c.issuer && ` — ${c.issuer}`}
                {c.date && ` (${c.date})`}
              </p>
            ))}
          </Section>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <h3 className="text-xs font-bold uppercase tracking-widest text-blue-600 mb-1">{title}</h3>
      <hr className="border-gray-200 mb-3" />
      {children}
    </div>
  );
}
