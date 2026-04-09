import { useState, useRef, useCallback } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import StepIndicator from './components/StepIndicator';
import ResumeUpload from './components/ResumeUpload';
import JDInput from './components/JDInput';
import ResumePreview from './components/ResumePreview';
import DiffPreview from './components/DiffPreview';
import CoverLetterPanel from './components/CoverLetterPanel';
import InfoPage from './pages/InfoPage';
import {
  generateResumeStream,
  previewResumeStream,
  confirmResume,
  type TextReplacement,
} from './api/client';
import { useAppStore } from './store/appStore';

const STEPS = ['Upload Resume', 'Job Description', 'Generate', 'Review', 'Preview'];

const AGENT_LABELS: Record<string, string> = {
  extract_keywords: 'Extracting JD keywords',
  analyse_resume: 'Analysing resume',
  score_before: 'Scoring original resume',
  rewrite_skills: 'Rewriting skills',
  rewrite_summary: 'Rewriting summary',
  rewrite_experience: 'Rewriting experience',
  qa_deduplicate: 'QA & deduplication',
  score_extract: 'Scoring & extracting data',
  refine_rewrite: 'Refinement pass',
  refine_qa: 'Refinement QA',
  compile_pdf: 'Compiling PDF',
};

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/info" element={<InfoPage />} />
    </Routes>
  );
};

const HomePage = () => {
  const {
    step,
    resumeText,
    jdText,
    resumeFileB64,
    resumeFileType,
    resumeFileName,
    generatedResume,
    rewrittenFileB64,
    completedAgents,
    currentAgent,
    previewData,
    genError,
    setStep,
    setResumeUploaded,
    setJdText,
    setGenerating,
    addCompletedAgent,
    setCurrentAgent,
    setComplete,
    setPreviewData,
    setGenError,
    setConfirmResult,
    reset,
  } = useAppStore();

  // Transient loading flags — not persisted
  const [previewLoading, setPreviewLoading] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Which steps the user can jump to directly
  const accessibleSteps: number[] = [1];
  if (resumeText) accessibleSteps.push(2);
  if (previewData) accessibleSteps.push(4);
  if (generatedResume) accessibleSteps.push(5);

  const handleStepClick = (s: number) => {
    if (accessibleSteps.includes(s) && s !== step) {
      abortRef.current?.abort();
      setStep(s as 1 | 2 | 3 | 4 | 5);
    }
  };

  const handleResumeUploaded = (text: string, fileB64?: string, fileType?: string, fileName?: string) => {
    setResumeUploaded(text, fileB64 ?? '', fileType ?? 'pdf', fileName ?? '');
  };

  const handleJdReady = useCallback(async (jd: string) => {
    setJdText(jd);
    setGenerating();

    abortRef.current = generateResumeStream(
      resumeText, jd, resumeFileB64, resumeFileType,
      {
        onAgentComplete: (evt) => addCompletedAgent(evt),
        onComplete: (data) => setComplete(data.resume, data.rewritten_file_b64),
        onError: (detail) => {
          setGenError(detail);
          setCurrentAgent(null);
          setStep(2);
        },
      },
    );
  }, [resumeText, resumeFileB64, resumeFileType, setJdText, setGenerating, addCompletedAgent, setComplete, setGenError, setCurrentAgent, setStep]);

  const handlePreviewFlow = useCallback((jd: string) => {
    setGenError(null);
    setJdText(jd);
    setGenerating(); // clears completedAgents, sets step 3
    setPreviewLoading(true);

    abortRef.current?.abort();
    abortRef.current = previewResumeStream(
      resumeText, jd,
      {
        onAgentComplete: (evt) => addCompletedAgent(evt),
        onComplete: (data) => {
          setPreviewLoading(false);
          setPreviewData(data);
        },
        onError: (detail) => {
          setPreviewLoading(false);
          setGenError(detail);
          setCurrentAgent(null);
          setStep(2);
        },
      },
    );
  }, [resumeText, setGenError, setJdText, setGenerating, addCompletedAgent, setPreviewData, setCurrentAgent, setStep]);

  const handleConfirm = useCallback(async (approved: TextReplacement[]) => {
    setConfirmLoading(true);
    try {
      const { rewritten_file_b64 } = await confirmResume(
        resumeText, approved, resumeFileB64, resumeFileType,
      );
      setConfirmResult(
        {
          name: '',
          skills: [],
          experience: [],
          education: [],
          certifications: [],
          ats_score_before: previewData?.ats_score_before ?? 0,
          ats_score: previewData?.ats_score ?? 0,
          matched_keywords: previewData?.matched_keywords ?? [],
        },
        rewritten_file_b64,
      );
    } catch (e: unknown) {
      setGenError(e instanceof Error ? e.message : 'Compilation failed.');
    } finally {
      setConfirmLoading(false);
    }
  }, [resumeText, resumeFileB64, resumeFileType, previewData, setConfirmResult, setGenError]);

  const handleStartOver = () => {
    abortRef.current?.abort();
    reset();
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 to-blue-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3 shadow-sm">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <span className="text-lg font-bold text-gray-800">Resume for ATS</span>
        <span className="text-xs text-gray-400 ml-1">AI-Powered Resume Tailor</span>
        <div className="flex-1" />
        <Link to="/info" className="text-sm text-blue-600 font-medium hover:underline">
          Pipeline Runs
        </Link>
      </header>

      <main className="flex-1 flex flex-col items-center py-10 px-4">
        <div className="w-full max-w-2xl">
          <StepIndicator
            currentStep={step}
            steps={STEPS}
            accessibleSteps={accessibleSteps}
            onStepClick={handleStepClick}
          />

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
            {step === 1 && <ResumeUpload onDone={handleResumeUploaded} />}

            {step === 2 && (
              <div className="space-y-4">
                <JDInput onDone={handleJdReady} onPreview={handlePreviewFlow} />
                {genError && (
                  <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
                    {genError}
                  </p>
                )}
              </div>
            )}

            {step === 3 && (
              <div className="flex flex-col items-center gap-6 py-10">
                <div className="w-14 h-14 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
                <div className="text-center space-y-1">
                  <p className="font-semibold text-gray-800">
                    {previewLoading ? 'Generating preview\u2026' : 'Tailoring your resume\u2026'}
                  </p>
                  <p className="text-sm text-gray-500">
                    {currentAgent
                      ? AGENT_LABELS[currentAgent] || currentAgent
                      : 'Starting pipeline\u2026'}
                  </p>
                </div>
                {/* Agent progress list */}
                {completedAgents.length > 0 && (
                  <div className="w-full max-w-md space-y-1 text-sm">
                    {completedAgents.map((evt, i) => (
                      <div key={i} className="flex items-center gap-2 text-green-700">
                        <svg className="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        <span>{AGENT_LABELS[evt.agent] || evt.agent}</span>
                        {evt.ats_score_before != null && (
                          <span className="text-gray-500 text-xs">ATS before: {evt.ats_score_before}%</span>
                        )}
                        {evt.ats_score != null && (
                          <span className="text-blue-600 text-xs font-semibold">ATS: {evt.ats_score}%</span>
                        )}
                        {evt.replacements_count != null && (
                          <span className="text-gray-500 text-xs">{evt.replacements_count} replacement{evt.replacements_count !== 1 ? 's' : ''}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {step === 4 && previewData && (
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-gray-800">Review Proposed Changes</h2>
                <DiffPreview
                  replacements={previewData.replacements}
                  atsScoreBefore={previewData.ats_score_before}
                  atsScore={previewData.ats_score}
                  matchedKeywords={previewData.matched_keywords}
                  stillMissing={previewData.still_missing_keywords}
                  onConfirm={handleConfirm}
                  onCancel={handleStartOver}
                  loading={confirmLoading}
                />
                {genError && (
                  <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
                    {genError}
                  </p>
                )}
              </div>
            )}

            {step === 5 && generatedResume && (
              <div className="space-y-6">
                <ResumePreview
                  resume={generatedResume}
                  onStartOver={handleStartOver}
                  rewrittenFileB64={rewrittenFileB64}
                  originalFileName={resumeFileName}
                />
                {/* Cover letter panel */}
                {resumeText && jdText && (
                  <CoverLetterPanel resumeText={resumeText} jdText={jdText} />
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
