import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ResumeData } from '../types/resume';
import type { SSEAgentEvent, PreviewResponse } from '../api/client';

export type Step = 1 | 2 | 3 | 4 | 5;

export interface AppState {
  step: Step;
  resumeText: string;
  jdText: string;
  resumeFileB64: string;
  resumeFileType: string;
  resumeFileName: string;
  generatedResume: ResumeData | null;
  rewrittenFileB64: string;
  completedAgents: SSEAgentEvent[];
  currentAgent: string | null;
  previewData: PreviewResponse | null;
  genError: string | null;

  // Actions
  setStep: (step: Step) => void;
  setResumeUploaded: (text: string, fileB64: string, fileType: string, fileName: string) => void;
  setJdText: (jd: string) => void;
  setGenerating: () => void;
  addCompletedAgent: (evt: SSEAgentEvent) => void;
  setCurrentAgent: (agent: string | null) => void;
  setComplete: (resume: ResumeData, rewrittenFileB64: string) => void;
  setPreviewData: (data: PreviewResponse) => void;
  setGenError: (error: string | null) => void;
  setConfirmResult: (resume: ResumeData, rewrittenFileB64: string) => void;
  reset: () => void;
}

const initialState = {
  step: 1 as Step,
  resumeText: '',
  jdText: '',
  resumeFileB64: '',
  resumeFileType: 'pdf',
  resumeFileName: '',
  generatedResume: null,
  rewrittenFileB64: '',
  completedAgents: [],
  currentAgent: null,
  previewData: null,
  genError: null,
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      ...initialState,

      setStep: (step) => set({ step }),

      setResumeUploaded: (text, fileB64, fileType, fileName) =>
        set({
          resumeText: text,
          resumeFileB64: fileB64,
          resumeFileType: fileType,
          resumeFileName: fileName,
          step: 2,
        }),

      setJdText: (jd) => set({ jdText: jd }),

      setGenerating: () =>
        set({
          genError: null,
          completedAgents: [],
          currentAgent: null,
          step: 3,
        }),

      addCompletedAgent: (evt) =>
        set((s) => ({
          completedAgents: [...s.completedAgents, evt],
          currentAgent: evt.agent,
        })),

      setCurrentAgent: (agent) => set({ currentAgent: agent }),

      setComplete: (resume, rewrittenFileB64) =>
        set({ generatedResume: resume, rewrittenFileB64, currentAgent: null, step: 5 }),

      setPreviewData: (data) => set({ previewData: data, step: 4 }),

      setGenError: (error) => set({ genError: error }),

      setConfirmResult: (resume, rewrittenFileB64) =>
        set({ generatedResume: resume, rewrittenFileB64, step: 5 }),

      reset: () => set(initialState),
    }),
    {
      name: 'resume-ats-app',
      // Skip large binary blobs from being persisted — they can be multi-MB
      // and will exceed localStorage quota. Everything else is persisted.
      partialize: (state) => {
        const { resumeFileB64, rewrittenFileB64, ...rest } = state;
        void resumeFileB64;
        void rewrittenFileB64;
        return rest;
      },
    },
  ),
);
