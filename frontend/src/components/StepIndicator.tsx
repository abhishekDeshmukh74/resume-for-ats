interface StepIndicatorProps {
  currentStep: number; // 1-based
  steps: string[];
  accessibleSteps?: number[];
  onStepClick?: (step: number) => void;
}

const StepIndicator = ({ currentStep, steps, accessibleSteps = [], onStepClick }: StepIndicatorProps) => (
  <div className="flex items-center justify-center gap-0 mb-8">
    {steps.map((label, i) => {
      const step = i + 1;
      const done = step < currentStep;
      const active = step === currentStep;
      const clickable = accessibleSteps.includes(step) && step !== currentStep && !!onStepClick;
      return (
        <div key={step} className="flex items-center">
          {/* Circle */}
          <div className="flex flex-col items-center">
            <div
              onClick={() => clickable && onStepClick?.(step)}
              className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-colors
                ${done ? 'bg-blue-600 border-blue-600 text-white' : ''}
                ${active ? 'bg-white border-blue-600 text-blue-600' : ''}
                ${!done && !active ? 'bg-white border-gray-300 text-gray-400' : ''}
                ${clickable ? 'cursor-pointer hover:opacity-80' : ''}`}
            >
              {done ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                step
              )}
            </div>
            <span
              className={`mt-1 text-xs whitespace-nowrap
                ${active ? 'text-blue-600 font-medium' : 'text-gray-400'}
                ${clickable ? 'cursor-pointer hover:text-blue-500' : ''}`}
              onClick={() => clickable && onStepClick?.(step)}
            >
              {label}
            </span>
          </div>
          {/* Connector */}
          {i < steps.length - 1 && (
            <div
              className={`h-0.5 w-16 mb-4 mx-1 transition-colors ${
                done ? 'bg-blue-600' : 'bg-gray-200'
              }`}
            />
          )}
        </div>
      );
    })}
  </div>
);

export default StepIndicator;
