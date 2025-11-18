import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { Check } from "lucide-react";
import type { ProgressUpdate } from "../hooks/use-websocket-predictions";

const PIPELINE_STEPS = [
  { id: 1, label: "Loading model artifacts" },
  { id: 2, label: "Fetching NOAA meteorological data" },
  { id: 3, label: "Preprocessing features" },
  { id: 4, label: "Running model inference" },
];

interface LoadingOverlayProps {
  progress?: ProgressUpdate | null;
}

/**
 * Loading overlay with real-time progressive pipeline steps
 */
export function LoadingOverlay({ progress }: LoadingOverlayProps) {
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  // Update completed steps when progress changes
  useEffect(() => {
    if (!progress) {
      return;
    }

    const currentStepIndex = PIPELINE_STEPS.findIndex(
      (step) => step.label === progress.step
    );

    // Reset completed steps if we're back at the first step with in_progress status
    // This indicates a new inference run has started
    if (currentStepIndex === 0 && progress.status === "in_progress") {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCompletedSteps(new Set());
      return;
    }

    if (currentStepIndex >= 0 && progress.status === "completed") {
      setCompletedSteps((prev) => new Set([...prev, currentStepIndex]));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress]);

  // Determine current active step
  const currentStepIndex = progress
    ? PIPELINE_STEPS.findIndex((step) => step.label === progress.step)
    : -1;

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm z-20">
      <div className="bg-slate-800/90 backdrop-blur-md rounded-2xl border border-slate-700/50 p-8 shadow-2xl max-w-md w-full mx-4">
        {/* Pyramid Stacking Animation */}
        <div className="relative w-24 h-24 mx-auto mb-6">
          {/* Bottom layer - 3 blocks */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex gap-1">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={`bottom-${i}`}
                className="w-6 h-6 bg-gradient-to-br from-[#EB088A] to-[#313CFF] rounded-sm"
                initial={{ y: -50, opacity: 0 }}
                animate={{
                  y: 0,
                  opacity: 1,
                }}
                transition={{
                  duration: 0.5,
                  delay: i * 0.15,
                  repeat: Infinity,
                  repeatDelay: 1.5,
                }}
              />
            ))}
          </div>

          {/* Middle layer - 2 blocks */}
          <div className="absolute bottom-7 left-1/2 -translate-x-1/2 flex gap-1">
            {[0, 1].map((i) => (
              <motion.div
                key={`middle-${i}`}
                className="w-6 h-6 bg-gradient-to-br from-[#8A25C9] to-[#313CFF] rounded-sm"
                initial={{ y: -50, opacity: 0 }}
                animate={{
                  y: 0,
                  opacity: 1,
                }}
                transition={{
                  duration: 0.5,
                  delay: 0.45 + i * 0.15,
                  repeat: Infinity,
                  repeatDelay: 1.5,
                }}
              />
            ))}
          </div>

          {/* Top layer - 1 block */}
          <motion.div
            className="absolute bottom-14 left-1/2 -translate-x-1/2 w-6 h-6 bg-gradient-to-br from-[#EB088A] to-[#8A25C9] rounded-sm"
            initial={{ y: -50, opacity: 0 }}
            animate={{
              y: 0,
              opacity: 1,
            }}
            transition={{
              duration: 0.5,
              delay: 0.75,
              repeat: Infinity,
              repeatDelay: 1.5,
            }}
          />

          {/* Ping effect at the top */}
          <motion.div
            className="absolute bottom-14 left-1/2 -translate-x-1/2 w-6 h-6 bg-[#EB088A] rounded-sm"
            animate={{
              scale: [1, 2, 2],
              opacity: [0.5, 0, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              repeatDelay: 0.75,
            }}
          />
        </div>

        {/* Pipeline Steps */}
        <div className="space-y-3">
          {PIPELINE_STEPS.map((step, index) => {
            const isCompleted = completedSteps.has(index);
            const isActive =
              currentStepIndex === index && progress?.status === "in_progress";
            const _isPending = !isCompleted && !isActive;

            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3"
              >
                {/* Status Icon */}
                <div
                  className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center transition-all ${
                    isCompleted
                      ? "bg-[#EB088A]"
                      : isActive
                        ? "bg-[#313CFF] animate-pulse"
                        : "bg-slate-700"
                  }`}
                >
                  <AnimatePresence mode="wait">
                    {isCompleted ? (
                      <motion.div
                        key="check"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <Check className="w-4 h-4 text-white" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="number"
                        className={`text-xs font-bold ${
                          isActive ? "text-white" : "text-slate-500"
                        }`}
                      >
                        {index + 1}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Step Label */}
                <div
                  className={`flex-1 text-sm transition-colors ${
                    isCompleted
                      ? "text-slate-400 line-through"
                      : isActive
                        ? "text-white font-medium"
                        : "text-slate-500"
                  }`}
                >
                  {step.label}
                </div>

                {/* Active Step Spinner */}
                {isActive && (
                  <motion.div
                    className="w-4 h-4 border-2 border-[#313CFF] border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Footer */}
        <motion.p
          className="text-xs text-slate-500 mt-6 text-center"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          This may take a few minutes
        </motion.p>
      </div>
    </div>
  );
}
