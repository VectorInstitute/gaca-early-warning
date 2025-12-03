import { motion } from "framer-motion";

/**
 * Loading overlay with animated pyramid stacking effect
 * Positioned to not cover map controls
 */
export function LoadingOverlay() {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
      <div className="bg-slate-800/95 backdrop-blur-md rounded-2xl border border-slate-700/50 p-8 shadow-2xl pointer-events-auto">
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

        {/* Loading Message */}
        <div className="text-center space-y-3">
          <h3 className="text-lg font-semibold text-white">Loading Predictions</h3>
          <p className="text-sm text-slate-400">
            Fetching latest forecast data from BigQuery
          </p>
        </div>

        {/* Footer */}
        <motion.p
          className="text-xs text-slate-500 mt-6 text-center"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          This should only take a moment
        </motion.p>
      </div>
    </div>
  );
}
