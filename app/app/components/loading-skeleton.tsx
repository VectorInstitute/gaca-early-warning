import { motion } from "framer-motion";

/**
 * Loading skeleton for statistics panel
 */
export function LoadingSkeleton() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col gap-4"
    >
      <div className="grid grid-cols-2 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4"
          >
            <div className="h-4 bg-slate-700 rounded w-1/2 mb-3 animate-pulse" />
            <div className="h-8 bg-slate-700 rounded w-3/4 animate-pulse" />
          </div>
        ))}
      </div>
    </motion.div>
  );
}
