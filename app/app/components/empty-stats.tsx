import { Gauge } from "lucide-react";

/**
 * Empty state for statistics panel
 */
export function EmptyStats() {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
      <Gauge className="w-16 h-16 text-slate-600 mx-auto mb-4" />
      <p className="text-slate-400">
        Statistics will appear here after running forecast
      </p>
    </div>
  );
}
