import { Cloud } from "lucide-react";

/**
 * Empty state component for when no predictions are loaded
 */
export function EmptyState() {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      <div className="text-center text-slate-400 bg-slate-900/80 backdrop-blur-md rounded-2xl p-8 border border-slate-700/50">
        <Cloud className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p className="mb-2 text-lg">No predictions loaded</p>
        <p className="text-sm">Forecasts run hourly at :15 past the hour</p>
        <p className="text-xs text-slate-500 mt-2">
          Dashboard will auto-refresh when data is available
        </p>
      </div>
    </div>
  );
}
