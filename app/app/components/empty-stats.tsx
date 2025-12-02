import { Gauge } from "lucide-react";

/**
 * Empty state for statistics panel
 */
export function EmptyStats() {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
      <Gauge className="w-16 h-16 text-slate-600 mx-auto mb-4" />
      <div className="space-y-2">
        <p className="text-slate-300 font-medium">Waiting for forecast data</p>
        <p className="text-sm text-slate-400">
          Temperature statistics and visualizations will appear when the next hourly
          forecast completes
        </p>
        <p className="text-xs text-slate-500 mt-3">
          Forecasts run automatically every hour at :15
        </p>
      </div>
    </div>
  );
}
