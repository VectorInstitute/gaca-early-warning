import { Activity, Loader } from "lucide-react";

interface MapControlsProps {
  onRunForecast: () => void;
  loading: boolean;
}

/**
 * Map controls component for forecast triggering
 */
export function MapControls({ onRunForecast, loading }: MapControlsProps) {
  return (
    <div className="absolute top-4 left-4">
      <button
        onClick={onRunForecast}
        disabled={loading}
        className="bg-gradient-to-r from-[#EB088A] to-[#313CFF] hover:from-[#d00780] hover:to-[#2831e6] disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-md px-3 py-1.5 text-xs font-medium transition-all shadow-md hover:shadow-lg flex items-center gap-1.5"
      >
        {loading ? (
          <>
            <Loader className="w-3.5 h-3.5 animate-spin" />
            <span>Running</span>
          </>
        ) : (
          <>
            <Activity className="w-3.5 h-3.5" />
            <span>Run Forecast</span>
          </>
        )}
      </button>
    </div>
  );
}
