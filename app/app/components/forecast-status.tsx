import { Activity, Clock, RefreshCw } from "lucide-react";
import type { ForecastStatus } from "../hooks/use-auto-predictions";

interface ForecastStatusProps {
  status: ForecastStatus | null;
  lastFetchTime: Date | null;
  onRefresh: () => void;
  loading: boolean;
}

/**
 * Compact unified forecast status display with inline refresh
 */
export function ForecastStatusDisplay({
  status,
  lastFetchTime,
  onRefresh,
  loading,
}: ForecastStatusProps) {
  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return null;
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  const formatRelativeTime = (date: Date | null) => {
    if (!date) return null;
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    return `${Math.floor(diff / 3600)}h`;
  };

  const isRunning = status?.scheduler?.is_running || false;
  const nextRun = status?.scheduler?.next_scheduled_run;
  // Use scheduler timestamp first, fallback to last forecast from BigQuery
  const lastRun =
    status?.scheduler?.last_run_timestamp || status?.last_forecast?.run_timestamp;
  const relativeTime = formatRelativeTime(lastFetchTime);

  return (
    <div className="absolute top-4 left-4">
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-700/50 shadow-lg">
        <div className="flex items-center gap-3 px-3 py-2">
          {/* Status Icon & Text */}
          <div className="flex items-center gap-2">
            <Activity
              className={`w-3.5 h-3.5 ${isRunning ? "text-green-400 animate-pulse" : "text-blue-400"}`}
            />
            <div className="text-xs text-white font-medium">
              {isRunning ? "Running" : "Auto"}
            </div>
          </div>

          {/* Divider */}
          <div className="h-4 w-px bg-slate-700/50" />

          {/* Time Info */}
          <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
            {lastRun ? (
              <>
                <Clock className="w-3 h-3" />
                {relativeTime && <span>{relativeTime}</span>}
                {nextRun && !isRunning && (
                  <>
                    <span className="text-slate-600">•</span>
                    <span>Next {formatTime(nextRun)}</span>
                  </>
                )}
              </>
            ) : (
              <span>Waiting for first run</span>
            )}
          </div>

          {/* Divider */}
          <div className="h-4 w-px bg-slate-700/50" />

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={loading}
            className="hover:bg-slate-800/80 disabled:opacity-50 disabled:cursor-not-allowed rounded p-1 transition-all group"
            title="Refresh predictions"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 text-blue-400 ${loading ? "animate-spin" : "group-hover:rotate-180 transition-transform duration-500"}`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
