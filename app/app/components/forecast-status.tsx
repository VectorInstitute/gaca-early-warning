import { Clock, RefreshCw } from "lucide-react";
import type { ForecastStatus } from "../hooks/use-auto-predictions";

interface ForecastStatusProps {
  status: ForecastStatus | null;
  lastFetchTime: Date | null;
  onRefresh: () => void;
  loading: boolean;
}

/**
 * Compact forecast status display showing live-update state and last refresh time
 */
export function ForecastStatusDisplay({
  status,
  lastFetchTime,
  onRefresh,
  loading,
}: ForecastStatusProps) {
  const formatRelativeTime = (date: Date | null) => {
    if (!date) return null;
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  const isRunning = status?.scheduler?.is_running || false;
  const relativeTime = formatRelativeTime(lastFetchTime);

  return (
    <div className="absolute top-4 left-4">
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-700/50 shadow-lg">
        <div className="flex items-center gap-2.5 px-3 py-2">
          {/* Live indicator dot */}
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                isRunning ? "bg-green-400 animate-pulse" : "bg-[#E6007E]"
              }`}
            />
            <span className="text-xs font-semibold text-white">
              {isRunning ? "Running…" : "Live Updates"}
            </span>
          </div>

          {/* Divider */}
          <div className="h-4 w-px bg-slate-700/50" />

          {/* Last updated time */}
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Clock className="w-3 h-3 flex-shrink-0" />
            <span>
              {relativeTime ? `Updated ${relativeTime}` : "Waiting for first run"}
            </span>
          </div>

          {/* Divider */}
          <div className="h-4 w-px bg-slate-700/50" />

          {/* Refresh button */}
          <button
            onClick={onRefresh}
            disabled={loading}
            className="hover:bg-slate-800/80 disabled:opacity-50 disabled:cursor-not-allowed rounded p-1 transition-all group"
            title="Refresh predictions now"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 text-[#E6007E] ${
                loading
                  ? "animate-spin"
                  : "group-hover:rotate-180 transition-transform duration-500"
              }`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
