import { useState, Fragment } from "react";
import {
  CheckCircle,
  XCircle,
  Clock,
  Database,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
} from "lucide-react";

interface ForecastLog {
  run_timestamp: string;
  status: string;
  duration_seconds: number;
  records_generated: number;
  noaa_data_timestamp: string | null;
  error_message: string | null;
}

type SortField = "timestamp" | "duration" | "records";
type SortOrder = "asc" | "desc";

interface LogTableProps {
  logs: ForecastLog[];
  loading: boolean;
  sortField?: SortField;
  sortOrder?: SortOrder;
  onSort?: (field: SortField) => void;
}

interface SortableHeaderProps {
  field: SortField;
  label: string;
  sortField?: SortField;
  sortOrder?: SortOrder;
  onSort?: (field: SortField) => void;
}

/**
 * Sortable table header component
 */
function SortableHeader({
  field,
  label,
  sortField,
  sortOrder,
  onSort,
}: SortableHeaderProps) {
  const isActive = sortField === field;
  const isAscending = isActive && sortOrder === "asc";

  return (
    <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
      <button
        onClick={() => onSort?.(field)}
        className="flex items-center gap-2 hover:text-slate-200 transition-colors group"
      >
        {label}
        {isActive ? (
          isAscending ? (
            <ChevronUp className="w-4 h-4 text-blue-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-blue-400" />
          )
        ) : (
          <ArrowUpDown className="w-4 h-4 opacity-0 group-hover:opacity-50 transition-opacity" />
        )}
      </button>
    </th>
  );
}

/**
 * Table component for displaying forecast run logs
 */
export function LogTable({
  logs,
  loading,
  sortField,
  sortOrder,
  onSort,
}: LogTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const toggleRowExpansion = (timestamp: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(timestamp)) {
        next.delete(timestamp);
      } else {
        next.add(timestamp);
      }
      return next;
    });
  };
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = (seconds % 60).toFixed(0);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const formatRecords = (count: number) => {
    return count.toLocaleString();
  };

  if (loading && logs.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12">
        <div className="flex items-center justify-center">
          <div className="flex items-center gap-3 text-slate-400">
            <div className="w-5 h-5 border-2 border-slate-600 border-t-slate-400 rounded-full animate-spin" />
            <span>Loading logs...</span>
          </div>
        </div>
      </div>
    );
  }

  if (logs.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
        <Clock className="w-16 h-16 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400">No forecast runs logged yet</p>
        <p className="text-sm text-slate-500 mt-2">
          Logs will appear here when forecasts complete
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-slate-800/90 backdrop-blur-sm z-10">
            <tr className="border-b border-slate-700/50">
              <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                Status
              </th>
              <SortableHeader
                field="timestamp"
                label="Run Time"
                sortField={sortField}
                sortOrder={sortOrder}
                onSort={onSort}
              />
              <SortableHeader
                field="duration"
                label="Duration"
                sortField={sortField}
                sortOrder={sortOrder}
                onSort={onSort}
              />
              <SortableHeader
                field="records"
                label="Records"
                sortField={sortField}
                sortOrder={sortOrder}
                onSort={onSort}
              />
              <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                NOAA Data Time
              </th>
              <th className="px-6 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                {/* Expand column */}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/30">
            {logs.map((log, index) => {
              const isExpanded = expandedRows.has(log.run_timestamp);
              const hasError = log.status === "error" && log.error_message;

              return (
                <Fragment key={`${log.run_timestamp}-${index}`}>
                  <tr
                    className={`hover:bg-slate-700/20 transition-colors ${
                      hasError ? "cursor-pointer" : ""
                    }`}
                    onClick={() => hasError && toggleRowExpansion(log.run_timestamp)}
                  >
                    {/* Status */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      {log.status === "success" ? (
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          <span className="text-sm font-medium text-green-400">
                            Success
                          </span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <XCircle className="w-5 h-5 text-red-400" />
                          <span className="text-sm font-medium text-red-400">
                            Error
                          </span>
                        </div>
                      )}
                    </td>

                    {/* Timestamp */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-slate-500" />
                        <span className="text-sm text-slate-300">
                          {formatTimestamp(log.run_timestamp)}
                        </span>
                      </div>
                    </td>

                    {/* Duration */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-slate-300">
                        {formatDuration(log.duration_seconds)}
                      </div>
                    </td>

                    {/* Records */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4 text-slate-500" />
                        <span className="text-sm text-slate-300">
                          {formatRecords(log.records_generated)}
                        </span>
                      </div>
                    </td>

                    {/* NOAA Data Timestamp */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      {log.noaa_data_timestamp ? (
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-sm text-slate-300">
                            {formatTimestamp(log.noaa_data_timestamp)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-xs text-slate-500">N/A</span>
                      )}
                    </td>

                    {/* Expand Button */}
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      {hasError && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleRowExpansion(log.run_timestamp);
                          }}
                          className="text-slate-400 hover:text-slate-200 transition-colors"
                        >
                          {isExpanded ? (
                            <ChevronUp className="w-5 h-5" />
                          ) : (
                            <ChevronDown className="w-5 h-5" />
                          )}
                        </button>
                      )}
                    </td>
                  </tr>

                  {/* Expandable Error Row */}
                  {hasError && isExpanded && (
                    <tr>
                      <td colSpan={6} className="px-6 py-4 bg-red-500/5">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                          <div className="flex-1">
                            <p className="text-sm font-medium text-red-400 mb-2">
                              Error Details
                            </p>
                            <div className="bg-slate-900/50 rounded-lg p-3 border border-red-500/20">
                              <code className="text-sm text-red-300 font-mono whitespace-pre-wrap break-words">
                                {log.error_message}
                              </code>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
