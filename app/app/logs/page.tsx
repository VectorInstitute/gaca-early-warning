"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { motion } from "framer-motion";
import {
  RefreshCw,
  AlertCircle,
  Terminal,
  Filter,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { Header, CollapsedSidebar } from "../components";
import { LogTable } from "../components/log-table";
import { API_BASE_URL } from "../constants";

interface ForecastLog {
  run_timestamp: string;
  status: string;
  duration_seconds: number;
  records_generated: number;
  noaa_data_timestamp: string | null;
  error_message: string | null;
}

interface LogsResponse {
  count: number;
  logs: ForecastLog[];
  setup_required?: boolean;
  setup_command?: string;
}

type StatusFilter = "all" | "success" | "error";
type SortField = "timestamp" | "duration" | "records";
type SortOrder = "asc" | "desc";

export default function LogsPage() {
  const [logs, setLogs] = useState<ForecastLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [setupRequired, setSetupRequired] = useState(false);
  const [setupCommand, setSetupCommand] = useState<string>("");
  const [lastFetchTime, setLastFetchTime] = useState<Date | null>(null);

  // Filtering and pagination state
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortField, setSortField] = useState<SortField>("timestamp");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const fetchLogs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch 500 logs from backend for client-side pagination
      const response = await fetch(`${API_BASE_URL}/logs/forecast-runs?limit=500`);

      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.status}`);
      }

      const data: LogsResponse = await response.json();

      if (data.setup_required) {
        setSetupRequired(true);
        setSetupCommand(data.setup_command || "");
        setLogs([]);
      } else {
        setSetupRequired(false);
        setLogs(data.logs || []);
      }

      setLastFetchTime(new Date());
    } catch (err) {
      console.error("Error fetching logs:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch logs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLogs();

    // Auto-refresh every 2 minutes
    const intervalId = setInterval(fetchLogs, 2 * 60 * 1000);

    return () => clearInterval(intervalId);
  }, [fetchLogs]);

  // Filter, sort, and paginate logs
  const { paginatedLogs, totalPages, totalFilteredLogs } = useMemo(() => {
    // Filter by status
    let filtered = logs;
    if (statusFilter !== "all") {
      filtered = logs.filter((log) => log.status === statusFilter);
    }

    // Sort logs
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case "timestamp":
          comparison =
            new Date(a.run_timestamp).getTime() - new Date(b.run_timestamp).getTime();
          break;
        case "duration":
          comparison = a.duration_seconds - b.duration_seconds;
          break;
        case "records":
          comparison = a.records_generated - b.records_generated;
          break;
      }

      return sortOrder === "asc" ? comparison : -comparison;
    });

    // Paginate
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const paginated = sorted.slice(startIndex, endIndex);

    return {
      paginatedLogs: paginated,
      totalPages: Math.ceil(sorted.length / pageSize),
      totalFilteredLogs: sorted.length,
    };
  }, [logs, statusFilter, sortField, sortOrder, currentPage, pageSize]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [statusFilter, sortField, sortOrder, pageSize]);

  const formatLastUpdate = () => {
    if (!lastFetchTime) return "";
    const now = new Date();
    const diff = Math.floor((now.getTime() - lastFetchTime.getTime()) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Toggle sort order
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      // New field, default to descending
      setSortField(field);
      setSortOrder("desc");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex">
      <CollapsedSidebar />

      <div className="flex-1 ml-16">
        <Header />

        <div className="max-w-[1920px] mx-auto p-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-4"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-white mb-1">
                  Forecast Run Logs
                </h1>
                <p className="text-sm text-slate-400">
                  Automated forecast execution history
                </p>
              </div>

              <div className="flex items-center gap-4">
                {lastFetchTime && (
                  <span className="text-xs text-slate-500">
                    Updated {formatLastUpdate()}
                  </span>
                )}
                <button
                  onClick={fetchLogs}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg border border-slate-700 transition-all"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                  Refresh
                </button>
              </div>
            </div>

            {/* Setup Required Notice */}
            {setupRequired && (
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-5 flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-amber-300 font-semibold mb-2">
                    BigQuery Setup Required
                  </p>
                  <p className="text-amber-200/80 text-sm mb-3">
                    The forecast_runs table hasn't been created yet. Run the setup
                    script to create all required BigQuery tables:
                  </p>
                  <div className="flex items-center gap-2 bg-slate-900/50 p-3 rounded border border-slate-700/50">
                    <Terminal className="w-4 h-4 text-slate-400" />
                    <code className="text-sm text-emerald-400 font-mono">
                      {setupCommand}
                    </code>
                  </div>
                  <p className="text-amber-200/60 text-xs mt-3">
                    After running the setup script, logs from forecast runs will appear
                    here automatically.
                  </p>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && !setupRequired && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-red-400 font-medium">Error loading logs</p>
                  <p className="text-red-300/80 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Filters and Controls */}
            {!setupRequired && logs.length > 0 && (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  {/* Left: Filters */}
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Filter className="w-4 h-4 text-slate-400" />
                      <span className="text-sm text-slate-400 font-medium">
                        Status:
                      </span>
                      <div className="flex gap-2">
                        {(["all", "success", "error"] as StatusFilter[]).map(
                          (status) => (
                            <button
                              key={status}
                              onClick={() => setStatusFilter(status)}
                              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                                statusFilter === status
                                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                  : "bg-slate-700/30 text-slate-400 border border-slate-600/30 hover:bg-slate-700/50"
                              }`}
                            >
                              {status.charAt(0).toUpperCase() + status.slice(1)}
                            </button>
                          )
                        )}
                      </div>
                    </div>

                    <div className="h-6 w-px bg-slate-600/50" />

                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">Per page:</span>
                      <select
                        value={pageSize}
                        onChange={(e) => setPageSize(Number(e.target.value))}
                        className="px-2 py-1 text-xs bg-slate-700/50 text-slate-300 border border-slate-600/50 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                      >
                        <option value={10}>10</option>
                        <option value={25}>25</option>
                        <option value={50}>50</option>
                        <option value={100}>100</option>
                      </select>
                    </div>
                  </div>

                  {/* Right: Results count */}
                  <div className="text-sm text-slate-400">
                    Showing {(currentPage - 1) * pageSize + 1}-
                    {Math.min(currentPage * pageSize, totalFilteredLogs)} of{" "}
                    {totalFilteredLogs} logs
                    {statusFilter !== "all" && ` (filtered from ${logs.length})`}
                  </div>
                </div>
              </div>
            )}

            {/* Log Table */}
            {!setupRequired && (
              <LogTable
                logs={paginatedLogs}
                loading={loading}
                sortField={sortField}
                sortOrder={sortOrder}
                onSort={handleSort}
              />
            )}

            {/* Pagination */}
            {!setupRequired && totalPages > 1 && (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-slate-400">
                    Page {currentPage} of {totalPages}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage(1)}
                      disabled={currentPage === 1}
                      className="px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-700 disabled:bg-slate-700/20 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded-md border border-slate-600/50 transition-all"
                    >
                      First
                    </button>
                    <button
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-700 disabled:bg-slate-700/20 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded-md border border-slate-600/50 transition-all flex items-center gap-1"
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </button>
                    <button
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-700 disabled:bg-slate-700/20 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded-md border border-slate-600/50 transition-all flex items-center gap-1"
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setCurrentPage(totalPages)}
                      disabled={currentPage === totalPages}
                      className="px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-700 disabled:bg-slate-700/20 disabled:cursor-not-allowed text-slate-300 disabled:text-slate-600 rounded-md border border-slate-600/50 transition-all"
                    >
                      Last
                    </button>
                  </div>

                  <div className="text-sm text-slate-400">
                    Jump to:
                    <input
                      type="number"
                      min={1}
                      max={totalPages}
                      value={currentPage}
                      onChange={(e) => {
                        const page = Number(e.target.value);
                        if (page >= 1 && page <= totalPages) {
                          setCurrentPage(page);
                        }
                      }}
                      className="ml-2 w-16 px-2 py-1 text-sm bg-slate-700/50 text-slate-300 border border-slate-600/50 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    />
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
