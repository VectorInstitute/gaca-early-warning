"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, AlertCircle, Calendar, Activity } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { API_BASE_URL } from "../constants";
import { Header } from "../components/header";
import { CollapsedSidebar } from "../components/sidebar";

// ============================================================================
// Types
// ============================================================================

interface HorizonMetrics {
  rmse: number;
  mae: number;
  sample_count: number;
}

interface EvaluationMetrics {
  overall: {
    rmse: number;
    mae: number;
    sample_count: number;
  };
  by_horizon: Record<string, HorizonMetrics>;
}

interface StaticEvaluation {
  evaluation_period: { start: string; end: string };
  metrics: EvaluationMetrics;
  computed_at: string;
  message?: string;
}

interface DynamicEvaluation {
  evaluation_window: { start: string; end: string; days: number };
  metrics: EvaluationMetrics;
  computed_at: string;
  message?: string;
}

interface ChartDataPoint {
  horizon: number;
  RMSE: number;
  MAE: number;
}

// ============================================================================
// Components
// ============================================================================

const MetricCardSkeleton = () => (
  <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
    <div className="flex items-center justify-between mb-3">
      <div className="h-4 w-24 bg-slate-700/50 rounded animate-pulse" />
      <div className="w-9 h-9 bg-slate-700/50 rounded-lg animate-pulse" />
    </div>
    <div className="h-10 w-32 bg-slate-700/50 rounded animate-pulse" />
    <div className="h-3 w-40 bg-slate-700/50 rounded mt-3 animate-pulse" />
  </div>
);

const ChartSkeleton = () => (
  <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
    <div className="h-5 w-48 bg-slate-700/50 rounded animate-pulse mb-6" />
    <div className="h-[350px] bg-slate-700/30 rounded animate-pulse" />
  </div>
);

const MetricCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  gradient = "from-[#EB088A] to-[#313CFF]",
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: any;
  gradient?: string;
}) => (
  <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6 hover:border-[#EB088A]/50 transition-all">
    <div className="flex items-center justify-between mb-3">
      <span className="text-sm font-medium text-slate-400">{title}</span>
      <div className={`p-2 rounded-lg bg-gradient-to-br ${gradient}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
    </div>
    <div className="text-4xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
      {value}
    </div>
    <div className="text-sm text-slate-500 mt-2">{subtitle}</div>
  </div>
);

const ErrorChart = ({ data, title }: { data: ChartDataPoint[]; title: string }) => {
  // Calculate dynamic Y-axis domain with padding
  const maxRMSE = Math.max(...data.map((d) => d.RMSE));
  const maxMAE = Math.max(...data.map((d) => d.MAE));
  const maxValue = Math.max(maxRMSE, maxMAE);
  const yMax = Math.ceil(maxValue * 1.15); // Add 15% padding above max value

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6 hover:border-[#313CFF]/50 transition-all">
      <h3 className="text-lg font-semibold text-white mb-6">{title}</h3>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="horizon"
            stroke="#94a3b8"
            label={{
              value: "Forecast Horizon (hours)",
              position: "insideBottom",
              offset: -10,
              fill: "#94a3b8",
            }}
          />
          <YAxis
            stroke="#94a3b8"
            label={{
              value: "Error (°C)",
              angle: -90,
              position: "insideLeft",
              fill: "#94a3b8",
            }}
            domain={[0, yMax]}
            ticks={[0, 1, 2, 3, 4, 5].filter((v) => v <= yMax)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              color: "#fff",
            }}
          />
          <Legend
            verticalAlign="top"
            height={36}
            wrapperStyle={{ paddingBottom: "10px" }}
          />
          <Line
            type="monotone"
            dataKey="RMSE"
            stroke="#EB088A"
            strokeWidth={3}
            dot={{ fill: "#EB088A", r: 5 }}
            activeDot={{ r: 7 }}
            name="RMSE (°C)"
          />
          <Line
            type="monotone"
            dataKey="MAE"
            stroke="#313CFF"
            strokeWidth={3}
            dot={{ fill: "#313CFF", r: 5 }}
            activeDot={{ r: 7 }}
            name="MAE (°C)"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-slate-400 text-center">
        Error increases with longer forecast horizons
      </div>
    </div>
  );
};

const ErrorMessage = ({ message }: { message: string }) => (
  <div className="bg-red-900/20 border border-red-700/50 rounded-xl p-6 flex items-start gap-3">
    <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
    <div>
      <h3 className="font-semibold text-red-300">Error</h3>
      <p className="text-red-400 text-sm mt-1">{message}</p>
    </div>
  </div>
);

const WarningMessage = ({ message }: { message: string }) => (
  <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-xl p-6">
    <p className="text-yellow-300">{message}</p>
  </div>
);

// ============================================================================
// Main Component
// ============================================================================

export default function EvaluationPage() {
  const [staticEval, setStaticEval] = useState<StaticEvaluation | null>(null);
  const [dynamicEval, setDynamicEval] = useState<DynamicEvaluation | null>(null);
  const [loadingStatic, setLoadingStatic] = useState(true);
  const [loadingDynamic, setLoadingDynamic] = useState(true);
  const [errorStatic, setErrorStatic] = useState<string | null>(null);
  const [errorDynamic, setErrorDynamic] = useState<string | null>(null);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;

    const fetchStaticEvaluation = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/evaluation/static`);
        if (!response.ok) throw new Error("Failed to fetch static evaluation");
        const data = await response.json();
        setStaticEval(data);
        setErrorStatic(null);
      } catch (error) {
        setErrorStatic(error instanceof Error ? error.message : "Unknown error");
      } finally {
        setLoadingStatic(false);
      }
    };

    const fetchDynamicEvaluation = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/evaluation/dynamic`);
        if (!response.ok) throw new Error("Failed to fetch dynamic evaluation");
        const data = await response.json();
        setDynamicEval(data);
        setErrorDynamic(null);
      } catch (error) {
        setErrorDynamic(error instanceof Error ? error.message : "Unknown error");
      } finally {
        setLoadingDynamic(false);
      }
    };

    fetchStaticEvaluation();
    fetchDynamicEvaluation();
  }, []);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const prepareChartData = (
    by_horizon: Record<string, HorizonMetrics>
  ): ChartDataPoint[] => {
    return Object.entries(by_horizon)
      .map(([horizon, metrics]) => ({
        horizon: parseInt(horizon),
        RMSE: parseFloat(metrics.rmse.toFixed(3)),
        MAE: parseFloat(metrics.mae.toFixed(3)),
      }))
      .sort((a, b) => a.horizon - b.horizon);
  };

  const renderStaticSection = () => {
    if (loadingStatic) {
      return (
        <div className="space-y-6">
          <div className="bg-[#EB088A]/10 border border-[#EB088A]/30 rounded-xl p-4">
            <div className="h-4 w-64 bg-slate-700/50 rounded animate-pulse" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </div>
          <ChartSkeleton />
        </div>
      );
    }

    if (errorStatic) {
      return <ErrorMessage message={errorStatic} />;
    }

    if (staticEval?.message) {
      return <WarningMessage message={staticEval.message} />;
    }

    if (!staticEval) return null;

    return (
      <AnimatePresence mode="wait">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          <div className="bg-[#EB088A]/10 border border-[#EB088A]/30 rounded-xl p-4">
            <div className="flex flex-wrap gap-6 text-sm">
              <div>
                <span className="text-slate-400">Period:</span>{" "}
                <span className="font-semibold text-white">
                  {formatDate(staticEval.evaluation_period.start)} to{" "}
                  {formatDate(staticEval.evaluation_period.end)}
                </span>
              </div>
              <div>
                <span className="text-slate-400">Samples:</span>{" "}
                <span className="font-semibold text-white">
                  {staticEval.metrics.overall.sample_count.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricCard
              title="Overall RMSE"
              value={`${staticEval.metrics.overall.rmse.toFixed(3)}°C`}
              subtitle="Root Mean Squared Error"
              icon={TrendingUp}
              gradient="from-[#EB088A] to-pink-600"
            />
            <MetricCard
              title="Overall MAE"
              value={`${staticEval.metrics.overall.mae.toFixed(3)}°C`}
              subtitle="Mean Absolute Error"
              icon={Activity}
              gradient="from-[#313CFF] to-blue-600"
            />
          </div>

          <ErrorChart
            data={prepareChartData(staticEval.metrics.by_horizon)}
            title="Error vs Forecast Horizon"
          />
        </motion.div>
      </AnimatePresence>
    );
  };

  const renderDynamicSection = () => {
    if (loadingDynamic) {
      return (
        <div className="space-y-6">
          <div className="bg-[#313CFF]/10 border border-[#313CFF]/30 rounded-xl p-4">
            <div className="h-4 w-64 bg-slate-700/50 rounded animate-pulse" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </div>
          <ChartSkeleton />
        </div>
      );
    }

    if (errorDynamic) {
      return <ErrorMessage message={errorDynamic} />;
    }

    if (dynamicEval?.message) {
      return <WarningMessage message={dynamicEval.message} />;
    }

    if (!dynamicEval) return null;

    return (
      <AnimatePresence mode="wait">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          <div className="bg-[#313CFF]/10 border border-[#313CFF]/30 rounded-xl p-4">
            <div className="flex flex-wrap gap-6 text-sm">
              <div>
                <span className="text-slate-400">Window:</span>{" "}
                <span className="font-semibold text-white">
                  Last {dynamicEval.evaluation_window.days} days
                </span>
              </div>
              <div>
                <span className="text-slate-400">Samples:</span>{" "}
                <span className="font-semibold text-white">
                  {dynamicEval.metrics.overall.sample_count.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricCard
              title="Overall RMSE"
              value={`${dynamicEval.metrics.overall.rmse.toFixed(3)}°C`}
              subtitle="Root Mean Squared Error"
              icon={TrendingUp}
              gradient="from-[#EB088A] to-pink-600"
            />
            <MetricCard
              title="Overall MAE"
              value={`${dynamicEval.metrics.overall.mae.toFixed(3)}°C`}
              subtitle="Mean Absolute Error"
              icon={Activity}
              gradient="from-[#313CFF] to-blue-600"
            />
          </div>

          <ErrorChart
            data={prepareChartData(dynamicEval.metrics.by_horizon)}
            title="Error vs Forecast Horizon"
          />
        </motion.div>
      </AnimatePresence>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex">
      <CollapsedSidebar />

      <div className="flex-1 ml-16">
        <Header />

        <div className="max-w-[1920px] mx-auto p-6">
          {/* Page Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mb-8"
          >
            <div className="flex items-center gap-4 mb-2">
              <div className="p-3 rounded-xl bg-gradient-to-br from-[#EB088A] to-[#313CFF]">
                <Activity className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold text-white">Model Evaluation</h1>
                <p className="text-slate-400 mt-1">
                  Performance metrics comparing predictions against NOAA ground truth
                </p>
              </div>
            </div>
          </motion.div>

          {/* Static Validation Section */}
          <section className="mb-12">
            <div className="flex items-center gap-3 mb-6">
              <Calendar className="w-6 h-6 text-[#EB088A]" />
              <h2 className="text-2xl font-bold text-white">
                Static Validation Period
              </h2>
            </div>
            {renderStaticSection()}
          </section>

          {/* Dynamic Evaluation Section */}
          <section>
            <div className="flex items-center gap-3 mb-6">
              <Activity className="w-6 h-6 text-[#313CFF]" />
              <h2 className="text-2xl font-bold text-white">
                Dynamic Evaluation (Rolling 30-Day Window)
              </h2>
            </div>
            {renderDynamicSection()}
          </section>
        </div>
      </div>
    </div>
  );
}
